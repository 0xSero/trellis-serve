"""RAM tier for the file-backed (HMM) expert store: keep the most-routed experts locked in RAM.

The GPU reads the store through HMM. A record whose pages are resident streams at pinned speed once the GPU has mapped
it; a record that is not resident is faulted in by the GPU from the file at a fraction of the drive's speed, and its
pages must be mapped again. GPU accesses do not set the CPU page tables' accessed bits, so the kernel's LRU sees the
hottest experts as unused and evicts them first. The pinner fixes the RAM tier explicitly:

  * decode routing is counted per (layer, expert) on the device (`OffloadRuntime.route_counts`, one index_add_ per
    layer, CUDA-graph safe), and read every `interval` seconds on a side stream;
  * a decayed score picks the experts to keep, best first, skipping the ones resident in the VRAM cache (they are not
    read from the host), until the RAM budget is full; their records are mlock'ed (file-backed pages: no copy), the
    ones that dropped out are munlock'ed (they stay in the page cache until the kernel needs the memory);
  * the score is saved next to the store, so the next start locks the same experts before the first request.

Pinned tier (pin_static, the default when a profile exists): HMM pays a slow first GPU touch per page and mapped pages
lose their GPU mappings under memory pressure (measured on an RTX 5070: random 1.86 MB records from a 7 GB set read at
1.3-3 GB/s on the first pass, 11 GB/s by the third, 13 GB/s from cudaHostRegister'ed memory on every pass). So at start,
before the runtime exists, the profile's hottest records (after the ones the VRAM cache will hold) are replaced IN PLACE:
an anonymous mapping is put over each record run at the same virtual address (MAP_FIXED), the bytes are read into it
from the store file, and the run is cudaHostRegister'ed. The kernels compute the same addresses as before (host base +
expert * record), so a hot expert now resolves to pinned RAM and a cold one to the file pages, with no kernel change.
The mlock tier (StorePinner with locking) is kept as an option; the pinner always counts routing and saves the profile.

Env: SGLANG_EXL3_OFFLOAD_PIN_GB (auto | GB | 0 = off: pinned tier at start from the saved profile; auto = MemAvailable
minus SGLANG_EXL3_OFFLOAD_PIN_HEADROOM_GB, default 5), SGLANG_EXL3_OFFLOAD_PIN_SKIP (fraction of the VRAM slots whose
top-ranked experts are left to the VRAM cache, default 0.75), SGLANG_EXL3_OFFLOAD_LOCK_GB (mlock tier, default 0 = off),
SGLANG_EXL3_OFFLOAD_LOCK_INTERVAL_S (profile save / re-plan interval, default 20), SGLANG_EXL3_OFFLOAD_PROFILE_DECAY
(score decay per interval, default 0.995), SGLANG_EXL3_OFFLOAD_SORT (auto: repack once when a profile exists and no
sorted store does | repack: repack from the current profile at this start | 0: serve the unsorted store).
Needs RLIMIT_MEMLOCK at least the budget (docker --ulimit memlock=-1).
"""
from __future__ import annotations

import ctypes
import logging
import os
import threading
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mlock.argtypes = _libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)


def mem_available_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _budget(var: str, default: str, head_var: str, store_bytes: int) -> int:
    v = os.environ.get(var, default).strip().lower()
    if v in ("0", "off", "none", ""):
        return 0
    if v == "auto":
        head = float(os.environ.get(head_var, "5")) * 1e9
        b = int(max(0.0, mem_available_bytes() - head))
    else:
        b = int(float(v) * 1e9)
    return min(b, store_bytes)


def lock_budget_bytes(store_bytes: int) -> int:
    return _budget("SGLANG_EXL3_OFFLOAD_LOCK_GB", "0", "SGLANG_EXL3_OFFLOAD_LOCK_HEADROOM_GB", store_bytes)


def pin_budget_bytes(store_bytes: int) -> int:
    return _budget("SGLANG_EXL3_OFFLOAD_PIN_GB", "auto", "SGLANG_EXL3_OFFLOAD_PIN_HEADROOM_GB", store_bytes)


_PROT_READ, _PROT_WRITE = 1, 2
_MAP_PRIVATE, _MAP_FIXED, _MAP_ANONYMOUS = 0x02, 0x10, 0x20
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)


def load_profile(profile_path: str, L: int, E: int):
    try:
        score = np.load(profile_path).astype(np.float64).reshape(-1)
    except (OSError, ValueError):
        return None
    return score.reshape(L, E) if score.size == L * E and score.any() else None


def order_from_profile(score: np.ndarray) -> np.ndarray:
    """pos[l, e] = stored position of original expert e of layer l: most routed first (ties by id)."""
    L, E = score.shape
    pos = np.empty((L, E), dtype=np.int32)
    for l in range(L):
        pos[l, np.lexsort((np.arange(E), -score[l]))] = np.arange(E, dtype=np.int32)
    return pos


def repack_store(src, dst_path: str, pos: np.ndarray) -> float:
    """Write the store `src` (identity order, mapped) to dst_path with every layer's records in `pos` order (stored
    position p holds original expert inv[p]); the marker (dst_path + ".json", same layout + the order's sha256) is
    written last and is the commit point. Returns seconds."""
    import hashlib, json
    t0 = time.time()
    L, E, rec, stride = src.L, src.E, src.lay.record_bytes, src.stride
    tmp = dst_path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, L * stride)
        layer = bytearray(E * rec)
        mv = memoryview(layer)
        for l in range(L):
            inv = np.argsort(pos[l])
            for p_, e in enumerate(inv):
                o = l * stride + int(e) * rec
                got = 0
                while got < rec:
                    k = os.preadv(src._fd, [mv[p_ * rec + got:(p_ + 1) * rec]], o + got)
                    if k <= 0:
                        raise OSError(f"repack: short read at {o + got}")
                    got += k
            w = 0
            while w < E * rec:
                w += os.pwrite(fd, mv[w:], l * stride + w)
        os.fsync(fd)
    finally:
        os.close(fd)
    np.save(dst_path + ".perm.npy", pos)
    os.replace(tmp, dst_path)
    meta = dict(src._meta, order_sha256=hashlib.sha256(pos.tobytes()).hexdigest())
    with open(dst_path + ".json.tmp", "w") as f:
        json.dump(meta, f)
    os.replace(dst_path + ".json.tmp", dst_path + ".json")
    return time.time() - t0


def sorted_order(dst_path: str, layout_meta: dict):
    """The order of a committed sorted store, or None if it is missing, partial or of another layout."""
    import hashlib, json
    try:
        meta = json.load(open(dst_path + ".json"))
        pos = np.load(dst_path + ".perm.npy")
    except (OSError, ValueError):
        return None
    if {k: meta.get(k) for k in layout_meta} != layout_meta or meta.get("order_sha256") != hashlib.sha256(pos.tobytes()).hexdigest():
        return None
    if os.path.getsize(dst_path) != layout_meta["stride"] * len(layout_meta["layers"]):
        return None
    return pos


def pin_static(store, budget_bytes: int, score: np.ndarray, pos: np.ndarray, vram_slots: int) -> dict:
    """Pinned RAM tier at start (no kernel may run yet) for a store sorted by `pos`: the global ranks
    [skip, skip + budget) of the profile become, in every layer, one contiguous band of stored positions; each band's
    pages are replaced in place by pinned anonymous memory holding the same bytes (MAP_FIXED + pread + one
    cudaHostRegister per layer: registration costs ~130 ms per call on HMM, so the experts must be contiguous)."""
    t0 = time.time()
    st = {"pinned": 0, "pinned_gb": 0.0, "runs": 0, "seconds": 0.0, "skipped_top": 0, "errors": 0, "bands": {}}
    L, E, rec = store.L, store.E, store.lay.record_bytes
    base = store.buf.data_ptr()
    if store.stride % 4096 or base % 4096:
        return st
    flat = score.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    order = order[flat[order] > 0]
    skip = int(vram_slots * float(os.environ.get("SGLANG_EXL3_OFFLOAD_PIN_SKIP", "0.75")))
    take = int(budget_bytes // rec)
    lay_of = order // E
    skip_l = np.bincount(lay_of[:skip], minlength=L)
    take_l = np.bincount(lay_of[skip:skip + take], minlength=L)
    st["skipped_top"] = int(min(skip, len(order)))
    cudart = torch.cuda.cudart()
    pinned_bytes = 0
    for l in range(L):
        if take_l[l] == 0:
            continue
        lo = l * store.stride + int(skip_l[l]) * rec
        hi = lo + int(take_l[l]) * rec
        lo, hi = lo // 4096 * 4096, -(-hi // 4096) * 4096
        n = hi - lo
        a = _libc.mmap(ctypes.c_void_p(base + lo), n, _PROT_READ | _PROT_WRITE, _MAP_PRIVATE | _MAP_FIXED | _MAP_ANONYMOUS,
                       -1, 0)
        if a != base + lo:
            raise OSError(f"pin_static: MAP_FIXED at {base + lo:#x} failed (errno {ctypes.get_errno()})")
        mv = memoryview((ctypes.c_char * n).from_address(a)).cast("B")
        got = 0
        while got < n:
            k = os.preadv(store._fd, [mv[got:]], lo + got)
            if k <= 0:
                raise OSError(f"pin_static: short read of the store at {lo + got}")
            got += k
        if int(cudart.cudaHostRegister(a, n, 1 | 2)) != 0:      # Portable | Mapped; device ptr == host ptr (UVA)
            st["errors"] += 1
            continue
        st["runs"] += 1
        st["pinned"] += int(take_l[l])
        st["bands"][l] = (lo - l * store.stride, hi - l * store.stride)   # registered bytes, relative to the bank
        st.setdefault("band_pos", {})[l] = (int(skip_l[l]), int(skip_l[l] + take_l[l]))  # pinned stored positions
        pinned_bytes += n
    st["pinned_gb"] = pinned_bytes / 1e9
    st["seconds"] = time.time() - t0
    return st


class StorePinner:
    def __init__(self, store, runtime, budget_bytes: int, profile_path: str, interval_s: float = 20.0, decay: float | None = None):
        self.store, self.rt = store, runtime
        self.L, self.E, self.rec = store.L, store.E, store.lay.record_bytes
        self.cap = int(budget_bytes // self.rec)
        # long-run average by default: 0.995 per poll = a half-life of ~2.3 h at 20 s polls
        decay = float(os.environ.get("SGLANG_EXL3_OFFLOAD_PROFILE_DECAY", "0.995")) if decay is None else decay
        self.profile_path, self.interval, self.decay = profile_path, interval_s, decay
        self.locked = np.zeros(self.L * self.E, dtype=bool)
        self.score = np.zeros(self.L * self.E, dtype=np.float64)
        self._last = np.zeros(self.L * self.E, dtype=np.int64)
        self._host = torch.zeros(self.L * (self.E + 1), dtype=torch.int32).pin_memory()
        self._stream = torch.cuda.Stream(device=runtime.device)
        self._stop = threading.Event()
        self.serving = threading.Event()          # set by the MoE method on the first eager forward (captures are done)
        self.stats = {"locked": 0, "locked_gb": 0.0, "plans": 0, "lock_s": 0.0, "lock_errors": 0}
        try:
            self.score = np.load(profile_path).astype(np.float64).reshape(-1)[: self.L * self.E]
            logger.info("EXL3 offload: expert profile %s loaded (%d experts routed before)", profile_path,
                        int((self.score > 0).sum()))
        except (OSError, ValueError):
            pass

    # ---- mlock of record runs
    def _addr(self, k: int) -> int:
        i, e = divmod(int(k), self.E)
        return self.store.buf.data_ptr() + i * self.store.stride + e * self.rec

    def _apply(self, keys: np.ndarray, lock: bool) -> int:
        """lock / unlock records `keys` (sorted), adjacent records of one layer as one call; returns failures."""
        bad = 0
        fn = _libc.mlock if lock else _libc.munlock
        if len(keys) == 0:
            return 0
        brk = np.nonzero((np.diff(keys) != 1) | (np.diff(keys // self.E) != 0))[0] + 1
        for run in np.split(keys, brk):
            if fn(ctypes.c_void_p(self._addr(run[0])), ctypes.c_size_t(len(run) * self.rec)) != 0:
                bad += 1
        return bad

    def plan(self) -> None:
        resident = self.rt.cache.slot_of.view(-1)[: self.L * self.E]
        with torch.cuda.stream(self._stream):
            vram = (resident >= 0).to("cpu", non_blocking=True)
        self._stream.synchronize()
        vram = vram.numpy()
        order = np.argsort(-self.score, kind="stable")
        order = order[(self.score[order] > 0) & ~vram[order]][: self.cap]
        want = np.zeros_like(self.locked)
        want[order] = True
        t = time.time()
        drop = np.nonzero(self.locked & ~want)[0]
        add = np.nonzero(want & ~self.locked)[0]
        self.stats["lock_errors"] += self._apply(drop, False)
        bad = self._apply(add, True)
        self.stats["lock_errors"] += bad
        self.locked = want
        n = int(want.sum())
        self.stats.update(locked=n, locked_gb=n * self.rec / 1e9, plans=self.stats["plans"] + 1,
                          lock_s=self.stats["lock_s"] + time.time() - t)
        if bad:
            logger.warning("EXL3 offload: %d mlock calls failed (errno %d): raise RLIMIT_MEMLOCK", bad, ctypes.get_errno())

    def poll(self) -> None:
        c = self.rt.route_counts
        with torch.cuda.stream(self._stream):
            self._host.copy_(c.view(-1), non_blocking=True)
        self._stream.synchronize()
        now = self._host.view(self.L, self.E + 1)[:, : self.E].numpy().reshape(-1).astype(np.int64)
        delta = np.maximum(now - self._last, 0)
        self._last = now
        if delta.any():
            self.score = self.score * self.decay + delta
            tmp = self.profile_path + ".tmp.npy"
            np.save(tmp, self.score.astype(np.float32))
            os.replace(tmp, self.profile_path)

    def run(self) -> None:
        self.serving.wait()                       # no CUDA call before graph capture is over: a sync would invalidate it
        try:
            if self.score.any() and self.cap > 0:
                self.plan()
                logger.info("EXL3 offload: RAM tier from the saved profile: %d experts locked (%.1f GB) in %.1f s",
                            self.stats["locked"], self.stats["locked_gb"], self.stats["lock_s"])
            while not self._stop.wait(self.interval):
                self.poll()
                if self.cap > 0:
                    self.plan()
        except Exception:  # pragma: no cover - the server keeps running on HMM alone
            logger.exception("EXL3 offload: RAM tier pinner stopped")

    def start(self) -> "StorePinner":
        logger.info("EXL3 offload: routing profile saved every %.0f s to %s%s", self.interval, self.profile_path,
                    f"; mlock tier {self.cap * self.rec / 1e9:.1f} GB = {self.cap} experts" if self.cap else "")
        threading.Thread(target=self.run, name="exl3-store-pinner", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
