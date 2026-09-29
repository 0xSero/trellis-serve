"""Tier 2 for the EXL3 n-gram ("PLE") table: rows stay on NVMe, a fixed-budget RAM row cache sits in front.

Drop-in for `Exl3NgramHostTable` (same `gather/allocate_output/reduce/forward/release` interface SGLang's
`Qwen4ExpPinnedHostEmbedding` path calls, same bf16 output bit for bit). Selected by `ngram_host.get_table()` when
`SGLANG_EXL3_NGRAM_TIER=nvme`.

Data path per lookup (CUDA-graph safe, one request in flight, ordered on the calling stream):
  GPU  nvme_publish : SGLang's hashed ids [T,16] -> mapped host `req`, bump device seq, release-store ctrl.req_seq
  CPU  service thread (C++): polls req_seq -> RAM cache lookup (CLOCK, row id keyed, in-request dedup) -> misses sorted
       by file offset, coalesced into 4 KiB-aligned runs, read with O_DIRECT (Linux AIO, deep queue; or pread pool)
       straight from ngram_embedding.safetensors into cache slots -> slot ids to mapped `resp` -> release done_seq
  GPU  nvme_wait    : 1-thread spin until done_seq == seq (stall recorded in ns), copy slot ids to the device
  GPU  ngram_gather_dequant(slot ids, slab)  -- the unchanged pinned-tier kernel, reading the cache slab zero-copy
In SGLang the gather runs on the PLE prefetch stream, started before layer 0, so the host fill overlaps layer 0.

Env (read at construction):
  SGLANG_EXL3_NGRAM_TIER        pinned (default, whole table in pinned RAM) | nvme (this module)
  SGLANG_EXL3_NGRAM_RAM_GB      RAM row-cache budget in GB (default 4; 102 B/row -> 39.2M rows at 4 GB)
  SGLANG_EXL3_NGRAM_IO          aio (default, O_DIRECT + io_submit) | pread (O_DIRECT, thread pool) | buffered (page cache)
  SGLANG_EXL3_NGRAM_QD          aio queue depth (default 128)          SGLANG_EXL3_NGRAM_THREADS  pread threads (16)
  SGLANG_EXL3_NGRAM_MAX_TOKENS  largest forward (tokens) one lookup may carry (default 32768 -> 524,288 ids)
  SGLANG_EXL3_NGRAM_MAX_RUN     largest coalesced read in bytes (default 65536)
  SGLANG_EXL3_NGRAM_MERGE_GAP   merge two runs if the gap is <= this many bytes (default 0: touching blocks only)
  SGLANG_EXL3_NGRAM_SPIN_US     service thread busy-polls this long after the last request, then sleeps 20 us (200000)
  SGLANG_EXL3_NGRAM_TIMEOUT_S   GPU wait timeout (default 60): on expiry the step continues with stale rows, error set
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .ngram_host import FILE, _header, _read_small

logger = logging.getLogger(__name__)

_ext_mod = None
_ext_lock = threading.Lock()


def load_ext():
    """JIT build of csrc/offload/ngram_nvme.cu (host RowStore + publish/wait kernels)."""
    global _ext_mod
    if _ext_mod is not None:
        return _ext_mod
    with _ext_lock:
        if _ext_mod is None:
            from torch.utils import cpp_extension
            here = os.path.dirname(os.path.abspath(__file__))
            csrc = os.path.normpath(os.path.join(here, "..", "..", "..", "csrc"))
            build = os.environ.get("SGLANG_EXL3_JIT_DIR", os.path.join(csrc, "build", "jit"))
            build = os.path.join(build, "ngram_nvme")
            os.makedirs(build, exist_ok=True)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
            _ext_mod = cpp_extension.load(
                name="trellis_ngram_nvme",
                sources=[os.path.join(csrc, "offload", "ngram_nvme.cu")],
                extra_cflags=["-O3", "-std=c++17", "-march=native"],
                extra_cuda_cflags=["-O3", "-std=c++17", "-Xcompiler", "-march=native"],
                build_directory=build,
                verbose=False,
            )
    return _ext_mod


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def parse_table(model_path: str):
    """Header facts of ngram_embedding.safetensors: extents (row0, rows, file offset) per shard, row width, aux."""
    import re
    path = os.path.join(model_path, FILE)
    hdr, base = _header(path)
    meta = hdr.pop("__metadata__", {}) or {}
    if meta.get("format") != "exl3_ngram_trellis":
        raise ValueError(f"{path}: not an exl3_ngram_trellis table ({meta.get('format')!r})")
    shards, prefix = {}, None
    for k, v in hdr.items():
        m = re.match(r"(.*)\.shard_(\d+)\.trellis$", k)
        if m:
            shards[int(m.group(2))] = v
            prefix = m.group(1)
    if not shards or sorted(shards) != list(range(len(shards))):
        raise ValueError(f"{path}: shard_N.trellis tensors missing or not contiguous")
    words = shards[0]["shape"][1]
    K = (words - 1) * 16 // 160
    if K != int(meta.get("K", K)) or 1 + 160 * K // 16 != words:
        raise ValueError(f"{path}: row width {words} does not match K={meta.get('K')}")
    row0, rows, offs, r = [], [], [], 0
    for i in range(len(shards)):
        a, b = shards[i]["data_offsets"]
        n = shards[i]["shape"][0]
        assert b - a == n * words * 2
        row0.append(r), rows.append(n), offs.append(base + a)
        r += n
    return dict(path=path, hdr=hdr, base=base, prefix=prefix, words=words, K=K, row0=row0, rows=rows, offs=offs,
                num_rows=r)


class Exl3NgramNvmeTable(torch.nn.Module):
    """NVMe row store + pinned RAM row cache; `gather()` is graph-safe (see module doc)."""

    def __init__(self, model_path: str, ram_gb: float | None = None, io: str | None = None, device=None,
                 max_tokens: int | None = None, start_service: bool = True):
        super().__init__()
        t = parse_table(model_path)
        self.path, self.words, self.K, self.num_rows = t["path"], t["words"], t["K"], t["num_rows"]
        self.prefix = t["prefix"]
        self.row_bytes = self.words * 2
        ram_gb = float(ram_gb if ram_gb is not None else _env("SGLANG_EXL3_NGRAM_RAM_GB", 4.0, float))
        io = io or _env("SGLANG_EXL3_NGRAM_IO", "aio")
        backend = {"aio": 0, "pread": 1, "buffered": 2}[io]
        max_tokens = int(max_tokens or _env("SGLANG_EXL3_NGRAM_MAX_TOKENS", 32768, int))
        aux_names = ("head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers")
        aux = {n: _read_small(t["path"], t["base"], t["hdr"][f"{self.prefix}.{n}"]) for n in aux_names
               if f"{self.prefix}.{n}" in t["hdr"]}
        self.num_heads = aux["head_bias"].shape[0]
        self.cap = max_tokens * self.num_heads
        nslots = max(int(ram_gb * 1e9) // self.row_bytes, self.cap)
        self.io = io
        self.ram_gb = nslots * self.row_bytes / 1e9
        self.ext = load_ext()
        t0 = time.time()
        self.store = self.ext.RowStore(t["path"], t["row0"], t["rows"], t["offs"], self.row_bytes, nslots, backend,
                                       _env("SGLANG_EXL3_NGRAM_THREADS", 16, int), _env("SGLANG_EXL3_NGRAM_QD", 128, int),
                                       _env("SGLANG_EXL3_NGRAM_MAX_RUN", 65536, int),
                                       _env("SGLANG_EXL3_NGRAM_MERGE_GAP", 0, int), self.cap)
        self.nslots = nslots
        self.file_hash = {k: aux[k].long() for k in ("head_offsets", "head_vocab_sizes", "layer_multipliers")}
        self.embedding_dim = 160
        self.num_embeddings = self.num_rows
        self._registered = []
        self._hint_pool = None
        self.device = None
        if device is not None or torch.cuda.is_available():
            dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
            self._attach(dev, aux["head_bias"])
        if start_service and self.device is not None:
            self.store.start_service(_env("SGLANG_EXL3_NGRAM_SPIN_US", 200000, int))
        logger.info("EXL3 n-gram table on NVMe (%s): %d rows x %d B, RAM row cache %.2f GB = %d slots, max %d ids/lookup, "
                    "init %.1f s", io, self.num_rows, self.row_bytes, self.ram_gb, nslots, self.cap, time.time() - t0)

    def _attach(self, dev, head_bias):
        from . import _ext
        self.device = dev
        self._gather_ext = _ext.load()
        with torch.cuda.device(dev):
            s = self.store
            self.slab_dev = self.ext.host_register(s.slab_ptr(), s.slab_bytes())
            self._registered.append(s.slab_ptr())
            self.req_dev = self.ext.host_register(s.req_ptr(), s.io_bytes())
            self._registered.append(s.req_ptr())
            self.resp_dev = self.ext.host_register(s.resp_ptr(), s.io_bytes())
            self._registered.append(s.resp_ptr())
            self.ctrl_dev = self.ext.host_register(s.ctrl_ptr(), 4096)
            self._registered.append(s.ctrl_ptr())
        self.register_buffer("head_bias", head_bias.to(dev, torch.float16).contiguous(), persistent=False)
        self.dseq = torch.zeros(4, dtype=torch.int32, device=dev)
        self.stall = torch.zeros(4, dtype=torch.int64, device=dev)
        self.slots_dev = torch.empty(self.cap, dtype=torch.int64, device=dev)
        self.timeout_ns = int(_env("SGLANG_EXL3_NGRAM_TIMEOUT_S", 60.0, float) * 1e9)

    # ---- lifecycle
    def release(self) -> None:
        s = getattr(self, "store", None)
        if s is None:
            return
        if self._hint_pool is not None:
            self._hint_pool.shutdown(wait=True)
            self._hint_pool = None
        if self.device is not None:
            torch.cuda.synchronize(self.device)
        s.close()
        for p in self._registered:
            self.ext.host_unregister(p)
        self._registered = []
        s.release_memory()
        self.store = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # ---- Qwen4ExpPinnedHostEmbedding interface
    def allocate_output(self, shape, device) -> torch.Tensor:
        with torch.inference_mode(False):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(self, ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if ids.shape[-1] != self.num_heads:
            raise ValueError(f"n-gram lookup expects [..., {self.num_heads}] ids, got {tuple(ids.shape)}")
        flat = ids.reshape(-1)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        n = flat.numel()
        if n > self.cap:
            raise ValueError(f"n-gram NVMe tier: {n} ids > capacity {self.cap} (raise SGLANG_EXL3_NGRAM_MAX_TOKENS)")
        if out is None:
            out = torch.empty((*ids.shape, 160), dtype=torch.bfloat16, device=ids.device)
        if out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() != n * 160:
            raise ValueError(f"n-gram gather output {tuple(out.shape)} {out.dtype} does not fit {tuple(ids.shape)} ids")
        if n == 0:
            return out
        self.ext.nvme_publish(flat, self.req_dev, self.ctrl_dev, self.dseq)
        slots = self.slots_dev[:n]
        self.ext.nvme_wait(self.ctrl_dev, self.dseq, self.stall, self.resp_dev, slots, n, self.timeout_ns)
        self._gather_ext.ngram_gather_dequant(slots, self.slab_dev, self.nslots, self.words, self.K, self.head_bias, out)
        return out

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gather(ids).view(*ids.shape, 160)

    # ---- prefetch hint (optional; e.g. from the scheduler for the next prefill chunk)
    def hash_tokens(self, tokens, history=None, eos: int | None = None) -> torch.Tensor:
        """Row ids [L, 16] for `tokens` (1-D) given the 2 preceding tokens (`history`, eos-filled at a sequence start)."""
        t = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
        if eos is None:
            eos = getattr(self, "eos_token_id", None)
        if eos is None:
            raise ValueError("hash_tokens needs eos (set table.eos_token_id)")
        h = torch.as_tensor(history if history is not None else [eos, eos], dtype=torch.long).reshape(-1)[-2:]
        full = torch.cat([h, t]).contiguous()
        out = torch.empty((t.numel(), self.num_heads), dtype=torch.long)
        fh = self.file_hash
        self.ext.ngram_hash_tokens(full, fh["layer_multipliers"].contiguous(), fh["head_vocab_sizes"].contiguous(),
                                   fh["head_offsets"].contiguous(), self.num_heads // 2, int(eos), out)
        return out

    def hint_tokens(self, tokens, history=None, eos: int | None = None, wait: bool = False):
        """Warm the RAM cache for positions that will be looked up soon (async on one worker thread)."""
        ids = self.hash_tokens(tokens, history, eos).reshape(-1)
        if wait:
            return self.store.warm(ids)
        if self._hint_pool is None:
            self._hint_pool = ThreadPoolExecutor(1, thread_name_prefix="ngram_nvme_hint")
        return self._hint_pool.submit(self.store.warm, ids)

    # ---- stats
    COUNTERS = ("requests", "lookups", "unique", "hits", "misses", "runs", "bytes", "blocks", "io_us", "resolve_us",
                "resident", "error", "warm_lookups", "warm_misses", "warm_bytes", "warm_us")

    def stats(self, gpu: bool = True) -> dict:
        d = dict(zip(self.COUNTERS, self.store.counters()))
        d["unique_hit_rate"] = d["hits"] / max(d["unique"], 1)
        d["lookup_hit_rate"] = 1 - d["misses"] / max(d["lookups"], 1)
        if gpu and self.device is not None:
            st = self.stall.cpu().tolist()
            d.update(gpu_wait_ns_total=st[0], gpu_waits=st[1], gpu_wait_ns_max=st[2])
        return d

    def reset_stats(self):
        self.store.reset_counters()
        if self.device is not None:
            self.stall.zero_()


_HINT_WARNED = [False]


def install_prefill_hints() -> None:
    """Warm the NVMe tier's RAM cache for the *next* prefill chunk while the current one computes.

    Wraps SGLang `Scheduler.get_new_batch_prefill`: after scheduling, if a request is still being chunked
    (`self.chunked_req`), its next `chunked_prefill_size` prompt tokens (after `len(req.fill_ids)`) are hashed on the
    CPU and warmed asynchronously. Only works when the scheduler and the model worker share a process (tp_size 1).
    A wrong or late hint costs nothing but hit rate: lookups always go through the normal graph-safe path.
    Disable with SGLANG_EXL3_NGRAM_HINT=0."""
    if os.environ.get("SGLANG_EXL3_NGRAM_HINT", "1") == "0":
        return
    from sglang.srt.managers import scheduler as sch
    cls = sch.Scheduler
    if getattr(cls, "_exl3_ngram_hint", False):
        return
    orig = cls.get_new_batch_prefill

    def get_new_batch_prefill(self, *a, **k):
        out = orig(self, *a, **k)
        try:
            req = getattr(self, "chunked_req", None)
            if req is not None:
                from .ngram_host import _TABLES
                tabs = [t for t in _TABLES.values() if isinstance(t, Exl3NgramNvmeTable)]
                if tabs:
                    tab = tabs[0]
                    ids = list(req.origin_input_ids)
                    # SGLang 0.5.20: Req.get_fill_ids() (up to the scheduled chunk's end); older: Req.fill_ids
                    start = len(req.get_fill_ids()) if hasattr(req, "get_fill_ids") else len(req.fill_ids)
                    size = int(getattr(self, "chunked_prefill_size", 0) or 8192)
                    if 0 < start < len(ids):
                        eos = getattr(tab, "eos_token_id", None)
                        hist = ([eos, eos] + ids[:start])[-2:]
                        tab.hint_tokens(ids[start:start + size], history=hist, eos=eos)
        except Exception as e:  # never break scheduling for a hint
            if not _HINT_WARNED[0]:
                _HINT_WARNED[0] = True
                logger.warning("n-gram NVMe prefill hint disabled after error: %r", e)
        return out

    cls.get_new_batch_prefill = get_new_batch_prefill
    cls._exl3_ngram_hint = True
