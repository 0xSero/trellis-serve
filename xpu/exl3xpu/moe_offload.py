"""
Two-tier EXL3 routed-expert store for the B70 (XPU): every expert of every MoE layer lives in USM host memory
(tier 1, read zero-copy over PCIe by the kernels), a subset lives in device slots (tier 0, the expert cache). The
grouped MoE kernels (`torch.ops.exl3xpu_moe.moe_forward`, csrc/exl3_moe.sycl) address experts only through a
per-layer POINTER TABLE, so moving an expert between tiers is a single 8-byte table write; both pointers are always
valid, so a table update never races a kernel into reading garbage.

Expert blob (one per expert, `blob_bytes(H, I, K)` bytes, 1,862,400 B for H=2560, I=640, K=3):
  gate|up trellis (u32, [H/16][2I/16][8K], PLANAR4 word order) | down trellis ([I/16][H/16][8K], PLANAR4) |
  suh_g[H] suh_u[H] svh_g[I] svh_u[I] suh_d[I] svh_d[H]   (fp16)
PLANAR4: in every k-row the tiles are grouped by 4 along n; a group's 4*8K words are stored plane-major
([plane i][tile][period g]) instead of tile-major ([tile][3g + i]) -- see pack_expert().

API (all device work is enqueued on the current XPU stream unless noted; nothing syncs the host):
  store = ExpertStore(H, I, K, n_experts, n_slots)          # n_slots device slots shared by all layers
  store.add_layer(key, blobs_cpu [E, BLOB] uint8 | None)     # allocates the layer's host USM arena (fills it)
  store.host_view(key)[e]                                    # uint8 view of expert e's host blob
  store.ptrs(key)                                            # int64 [E] device pointer table (fixed address)
  store.make_resident(key, experts, stream=None)             # copy blobs into free/evicted slots, then repoint
  store.evict(key, experts)                                  # repoint to host (slot freed; reuse is stream-ordered)
  store.forward(key, x, topk_ids, topk_w) -> out             # routed-MoE output (weights applied, shared expert NOT)
  store.forward_cached(key, x, topk_ids, topk_w)            # decode with the DEVICE-managed LRU cache: misses are read
                                                             #   zero-copy and written through into LRU slots, the
                                                             #   table is repointed by the same call (no host sync)
  store.stage_layer(key, buf, stream) -> ptrs                # prefill: copy the layer's non-resident experts into
                                                             #   staging buffer buf (0/1) on `stream`; returns a
                                                             #   pointer table (slots for residents, staging else)
  store.slot_of[key]  (cpu int32 [E], -1 = host only)        # the slot map
"""
from __future__ import annotations

import os
import threading

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib_loaded = False
_lock = threading.Lock()


def ops():
    """torch.ops.exl3xpu_moe, loading exl3xpu/_moe.so (or $EXL3_MOE_LIB) once."""
    global _lib_loaded
    with _lock:
        if not _lib_loaded:
            torch.ops.load_library(os.environ.get("EXL3_MOE_LIB") or os.path.join(_HERE, "_moe.so"))

            @torch.library.register_fake("exl3xpu_moe::moe_forward")
            def _fake(x, topk_ids, topk_w, ptrs, I, K, n_experts):   # noqa: N803
                return torch.empty_like(x)

            if hasattr(torch.ops.exl3xpu_moe, "moe_forward_cached"):
                @torch.library.register_fake("exl3xpu_moe::moe_forward_cached")
                def _fake_c(x, *args):
                    return torch.empty_like(x)

            _lib_loaded = True
    return torch.ops.exl3xpu_moe


def s64(p: int) -> int:
    return p - (1 << 64) if p >= (1 << 63) else p


def planar4(tr: torch.Tensor, K: int) -> torch.Tensor:
    """int16 trellis [rows, tiles, 16K] -> int32 words in PLANAR4 order (K with gcd(K, 32) = 1: 8 periods/tile)."""
    assert K in (3, 5, 7), "PLANAR4 packing implemented for odd K (D = K words per 32-value period)"
    r, n, _ = tr.shape
    w = tr.contiguous().view(torch.int32).view(r, n // 4, 4, 8, K)       # [row, group, tile, period g, plane i]
    return w.permute(0, 1, 4, 2, 3).contiguous()                         # [row, group, plane i, tile, g]


def pack_expert(gate: dict, up: dict, down: dict, K: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Checkpoint tensors {trellis, suh, svh} of one expert -> blob (uint8 CPU tensor, or written into `out`)."""
    b = lambda z: z.contiguous().view(torch.uint8).flatten()     # noqa: E731
    gu = torch.cat([gate["trellis"], up["trellis"]], dim=1)
    parts = [b(planar4(gu, K)), b(planar4(down["trellis"], K)), b(gate["suh"].half()), b(up["suh"].half()),
             b(gate["svh"].half()), b(up["svh"].half()), b(down["suh"].half()), b(down["svh"].half())]
    if out is None:
        return torch.cat(parts)
    o = 0
    for p in parts:
        out[o:o + p.numel()].copy_(p)
        o += p.numel()
    assert o == out.numel(), (o, out.numel())
    return out


class ExpertStore:
    def __init__(self, H: int, I: int, K: int, n_experts: int, n_slots: int, device=None, max_layers: int = 64):
        self.H, self.I, self.K, self.E = H, I, K, n_experts
        self.dev = device or torch.device("xpu", torch.xpu.current_device())
        self.X = ops()
        self.blob = int(self.X.blob_bytes(H, I, K))
        self.n_slots = n_slots
        self.max_layers = max_layers
        self.slots = torch.empty((n_slots, self.blob), dtype=torch.uint8, device=self.dev) if n_slots else None
        self.slot_owner: list = [None] * n_slots            # slot -> (key, e)   (static placement mirror)
        self.free_slots = list(range(n_slots - 1, -1, -1))
        self.host: dict = {}                                # key -> uint8 CPU tensor [E, BLOB] (USM host)
        self.layer_index: dict = {}                         # key -> row in the device tables
        self.slot_of: dict = {}                             # key -> cpu int32 [E] (mirror of static placement)
        self._host_ptr: dict = {}
        self._stage = [None, None]
        self._slot_ready_ev: dict = {}
        # device-side cache state (authoritative for forward_cached)
        LE = max_layers * n_experts
        self.ptrs_all = torch.zeros(LE, dtype=torch.int64, device=self.dev)
        self.slot_of_dev = torch.full((LE,), -1, dtype=torch.int32, device=self.dev)
        self.slot_key = torch.full((n_slots,), -1, dtype=torch.int32, device=self.dev)
        self.slot_last = torch.full((n_slots,), -1, dtype=torch.int32, device=self.dev)
        self.tick = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.host_base = torch.zeros(max_layers, dtype=torch.int64, device=self.dev)
        self.fill_all = torch.zeros(LE, dtype=torch.int64, device=self.dev)
        self.fill_list = torch.zeros(1 + n_experts, dtype=torch.int32, device=self.dev)
        self.slot_base = s64(self.slots.data_ptr()) if n_slots else 0

    # ---- layers
    def add_layer(self, key, blobs: torch.Tensor | None = None) -> torch.Tensor:
        li = len(self.layer_index)
        if li >= self.max_layers:
            raise RuntimeError(f"ExpertStore: more than max_layers={self.max_layers} MoE layers")
        self.layer_index[key] = li
        h = self.X.host_alloc(self.E * self.blob).view(self.E, self.blob)
        if blobs is not None:
            h.copy_(blobs)
        self.host[key] = h
        base = s64(h.data_ptr())
        self._host_ptr[key] = base + torch.arange(self.E, dtype=torch.int64) * self.blob
        self.ptrs_all[li * self.E:(li + 1) * self.E].copy_(self._host_ptr[key])
        self.host_base[li] = base
        self.slot_of[key] = torch.full((self.E,), -1, dtype=torch.int32)
        return h

    def host_view(self, key) -> torch.Tensor:
        return self.host[key]

    def ptrs(self, key) -> torch.Tensor:
        li = self.layer_index[key]
        return self.ptrs_all[li * self.E:(li + 1) * self.E]

    def slot_ptr(self, s: int) -> int:
        return self.slot_base + s * self.blob

    # ---- static residency (load time / host-driven policies)
    def make_resident(self, key, experts, stream=None) -> int:
        """Copy experts into free device slots, then repoint the table (and the device slot map). Copies and the
        table update are ordered on `stream` (default: current stream). Returns #copied."""
        s_ = stream or torch.xpu.current_stream()
        so = self.slot_of[key]
        li = self.layer_index[key]
        todo = [int(e) for e in experts if so[int(e)] < 0]
        if not todo:
            return 0
        es, sls = [], []
        with torch.xpu.stream(s_):
            for e in todo:
                if not self.free_slots:
                    raise RuntimeError("ExpertStore: no free slot (evict first or raise n_slots)")
                sl = self.free_slots.pop()
                ev = self._slot_ready_ev.pop(sl, None)
                if ev is not None:
                    s_.wait_event(ev)
                self.X.memcpy_async(self.slot_ptr(sl), s64(self.host[key][e].data_ptr()), self.blob)
                self.slot_owner[sl] = (key, e)
                so[e] = sl
                es.append(e)
                sls.append(sl)
            keys = torch.tensor([li * self.E + e for e in es], dtype=torch.int64)
            slt = torch.tensor(sls, dtype=torch.int64)
            d = self.dev
            self.ptrs_all.index_copy_(0, keys.to(d), (self.slot_base + slt * self.blob).to(d))
            self.slot_of_dev.index_copy_(0, keys.to(d), slt.to(torch.int32).to(d))
            self.slot_key.index_copy_(0, slt.to(d), keys.to(torch.int32).to(d))
            self.slot_last.index_fill_(0, slt.to(d), 0)
        return len(todo)

    def evict(self, key, experts) -> None:
        """Repoint to the host copy; the slot becomes reusable after the current stream's already-queued work."""
        so = self.slot_of[key]
        li = self.layer_index[key]
        es, sls = [], []
        for e in experts:
            e = int(e)
            sl = int(so[e])
            if sl < 0:
                continue
            es.append(e)
            sls.append(sl)
            so[e] = -1
            self.slot_owner[sl] = None
        if not es:
            return
        d = self.dev
        et = torch.tensor(es, dtype=torch.int64)
        keys = (li * self.E + et).to(d)
        slt = torch.tensor(sls, dtype=torch.int64).to(d)
        self.ptrs_all.index_copy_(0, keys, self._host_ptr[key][et].to(d))
        self.slot_of_dev.index_fill_(0, keys, -1)
        self.slot_key.index_fill_(0, slt, -1)
        self.slot_last.index_fill_(0, slt, -1)
        ev = torch.xpu.Event()
        ev.record(torch.xpu.current_stream())
        for sl in sls:
            self._slot_ready_ev[sl] = ev
            self.free_slots.append(sl)

    def resident_count(self, key=None) -> int:
        if key is None:
            return sum(int((v >= 0).sum()) for v in self.slot_of.values())
        return int((self.slot_of[key] >= 0).sum())

    def device_resident_count(self, key) -> int:
        """From the device slot map (syncs; debug/metrics only)."""
        li = self.layer_index[key]
        return int((self.slot_of_dev[li * self.E:(li + 1) * self.E] >= 0).sum().item())

    # ---- compute
    @staticmethod
    def _rt(topk_ids, topk_w):
        ids = topk_ids if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous() else topk_ids.to(torch.int32).contiguous()
        w = topk_w if topk_w.dtype == torch.float32 and topk_w.is_contiguous() else topk_w.to(torch.float32).contiguous()
        return ids, w

    def forward(self, key, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor, ptrs=None) -> torch.Tensor:
        ids, w = self._rt(topk_ids, topk_w)
        return self.X.moe_forward(x, ids, w, self.ptrs(key) if ptrs is None else ptrs, self.I, self.K, self.E)

    def forward_cached(self, key, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
                       max_fill: int | None = None) -> torch.Tensor:
        """Decode with the device-managed LRU expert cache: hits read their slot, misses read host memory zero-copy
        and are written through into the LRU slot (committed at the end of the call). No host sync, graph-safe.
        Do not mix with make_resident/evict on the same slots while cached calls are queued."""
        ids, w = self._rt(topk_ids, topk_w)
        return self.X.moe_forward_cached(x, ids, w, self.ptrs_all, self.layer_index[key], self.slot_of_dev,
                                         self.slot_key, self.slot_last, self.tick, self.host_base, self.slot_base,
                                         self.fill_all, self.fill_list, self.E if max_fill is None else max_fill,
                                         self.I, self.K, self.E)

    # ---- prefill staging (FreeToken-style layer streaming)
    def stage_layer(self, key, buf: int, stream) -> torch.Tensor:
        """Copy every non-resident expert of `key` into staging buffer `buf` on `stream`; returns the pointer table
        to pass to forward(ptrs=...) once `stream` has been waited on. Contiguous expert runs are copied together.
        Uses the static-placement mirror (slot_of); after forward_cached calls, refresh it with sync_mirror()."""
        if self._stage[buf] is None:
            self._stage[buf] = torch.empty((self.E, self.blob), dtype=torch.uint8, device=self.dev)
        st = self._stage[buf]
        so = self.slot_of[key]
        miss = (so < 0).nonzero().flatten().tolist()
        with torch.xpu.stream(stream):
            i = 0
            while i < len(miss):
                j = i
                while j + 1 < len(miss) and miss[j + 1] == miss[j] + 1:
                    j += 1
                n = j - i + 1
                self.X.memcpy_async(s64(st[miss[i]].data_ptr()), s64(self.host[key][miss[i]].data_ptr()), n * self.blob)
                i = j + 1
        sp = s64(st.data_ptr()) + torch.arange(self.E, dtype=torch.int64) * self.blob
        if self.n_slots:
            dp = self.slot_base + so.clamp_min(0).to(torch.int64) * self.blob
            sp = torch.where(so >= 0, dp, sp)
        return sp.to(self.dev, non_blocking=True)      # CPU mirror only: no device->host read

    def sync_mirror(self) -> None:
        """Refresh the CPU slot mirror from the device cache state (device->host read; not for the decode loop)."""
        sod = self.slot_of_dev.cpu()
        for key, li in self.layer_index.items():
            self.slot_of[key] = sod[li * self.E:(li + 1) * self.E].clone()
