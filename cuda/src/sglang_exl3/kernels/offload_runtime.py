"""OffloadRuntime: the whole routed-expert offload for one model on one GPU (no SGLang dependency beyond its align op).

  store    HostExpertStore: all layers' experts, pinned + mapped host memory (record layout)
  cache    om.ExpertCache: global GPU slot pool (CLOCK, fused admission), per-layer pointer tables, hit/miss counters
  decode   forward(l, ...) with T < prefill_min_tokens (or inside CUDA-graph capture): cache.run -> hits from slots,
           misses zero-copy from the host bank and written into their victim slots by the GEMM. Graph-safe.
  prefill  T >= prefill_min_tokens (eager): layer-ahead staging. At the first prefill layer of a forward the slot map is
           read back once (prefill never changes the cache), then for layer l the missing experts are copied with
           cudaMemcpyAsync (runs of adjacent experts) on a dedicated copy stream into staging buffer l % 2 while layer
           l-1 computes; the grouped kernel then reads resident experts from their slots and the rest from staging.
           Two staging buffers + ready/release events. Staged experts are NOT admitted (decision K05: prefill touches
           every expert, admitting would flush the decode working set and invalidate the pass's slot-map snapshot; K03:
           seeding the cache from the prompt's hot set is worth +0.1 pt).
  stats    stats() -> per-layer decode hits/misses (device counters) and prefill copied/resident experts.
"""
from __future__ import annotations

import numpy as np
import torch

from . import marlin_moe, offload_moe as om
from .offload_store import HostExpertStore


def _align(ids, block, num_experts):
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    return moe_align_block_size(ids, block, num_experts, ignore_invalid_expert=True)


class OffloadRuntime:
    def __init__(self, store: HostExpertStore, slots: int, codebook: int = 2, top_k: int = 10,
                 prefill_min_tokens: int = 256, staging: bool = True, device=None):
        self.store, self.lay, self.cb, self.top_k = store, store.lay, codebook, top_k
        self.L, self.E = store.L, store.E
        dev = torch.device(device or "cuda")
        self.device = dev
        self.cache = om.ExpertCache(self.L, self.E, self.lay, [store.base_address(i) for i in range(self.L)], slots, dev)
        self.prefill_min_tokens = prefill_min_tokens
        rec = self.lay.record_bytes
        self.staging = [torch.empty((self.E, rec), dtype=torch.uint8, device=dev) for _ in range(2)] if staging else []
        self.ptables = [om.new_table(self.E, dev) for _ in range(2)]
        self.copy_stream = torch.cuda.Stream(device=dev)
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.release = [torch.cuda.Event() for _ in range(2)]
        for ev in self.release:
            ev.record()
        self._snapshot = None
        self._pending: dict[int, int] = {}
        self._erange = torch.arange(self.E, dtype=torch.int64, device=dev)
        self.prefill_copied = np.zeros(self.L, dtype=np.int64)
        self.prefill_resident = np.zeros(self.L, dtype=np.int64)
        self.prefill_calls = np.zeros(self.L, dtype=np.int64)

    # ---- forward
    def forward(self, layer: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                routing=None, force: str | None = None) -> torch.Tensor:
        """x fp16|bf16 [T, H]; topk_ids [T, k] (layer-local ids, sentinel E = dropped); topk_weights [T, k].
        routing: optional precomputed (sorted_ids, expert_ids, num_post) for block = moe_block_size(T, k, E) (tests).
        force: None | "decode" | "prefill"."""
        T = x.shape[0]
        if T == 0:
            return torch.empty_like(x)
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        # SGLang masks padded rows of a graph batch to -1 (undefined in the align op): map them to the drop sentinel E
        ids = torch.where(ids < 0, torch.full_like(ids, self.E), ids).contiguous()
        w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
        w = w.contiguous()
        x = x.contiguous()
        block = marlin_moe.moe_block_size(T, ids.shape[1], self.E)
        al = routing if routing is not None else _align(ids, block, self.E)
        capturing = torch.cuda.is_current_stream_capturing()
        use_prefill = (force == "prefill") or (force is None and not capturing and T >= self.prefill_min_tokens
                                               and bool(self.staging))
        if not use_prefill:
            return self.cache.run(layer, x, w, ids, *al, block, self.cb)
        return self._prefill(layer, x, ids, w, al, block)

    def _issue(self, layer: int) -> None:
        b = layer % 2
        miss = np.nonzero(self._snapshot[layer] < 0)[0]
        self.prefill_copied[layer] += len(miss)
        self.prefill_resident[layer] += self.E - len(miss)
        bank, stg = self.store.bank(layer), self.staging[b]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.release[b])
            if len(miss):
                # runs of adjacent expert ids -> one cudaMemcpyAsync each
                brk = np.nonzero(np.diff(miss) != 1)[0] + 1
                for run in np.split(miss, brk):
                    a, z = int(run[0]), int(run[-1]) + 1
                    stg[a:z].copy_(bank[a:z], non_blocking=True)
            self.ready[b].record(self.copy_stream)
        self._pending[layer] = b

    def _prefill(self, layer, x, ids, w, al, block):
        if layer not in self._pending:          # first prefill layer of this forward: snapshot the slot map once
            self._pending.clear()
            self._snapshot = self.cache.slot_of.view(self.L, self.E).cpu().numpy()
            self._issue(layer)
        b = self._pending.pop(layer)
        if layer + 1 < self.L:
            self._issue(layer + 1)              # layer-ahead: copy l+1 while l computes
        cur = torch.cuda.current_stream()
        cur.wait_event(self.ready[b])
        rec = self.lay.record_bytes
        sl = self.cache.slot_of[layer * self.E:(layer + 1) * self.E].to(torch.int64)
        bases = torch.where(sl >= 0, self.cache.arena.data_ptr() + sl * rec, self.staging[b].data_ptr() + self._erange * rec)
        table = om.fill_table_(self.ptables[b], bases, self.cache.offs)
        y = om.run(x, w, ids, *al, block, table, self.lay, self.cb)
        self.release[b].record(cur)
        self.prefill_calls[layer] += 1
        return y

    # ---- counters
    def stats(self, reset: bool = False) -> dict:
        s = self.cache.stats.cpu().numpy()
        out = {"decode_hits": s[:, 0].tolist(), "decode_misses": s[:, 1].tolist(),
               "decode_hit_rate": float(s[:, 0].sum() / max(1, s.sum())),
               "prefill_copied": self.prefill_copied.tolist(), "prefill_resident": self.prefill_resident.tolist(),
               "prefill_calls": self.prefill_calls.tolist(), "slots": self.cache.S,
               "resident": int((self.cache.owner >= 0).sum())}
        if reset:
            self.cache.stats.zero_()
            self.prefill_copied[:] = 0; self.prefill_resident[:] = 0; self.prefill_calls[:] = 0
        return out
