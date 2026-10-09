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
                 prefill_min_tokens: int = 192, staging: bool = True, device=None, staging_parts: int = 1,
                 prefill_subchunk: int = 1 << 30):
        self.store, self.lay, self.cb, self.top_k = store, store.lay, codebook, top_k
        self.L, self.E = store.L, store.E
        dev = torch.device(device or "cuda")
        self.device = dev
        self.cache = om.ExpertCache(self.L, self.E, self.lay, [store.base_address(i) for i in range(self.L)], slots, dev)
        self.prefill_min_tokens = prefill_min_tokens
        rec = self.lay.record_bytes
        # staging_parts P: the layer's experts are staged in P parts through 2 buffers of E / P records (P = 1: two full
        # layer buffers, 1.91 GB; P = 4: 0.48 GB), the grouped kernel runs once per part (K06b)
        if self.E % staging_parts:
            raise ValueError("staging_parts must divide the expert count")
        import os as _os
        # K07: fused decode prologue (SGLANG_EXL3_OFFLOAD_FUSED=1): one launch replaces id clean + align + cache step +
        # commit per layer at decode sizes
        self.fused_decode = _os.environ.get("SGLANG_EXL3_OFFLOAD_FUSED", "0") == "1"
        # K08: fp16-accumulate grouped GEMM for the staged prefill (SGLANG_EXL3_MOE_PREFILL_FP16_ACC=1): cb + 10
        self.prefill_acc16 = _os.environ.get("SGLANG_EXL3_MOE_PREFILL_FP16_ACC", "0") == "1"
        # K08: part-wise prefill touching only each part's slots + one combine (SGLANG_EXL3_OFFLOAD_COMPACT_PARTS=1)
        self.compact_parts = _os.environ.get("SGLANG_EXL3_OFFLOAD_COMPACT_PARTS", "0") == "1"
        self.P = staging_parts
        self.subchunk = prefill_subchunk     # tokens per GEMM pass inside a staged layer (bounds activation memory)
        self.part_e = self.E // staging_parts
        self.staging = [torch.empty((self.part_e, rec), dtype=torch.uint8, device=dev) for _ in range(2)] if staging else []
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
        # decode routing per (layer, expert) for the RAM-tier pinner (kernels/offload_pin.py); column E counts dropped ids
        self.route_counts = torch.zeros((self.L, self.E + 1), dtype=torch.int32, device=dev)
        self._ones = torch.ones(4096, dtype=torch.int32, device=dev)
        self.pos = None                                    # [L, E + 1] original id -> stored position (sorted store)
        self.pinned_bands = {}                             # layer -> (lo, hi) registered bytes of the bank (offload_pin)
        self.mask_tau = 0.0

    def _stage(self, layer: int, stg: torch.Tensor, d0: int, bank: torch.Tensor, s0: int, n: int) -> None:
        """stg[d0:d0+n] <- bank[s0:s0+n] (records) as cudaMemcpyAsync calls that never span the edge of the layer's
        registered (pinned) band: a copy from partly registered host memory is rejected (cudaErrorInvalidValue)."""
        band = self.pinned_bands.get(layer)
        if band is None:
            stg[d0:d0 + n].copy_(bank[s0:s0 + n], non_blocking=True)
            return
        rec = self.lay.record_bytes
        sb, db = bank.view(-1), stg.view(-1)
        a, z = s0 * rec, (s0 + n) * rec
        cuts = sorted({a, z, *(c for c in band if a < c < z)})
        for u, v in zip(cuts, cuts[1:]):
            db[(d0 - s0) * rec + u:(d0 - s0) * rec + v].copy_(sb[u:v], non_blocking=True)

    def set_mask(self, tau: float, band_pos: dict) -> None:
        """Lossy decode option (SGLANG_EXL3_OFFLOAD_MASK_TAU > 0): a decode pick whose expert is in neither fast tier
        (not resident in the VRAM cache, not in the layer's pinned RAM band) and whose routing weight is below tau is
        dropped and the kept weights renormalised, so cold experts are not faulted in from the file. Prefill is exact."""
        self.mask_tau = tau
        lo = torch.zeros(self.L, dtype=torch.int64)
        hi = torch.zeros(self.L, dtype=torch.int64)
        for l, (a, b) in band_pos.items():
            lo[l], hi[l] = a, b
        self.band_lo, self.band_hi = lo.to(self.device), hi.to(self.device)
        self.mask_stats = torch.zeros(2, dtype=torch.int64, device=self.device)     # dropped picks, valid picks

    def _mask_cold(self, layer: int, ids: torch.Tensor, w: torch.Tensor):
        E = self.E
        t = ids.to(torch.int64)
        valid = t < E
        safe = torch.where(valid, t, torch.zeros_like(t))
        resident = self.cache.slot_of[layer * E + safe] >= 0
        pinned = (safe >= self.band_lo[layer]) & (safe < self.band_hi[layer])
        drop = valid & ~resident & ~pinned & (w < self.mask_tau)
        keep_w = torch.where(drop, torch.zeros_like(w), w)
        s = keep_w.sum(-1, keepdim=True)
        w2 = torch.where(s > 0, keep_w * (w.sum(-1, keepdim=True) / s.clamp_min(1e-12)), w)
        ids2 = torch.where(drop, torch.full_like(t, E), t).to(ids.dtype)
        self.mask_stats[0] += drop.sum()
        self.mask_stats[1] += valid.sum()
        return ids2, w2

    def set_order(self, pos) -> None:
        """Serve a store whose records are in `pos` order (pos[l, e] = stored position of original expert e). Column E
        keeps the drop sentinel. Routing counts stay in original ids."""
        p = torch.as_tensor(np.asarray(pos), dtype=torch.int32)
        self.pos = torch.cat([p, torch.full((self.L, 1), self.E, dtype=torch.int32)], 1).to(self.device).contiguous()

    # ---- forward
    def forward(self, layer: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                routing=None, force: str | None = None) -> torch.Tensor:
        """x fp16|bf16 [T, H]; topk_ids [T, k] (layer-local ids, sentinel E = dropped); topk_weights [T, k].
        routing: optional precomputed (sorted_ids, expert_ids, num_post) for block = moe_block_size(T, k, E) (tests).
        force: None | "decode" | "prefill"."""
        T = x.shape[0]
        if T == 0:
            return torch.empty_like(x)
        if T < self.prefill_min_tokens or torch.cuda.is_current_stream_capturing():   # decode only: prefill touches all
            flat = topk_ids.reshape(-1)
            if flat.numel() <= self._ones.numel():
                flat = torch.where((flat < 0) | (flat > self.E), torch.full_like(flat, self.E), flat).to(torch.int64)
                self.route_counts[layer].index_add_(0, flat, self._ones[: flat.numel()])
        if self.pos is not None:                           # original ids -> stored positions (graph-safe gather)
            t = topk_ids.to(torch.int64)
            t = torch.where((t < 0) | (t > self.E), torch.full_like(t, self.E), t)
            topk_ids = self.pos[layer][t]
        if self.mask_tau > 0 and (T < self.prefill_min_tokens or torch.cuda.is_current_stream_capturing()):
            topk_ids, topk_weights = self._mask_cold(layer, topk_ids, topk_weights.float())
        if self.fused_decode and force != "prefill" and T * topk_ids.shape[1] <= 1024 and \
                marlin_moe.moe_block_size(T, topk_ids.shape[1], self.E) == 8 and routing is None and \
                (force == "decode" or torch.cuda.is_current_stream_capturing() or T < self.prefill_min_tokens):
            w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
            return self.cache.run_fused(layer, x.contiguous(), w.contiguous(), topk_ids, self.cb)
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
        if self.P > 1 or T > self.subchunk:
            return self._prefill_parts(layer, x, ids, w, block)
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
                    self._stage(layer, stg, a, bank, a, z - a)
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
        y = om.run(x, w, ids, *al, block, table, self.lay, self._pcb(block))
        self.release[b].record(cur)
        self.prefill_calls[layer] += 1
        return y

    # ---- part-wise staged prefill (staging_parts > 1)
    def _issue_part(self, layer: int, part: int) -> None:
        g = layer * self.P + part
        b = g % 2
        lo = part * self.part_e
        miss = np.nonzero(self._snapshot[layer, lo:lo + self.part_e] < 0)[0]
        self.prefill_copied[layer] += len(miss)
        self.prefill_resident[layer] += self.part_e - len(miss)
        bank, stg = self.store.bank(layer), self.staging[b]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.release[b])
            if len(miss):
                brk = np.nonzero(np.diff(miss) != 1)[0] + 1
                for run in np.split(miss, brk):
                    a, z = int(run[0]), int(run[-1]) + 1
                    self._stage(layer, stg, a, bank, lo + a, z - a)
            self.ready[b].record(self.copy_stream)
        self._pending[(layer, part)] = b

    def _prefill_parts(self, layer, x, ids, w, block):
        """Expert part p (staging buffer (layer * P + p) % 2) x token sub-chunk c (<= subchunk tokens): had_in -> gate/up ->
        glu -> down on the sub-chunk's slots routed to part p, then y32[c] += combine (fp32). Activation memory is that of
        one sub-chunk; the staging copy of the next part overlaps the current part's GEMMs."""
        mod = om._mod()
        P, pe, E, rec, lay = self.P, self.part_e, self.E, self.lay.record_bytes, self.lay
        if (layer, 0) not in self._pending:
            self._pending.clear()
            self._snapshot = self.cache.slot_of.view(self.L, E).cpu().numpy()
            self._issue_part(layer, 0)
        T, H = x.shape
        k = ids.shape[1]
        S = min(T, self.subchunk)
        if self.compact_parts and S >= T:
            return self._prefill_parts_compact(layer, x, ids, w, block)
        I = lay.inter
        f16 = dict(dtype=torch.float16, device=x.device)
        xh = torch.empty((2 * S * k, H), **f16)
        gu = torch.empty((S * k, 2 * I), **f16)
        act, xd = torch.empty((S * k, I), **f16), torch.empty((S * k, I), **f16)
        y32 = torch.zeros((T, H), dtype=torch.float32, device=x.device)
        sl = self.cache.slot_of[layer * E:(layer + 1) * E].to(torch.int64)
        host_rows = self.cache.host_bases[layer] + self._erange * rec
        part_of = torch.div(ids, pe, rounding_mode="floor")
        cur = torch.cuda.current_stream()
        for p in range(P):
            b = self._pending.pop((layer, p))
            nxt = (layer, p + 1) if p + 1 < P else (layer + 1, 0)
            if nxt[0] < self.L:
                self._issue_part(*nxt)
            cur.wait_event(self.ready[b])
            inpart = (self._erange >= p * pe) & (self._erange < (p + 1) * pe)
            stage = self.staging[b].data_ptr() + (self._erange - p * pe) * rec
            bases = torch.where(sl >= 0, self.cache.arena.data_ptr() + sl * rec, torch.where(inpart, stage, host_rows))
            table = om.fill_table_(self.ptables[b], bases, self.cache.offs)
            ids_p = torch.where(part_of == p, ids, torch.full_like(ids, E))
            for c0 in range(0, T, S):
                c1 = min(T, c0 + S)
                n = (c1 - c0) * k
                ic = ids_p[c0:c1].contiguous()
                blk = marlin_moe.moe_block_size(c1 - c0, k, E)
                s_ids, e_ids, npost = _align(ic, blk, E)
                xhc = xh[: 2 * n]
                gc, ac, xc = gu[:n], act[:n], xd[:n]
                mod.moe_had_in_ptr(x[c0:c1], table, om.F_SUH13, 2, ic, xhc)
                mod.moe_gemm_ptr(xhc, gc, table, om.F_W13, om.F_SVH13, lay.bits, s_ids, e_ids, npost, blk, I, self._pcb(blk))
                mod.moe_glu_had_in_ptr(gc, table, om.F_SUH2, ic, ac, xc)
                yd = xhc[:n]
                mod.moe_gemm_ptr(xc, yd, table, om.F_W2, om.F_SVH2, lay.bits, s_ids, e_ids, npost, blk, 0, self._pcb(blk))
                mod.moe_combine_acc(yd, w[c0:c1].contiguous(), ic, E, y32[c0:c1])
            self.release[b].record(cur)
        self.prefill_calls[layer] += 1
        return y32.to(x.dtype)

    def _prefill_parts_compact(self, layer, x, ids, w, block):
        """K08: every expert part runs had_in / glu over ITS slots only (compact kernels over the part's align output);
        its down GEMM writes those slots' rows of the gate slab (already consumed by its gate/up GEMM), one moe_combine at
        the end -> no fp32 accumulator, no per-part sweep over all slots."""
        mod = om._mod()
        P, pe, E, rec, lay = self.P, self.part_e, self.E, self.lay.record_bytes, self.lay
        T, H = x.shape
        k = ids.shape[1]
        slots, I = T * k, lay.inter
        f16 = dict(dtype=torch.float16, device=x.device)
        xh = torch.empty((2 * slots, H), **f16)
        gu = torch.empty((slots, 2 * I), **f16)
        act, xd = torch.empty((slots, I), **f16), torch.empty((slots, I), **f16)
        yd = xh[:slots]
        sl = self.cache.slot_of[layer * E:(layer + 1) * E].to(torch.int64)
        host_rows = self.cache.host_bases[layer] + self._erange * rec
        part_of = torch.div(ids, pe, rounding_mode="floor")
        cur = torch.cuda.current_stream()
        cb = self._pcb(block)
        for p in range(P):
            b = self._pending.pop((layer, p))
            nxt = (layer, p + 1) if p + 1 < P else (layer + 1, 0)
            if nxt[0] < self.L:
                self._issue_part(*nxt)
            cur.wait_event(self.ready[b])
            inpart = (self._erange >= p * pe) & (self._erange < (p + 1) * pe)
            stage = self.staging[b].data_ptr() + (self._erange - p * pe) * rec
            bases = torch.where(sl >= 0, self.cache.arena.data_ptr() + sl * rec, torch.where(inpart, stage, host_rows))
            table = om.fill_table_(self.ptables[b], bases, self.cache.offs)
            ids_p = torch.where(part_of == p, ids, torch.full_like(ids, E)).contiguous()
            s_ids, e_ids, npost = _align(ids_p, block, E)
            mod.moe_had_in_ptr_compact(x, table, om.F_SUH13, ids_p, s_ids, npost, xh)
            mod.moe_gemm_ptr(xh, gu, table, om.F_W13, om.F_SVH13, lay.bits, s_ids, e_ids, npost, block, I, cb)
            mod.moe_glu_had_in_ptr_compact(gu, table, om.F_SUH2, ids_p, s_ids, npost, act, xd)
            mod.moe_gemm_ptr(xd, yd, table, om.F_W2, om.F_SVH2, lay.bits, s_ids, e_ids, npost, block, 0, cb)
            self.release[b].record(cur)
        y = torch.empty_like(x)
        mod.moe_combine(yd, w, ids, E, y)
        self.prefill_calls[layer] += 1
        return y

    def _pcb(self, block: int) -> int:
        return self.cb + 10 if (self.prefill_acc16 and block > 8 and self.cb in (1, 2)) else self.cb

    # ---- counters
    def stats(self, reset: bool = False) -> dict:
        s = self.cache.stats.cpu().numpy()
        out = {"decode_hits": s[:, 0].tolist(), "decode_misses": s[:, 1].tolist(),
               "masked_picks": int(self.mask_stats[0]) if self.mask_tau > 0 else 0,
               "masked_of": int(self.mask_stats[1]) if self.mask_tau > 0 else 0,
               "decode_hit_rate": float(s[:, 0].sum() / max(1, s.sum())),
               "prefill_copied": self.prefill_copied.tolist(), "prefill_resident": self.prefill_resident.tolist(),
               "prefill_calls": self.prefill_calls.tolist(), "slots": self.cache.S,
               "resident": int((self.cache.owner >= 0).sum())}
        if reset:
            self.cache.stats.zero_()
            self.prefill_copied[:] = 0; self.prefill_resident[:] = 0; self.prefill_calls[:] = 0
        return out
