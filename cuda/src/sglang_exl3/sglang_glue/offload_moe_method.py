"""Exl3OffloadMoEMethod: SGLang FusedMoE quant method with the routed experts offloaded to pinned host memory and a
global GPU expert cache (kernels/offload_runtime.OffloadRuntime). Selected by Exl3Config.get_quant_method when
SGLANG_EXL3_MOE_OFFLOAD=1 for main-model MoE layers (the MTP draft layer keeps Exl3MoEMethod).

Weights: the expert placeholders are registered like Exl3MoEMethod's (so SGLang's loader finds every expert tensor),
but their loader DISCARDS the tensors: the experts are read once, straight from the shards, by HostExpertStore when
the first layer finishes loading (all layers in one pass). To also skip SGLang's own read of those tensors, filter the
weight iterator with `is_offloaded_expert_key(name)`.

Knobs (env): SGLANG_EXL3_EXPERT_CACHE_GB (slot-pool byte budget, GB) or SGLANG_EXL3_OFFLOAD_SLOTS (slots) or
SGLANG_EXL3_OFFLOAD_CACHE_GB (default 8 GB),
SGLANG_EXL3_OFFLOAD_PREFILL_MIN (tokens from which the staged prefill path is used, default 192 = K05 crossover),
SGLANG_EXL3_OFFLOAD_STATS_EVERY (log per-layer hit rates every N decode forwards of layer 0; 0 = off),
SGLANG_EXL3_OFFLOAD_STORE=hmm + SGLANG_EXL3_OFFLOAD_STORE_DIR (the expert store as one file in that directory, read by
the GPU through HMM instead of pinned RAM: see kernels/offload_store.py; the directory must be on a local NVMe SSD).
"""
from __future__ import annotations

import logging
import os
import re

import torch
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

from ..kernels import offload_moe as om
from ..kernels.offload_runtime import OffloadRuntime
from ..kernels.offload_store import HostExpertStore
from .moe import Exl3MoEMethod

logger = logging.getLogger(__name__)
_RUNTIMES: dict = {}
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_KEY_RE = re.compile(r"^(.*layers\.)(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.")


def offload_enabled() -> bool:
    return os.environ.get("SGLANG_EXL3_MOE_OFFLOAD", "0") in ("1", "gpu_cache")


def offload_stats(reset: bool = False) -> dict:
    """Per-layer counters of every live runtime: decode hits / misses (device counters), prefill copied / resident
    experts, slots, resident slots. Host sync: call outside CUDA-graph capture (e.g. per request for logging)."""
    return {f"{k[0]}@cuda:{k[1]}": rt.stats(reset) for k, rt in _RUNTIMES.items()}


def is_offloaded_expert_key(name: str) -> bool:
    """True for checkpoint / SGLang weight names of main-model routed experts (their data is read by the store)."""
    return offload_enabled() and ".mlp.experts." in name \
        and not name.startswith("mtp.") and "shared_expert" not in name


def _slots_from_env(record_bytes: int, staging_bytes: int = 0) -> int:
    v = os.environ.get("SGLANG_EXL3_EXPERT_CACHE_GB", "")
    if v.strip().lower() == "auto":
        # auto-fit: free VRAM now (weights loaded, before KV / mamba / graphs are sized) minus the prefill staging and
        # a reserve for everything SGLang allocates later (SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB; the default is
        # calibrated on the 3090 serve config: KV 210k fp8, mamba 16, 8k chunks, graphs bs 1-3)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        reserve = float(os.environ.get("SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB", "8.9")) * 1e9
        nbytes = max(0, free - staging_bytes - reserve)
        logger.info("EXL3 offload: expert cache auto-fit: free %.2f GB of %.2f, staging %.2f GB, reserve %.2f GB -> %.2f GB",
                    free / 1e9, total / 1e9, staging_bytes / 1e9, reserve / 1e9, nbytes / 1e9)
        return int(nbytes // record_bytes)
    if v:                                                          # integration contract: byte budget
        return int(float(v) * 1e9 // record_bytes)
    if os.environ.get("SGLANG_EXL3_OFFLOAD_SLOTS"):
        return int(os.environ["SGLANG_EXL3_OFFLOAD_SLOTS"])
    gb = float(os.environ.get("SGLANG_EXL3_OFFLOAD_CACHE_GB", "8"))
    return int(gb * 1e9 // record_bytes)


def get_runtime(config, num_experts: int, hidden: int, inter: int, bits: int, codebook: int) -> OffloadRuntime:
    """One OffloadRuntime per (model path, device); built (store loaded + pinned, cache allocated) on first use."""
    dev = torch.cuda.current_device()
    key = (config.model_path if hasattr(config, "model_path") else id(config), dev)
    rt = _RUNTIMES.get(key)
    if rt is not None:
        return rt
    layers, prefix = set(), None
    for k in config.modules:
        m = _KEY_RE.match(k + ".")
        if m and not k.startswith("mtp."):
            layers.add(int(m.group(2))); prefix = m.group(1)
    layers = sorted(layers)
    model_dir = getattr(config, "model_path", None) or config.path
    store_path, pos = None, None
    if os.environ.get("SGLANG_EXL3_OFFLOAD_STORE", "pinned") == "hmm":
        d = os.environ.get("SGLANG_EXL3_OFFLOAD_STORE_DIR", "/nvx")
        store_path = os.path.join(d, f"experts-{os.path.basename(os.path.normpath(model_dir))}-k{bits}.bin")
    store = HostExpertStore(model_dir, layers, num_experts, hidden, inter, bits, prefix=prefix, store_path=store_path)
    st = store.load()
    if store_path:
        logger.info("EXL3 offload: expert store %s (%s, %.1f GB, read through HMM: page cache = RAM tier)", store_path,
                    "mapped" if store.prebuilt else "written", store.L * store.stride / 1e9)
        # popularity-ordered copy of the store (experts renumbered per layer, most routed first) so that the pinned RAM
        # tier is one contiguous band per layer; ids are remapped at the runtime's entry (OffloadRuntime.pos)
        from ..kernels import offload_pin as op
        profile = store_path + ".profile.npy"
        sorted_path = store_path[:-4] + ".sorted.bin"
        if os.environ.get("SGLANG_EXL3_OFFLOAD_SORT", "auto") != "0":
            score = op.load_profile(profile, store.L, num_experts)
            pos = op.sorted_order(sorted_path, store._meta)
            if score is not None and (pos is None or os.environ.get("SGLANG_EXL3_OFFLOAD_SORT") == "repack"):
                pos = op.order_from_profile(score)
                secs = op.repack_store(store, sorted_path, pos)
                logger.info("EXL3 offload: store repacked by routing profile -> %s in %.1f s", sorted_path, secs)
            if pos is not None:
                store.release()
                store = HostExpertStore(model_dir, layers, num_experts, hidden, inter, bits, prefix=prefix,
                                        store_path=sorted_path)
                if not store.prebuilt:
                    raise RuntimeError(f"{sorted_path}: sorted store marker does not match its layout")
                logger.info("EXL3 offload: serving the popularity-ordered store %s", sorted_path)
    parts = int(os.environ.get("SGLANG_EXL3_OFFLOAD_STAGING_PARTS", "1"))
    staging_bytes = 2 * (-(-num_experts // parts)) * store.lay.record_bytes
    slots = _slots_from_env(store.lay.record_bytes, staging_bytes)
    ps = {}
    if store_path and pos is not None:
        from ..kernels.offload_pin import load_profile, pin_budget_bytes, pin_static
        pb = pin_budget_bytes(store.L * store.stride)
        score = load_profile(store_path + ".profile.npy", store.L, num_experts)
        if pb > 0 and score is not None:
            ps = pin_static(store, pb, score, pos, slots)
            logger.info("EXL3 offload: pinned RAM tier from the profile: %d experts (%.1f GB) in %d runs, %.1f s "
                        "(top %d left to the VRAM cache, %d register errors)", ps["pinned"], ps["pinned_gb"], ps["runs"],
                        ps["seconds"], ps["skipped_top"], ps["errors"])
    rt = OffloadRuntime(store, slots, codebook, prefill_min_tokens=int(os.environ.get("SGLANG_EXL3_OFFLOAD_PREFILL_MIN", "192")),
                        staging_parts=int(os.environ.get("SGLANG_EXL3_OFFLOAD_STAGING_PARTS", "1")),
                        prefill_subchunk=int(os.environ.get("SGLANG_EXL3_OFFLOAD_PREFILL_SUBCHUNK", str(1 << 30))))
    if pos is not None:
        rt.set_order(pos)
    if ps.get("bands"):
        rt.pinned_bands = ps["bands"]             # staged prefill copies must not cross a registration edge
    rt.set_mask(float(os.environ.get("SGLANG_EXL3_OFFLOAD_MASK_TAU", "0")), ps.get("band_pos", {}))
    logger.info("EXL3 offload: %d layers x %d experts " + ("in the file store" if store_path else "pinned") + " (%.1f GB) in %.1fs (read wait %.1fs, relayout %.1fs, "
                "register %.1fs); cache %d slots (%.1f GB)", store.L, num_experts, store.L * store.bank_bytes / 1e9,
                st.seconds_total, st.seconds_read_wait, st.seconds_relayout, st.seconds_register, slots,
                slots * store.lay.record_bytes / 1e9)
    if store_path:
        from ..kernels.offload_pin import StorePinner, lock_budget_bytes
        budget = 0 if pos is not None else lock_budget_bytes(store.L * store.stride)   # 0 (default): profile only
        rt.pinner = StorePinner(store, rt, budget, store_path + ".profile.npy",
                                float(os.environ.get("SGLANG_EXL3_OFFLOAD_LOCK_INTERVAL_S", "20"))).start()
    _RUNTIMES[key] = rt
    return rt


class Exl3OffloadMoEMethod(Exl3MoEMethod):
    def __init__(self, config, prefix: str, experts: dict):
        super().__init__(config, prefix, experts)
        m = _LAYER_RE.search(prefix)
        if m is None:
            raise ValueError(f"{prefix}: cannot find the layer index for the offloaded MoE")
        self.layer_id = int(m.group(1))
        self._stats_every = int(os.environ.get("SGLANG_EXL3_OFFLOAD_STATS_EVERY", "0"))
        self._calls = 0

    def _make_loader(self, layer, suffix):
        def load(param, loaded_weight, weight_name=None, shard_id=None, expert_id=None, *a, **k):
            layer.exl3_discarded = getattr(layer, "exl3_discarded", 0) + 1      # the store reads the shards itself
        return load

    def process_weights_after_loading(self, layer) -> None:
        n = layer.exl3_num_experts
        k, nn_, twice, codebook = self.infos[("gate", 0)][:4]
        bits = twice // 2
        cb = {"3inst": 0, "mcg": 1, "mul1": 2}[codebook] if isinstance(codebook, str) else int(codebook)
        rt = get_runtime(self.config, n, layer.exl3_hidden, layer.exl3_inter, bits, cb)
        layer.exl3_offload = rt
        layer.exl3_store_index = rt.store.layers.index(self.layer_id)
        for w in ("w13", "w2"):
            for suffix in ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1"):
                if hasattr(layer, f"{w}_{suffix}"):
                    delattr(layer, f"{w}_{suffix}")
        layer.exl3_pack, layer.exl3_tables = None, None
        if hasattr(layer, "exl3_store"):
            del layer.exl3_store
        # warm: the align / per-slot kernels compile here, before any graph capture
        top_k = self.moe_runner_config.top_k if self.moe_runner_config else 10
        x = torch.zeros((1, layer.exl3_hidden), dtype=torch.bfloat16, device="cuda")
        ids = torch.full((1, top_k), n, dtype=torch.int32, device="cuda")     # all dropped: no cache side effects
        w = torch.zeros((1, top_k), dtype=torch.float32, device="cuda")
        rt.forward(layer.exl3_store_index, x, ids, w, force="decode")
        torch.cuda.synchronize()

    def _experts(self, layer, x, topk_ids, topk_weights):
        rt = layer.exl3_offload
        try:
            from sglang.srt.model_executor.runner import get_is_capture_mode
            capture = bool(get_is_capture_mode())
        except Exception:  # pragma: no cover
            capture = False
        # capture mode (or stream capture): decode path only - no host sync, graph-safe
        if getattr(rt, "pinner", None) is not None and not rt.pinner.serving.is_set():
            if capture or torch.cuda.is_current_stream_capturing():
                rt.pinner.seen_capture = True
            elif getattr(rt.pinner, "seen_capture", False):
                rt.pinner.serving.set()           # first eager forward after graph capture: serving has begun
        y = rt.forward(layer.exl3_store_index, x, topk_ids, topk_weights, force="decode" if capture else None)
        if (os.environ.get("SGLANG_EXL3_OFFLOAD_LOG_PREFILL", "1") == "1" and layer.exl3_store_index == 0
                and x.shape[0] >= rt.prefill_min_tokens and not torch.cuda.is_current_stream_capturing()):
            # eager prefill of a new request: report (and reset) the decode hit rate accumulated since the last one
            # (decode runs inside CUDA graphs, where no Python code executes)
            s = rt.stats(reset=True)
            if sum(s["decode_hits"]) + sum(s["decode_misses"]):
                logger.info("EXL3 offload: decode hit rate %.4f since last prefill (%d hits, %d misses), resident %d/%d%s",
                            s["decode_hit_rate"], sum(s["decode_hits"]), sum(s["decode_misses"]), s["resident"],
                            s["slots"], f", masked {s['masked_picks']}/{s['masked_of']} picks" if s["masked_of"] else "")
        if self._stats_every and layer.exl3_store_index == 0 and x.shape[0] < rt.prefill_min_tokens \
                and not torch.cuda.is_current_stream_capturing():
            self._calls += 1
            if self._calls % self._stats_every == 0:
                s = rt.stats()
                logger.info("EXL3 offload: decode hit rate %.3f, resident %d/%d slots", s["decode_hit_rate"],
                            s["resident"], s["slots"])
        return y
