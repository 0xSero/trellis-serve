"""Exl3LinearMethod for SGLang linears and ParallelLMHead.

Per-shard storage: every checkpoint matrix (q, k, v / gate, up / GDN qkv, z) keeps its own trellis, suh, svh.
Placeholders named exactly like the checkpoint suffixes are registered as parameters so the model's unmodified
`load_weights` routes `<layer>.<suffix>` (+ shard id) to our loader; the loader keeps whole tensors per shard id.
"""
from __future__ import annotations

import logging
import os

import torch
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.utils.common import set_weight_attrs

from ..kernels import marlin, reference
from ..runtime import ops

logger = logging.getLogger(__name__)
_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")
_CODEBOOK_ID = {"3inst": 0, "mcg": 1, "mul1": 2}
_QKV = {"q": 0, "k": 1, "v": 2}
_MARLIN = os.environ.get("SGLANG_EXL3_KERNEL", "auto")   # auto | exllamav3 | marlin
_MARLIN_MAX_N = int(os.environ.get("SGLANG_EXL3_MARLIN_MAX_N", "65536"))   # 24 GB cards: keep the K=6 head lean


_knobs_done = False
TARGET_LM_HEADS: list = []        # the target's quantized lm_head(s), for the draft's hot-token head (plugin.py)


def _apply_kernel_knobs() -> None:
    """Runtime knobs of the Marlin-template kernels (once per process): SGLANG_EXL3_BLOCKS_PER_SM (1|2, default 2),
    SGLANG_EXL3_IN_HAD_INLAUNCH (0|1: input Hadamard as the cooperative prologue of the GEMM launch),
    SGLANG_EXL3_OUT_HAD_INLAUNCH (1|0), SGLANG_EXL3_HAD_WARPS (8|1)."""
    global _knobs_done
    if _knobs_done:
        return
    _knobs_done = True
    bps = int(os.environ.get("SGLANG_EXL3_BLOCKS_PER_SM", "2"))   # 2 measured +10 % per step on the 3090; per-launch fallback in the host
    if bps != 1:
        marlin._load().set_blocks_per_sm(bps)
    if os.environ.get("SGLANG_EXL3_IN_HAD_INLAUNCH", "0") == "1":
        marlin.set_in_had_inlaunch(True)
    if os.environ.get("SGLANG_EXL3_OUT_HAD_INLAUNCH", "1") == "0":
        marlin.set_out_had_inlaunch(False)
    warps = int(os.environ.get("SGLANG_EXL3_HAD_WARPS", "8"))
    if warps != 8:
        marlin.set_had_warps(warps)
    logger.info("sglang-exl3 kernel knobs: blocks_per_sm=%d in_had_inlaunch=%s out_had_inlaunch=%s had_warps=%d", bps,
                os.environ.get("SGLANG_EXL3_IN_HAD_INLAUNCH", "0"), os.environ.get("SGLANG_EXL3_OUT_HAD_INLAUNCH", "1"), warps)


def _partitions(shard_id) -> tuple[int, ...]:
    if shard_id is None:
        return ()
    if isinstance(shard_id, tuple):
        return tuple(shard_id)
    return (_QKV.get(shard_id, shard_id),)


def _unpack_signs(packed: torch.Tensor) -> torch.Tensor:
    bits = (packed.to(torch.int32).unsqueeze(1) >> torch.arange(16, device=packed.device)) & 1
    return (1.0 - 2.0 * bits.flatten()).to(torch.float16)


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, prefix: str, infos: list[tuple]):
        self.prefix = prefix
        self.infos = infos      # per source matrix, checkpoint order: (k, n, 2K, codebook, has_bias, legacy)

    # ---- weights
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
            raise NotImplementedError(f"{self.prefix}: tensor-parallel EXL3 is not implemented (single GPU only)")
        if any(info[4] for info in self.infos):
            raise NotImplementedError(f"{self.prefix}: EXL3 linears with bias are not implemented yet")
        layer.exl3_out_sizes = list(output_partition_sizes)
        layer.exl3_in_size = input_size
        layer.exl3_shards = {s: {} for s in _SUFFIXES}
        for suffix in _SUFFIXES:
            p = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            set_weight_attrs(p, {"weight_loader": self._make_loader(layer, suffix), "exl3_placeholder": True})
            layer.register_parameter(suffix, p)
        if not hasattr(layer, "weight"):
            # ParallelLMHead: get_embed_and_head / should_apply_lm_head_quant_method read `.weight`
            # zero-width (rows = vocab): SGLang's speculative token map slices head.data[hot_ids] before handing it to
            # the draft, which must work on the placeholder too (plugin.py builds the dense hot-token head)
            w = torch.nn.Parameter(torch.empty((sum(output_partition_sizes), 0), dtype=params_dtype), requires_grad=False)
            set_weight_attrs(w, {"weight_loader": self._ignore_loader, "exl3_placeholder": True})
            layer.register_parameter("weight", w)

    @staticmethod
    def _ignore_loader(param, loaded_weight, *a, **k):
        logger.warning("EXL3: ignoring a dense .weight for a quantized layer (shape %s)", tuple(loaded_weight.shape))

    def _make_loader(self, layer, suffix):
        def load(param, loaded_weight, loaded_shard_id=None, *a, **k):
            store = layer.exl3_shards[suffix]
            key = loaded_shard_id if not isinstance(loaded_shard_id, list) else tuple(loaded_shard_id)
            if key in store:
                raise ValueError(f"{self.prefix}.{suffix}: shard {key!r} loaded twice")
            dev = torch.device("cuda", torch.cuda.current_device())
            store[key] = loaded_weight.to(dev, copy=True)
        return load

    def process_weights_after_loading(self, layer) -> None:
        got = layer.exl3_shards
        ids = sorted(got["trellis"], key=lambda sid: _partitions(sid) or (0,))
        if not ids:
            # e.g. the draft model's own lm_head: never loaded (the target's lm_head module replaces it)
            logger.warning("%s: no EXL3 tensors were delivered; layer left empty", self.prefix)
            layer.exl3_count = 0
            return
        if len(ids) != len(self.infos):
            raise ValueError(f"{self.prefix}: expected {len(self.infos)} EXL3 matrices, checkpoint delivered {len(ids)} ({ids})")
        codebooks, widths = [], []
        for i, sid in enumerate(ids):
            suh, svh = got["suh"].get(sid), got["svh"].get(sid)
            if suh is None and sid in got["su"]:
                suh, svh = _unpack_signs(got["su"][sid]), _unpack_signs(got["sv"][sid])
            if suh is None or svh is None:
                raise ValueError(f"{self.prefix}: shard {sid!r} has a trellis but no scale vectors")
            codebook = "mcg" if sid in got["mcg"] else "mul1" if sid in got["mul1"] else "3inst"
            trellis = got["trellis"][sid].contiguous()
            k, n, twice, declared = self.infos[i][:4]
            if (trellis.shape[0] * 16, trellis.shape[1] * 16, trellis.shape[2] // 8, codebook) != (k, n, twice, declared):
                raise ValueError(f"{self.prefix}: shard {sid!r} {tuple(trellis.shape)}/{codebook} does not match the header scan {(k, n, twice, declared)}")
            parts = _partitions(sid)
            widths.append(sum(layer.exl3_out_sizes[p] for p in parts) if parts else sum(layer.exl3_out_sizes))
            codebooks.append(_CODEBOOK_ID[codebook])
            for name, t in (("trellis", trellis), ("suh", suh.contiguous()), ("svh", svh.contiguous())):
                layer.register_buffer(f"exl3_{name}_{i}", t, persistent=False)
        if sum(widths) != sum(layer.exl3_out_sizes):
            raise ValueError(f"{self.prefix}: shards cover {sum(widths)} of {sum(layer.exl3_out_sizes)} output columns")
        for suffix in _SUFFIXES:
            delattr(layer, suffix)
        del layer.exl3_shards
        layer.exl3_count, layer.exl3_codebooks, layer.exl3_widths = len(ids), codebooks, widths
        mats = [[getattr(layer, f"exl3_{name}_{i}") for i in range(len(ids))] for name in ("trellis", "suh", "svh")]
        # our Marlin-template kernel (sm_80+), one launch set per fused group
        layer.exl3_marlin_count, layer.exl3_marlin_ends = 0, []
        layer.exl3_trellis_resident = True
        if _MARLIN in ("auto", "marlin") and marlin.probe().available:
            ok, why = marlin.supports(mats[0], widths, codebooks)
            if ok and sum(widths) > _MARLIN_MAX_N:
                # K5: K = 3, 4, 5 and 6 groups route here via marlin.supports; the cap keeps the lm_head (K=6 or K=5) on ExLlamaV3
                ok, why = False, f"n={sum(widths)} above SGLANG_EXL3_MARLIN_MAX_N={_MARLIN_MAX_N} (lm_head stays on ExLlamaV3's kernel)"
            if ok:
                _apply_kernel_knobs()
                *tensors, layer.exl3_marlin_ends = marlin.prepare(*mats)
                for j, t in enumerate(tensors):
                    layer.register_buffer(f"exl3_marlin_{j}", t, persistent=False)
                layer.exl3_marlin_count = len(tensors)
                if os.environ.get("SGLANG_EXL3_KEEP_TRELLIS", "0") != "1":
                    # 24 GB card: the repacked form is the only resident copy (unpacked on demand for prefill)
                    for i in range(len(ids)):
                        delattr(layer, f"exl3_trellis_{i}")
                    mats[0] = []
                    layer.exl3_trellis_resident = False
                    torch.cuda.empty_cache()
            else:
                logger.info("%s: ExLlamaV3 kernels (%s)", self.prefix, why)
        elif _MARLIN == "marlin":
            raise RuntimeError("SGLANG_EXL3_KERNEL=marlin but trellis_exl3_kernels is not importable")
        # ExLlamaV3 sliced multi-matrix launch for fused groups (one input Hadamard per source)
        layer.exl3_sliced_count, layer.exl3_sliced_min_rows = 0, 1
        if (len(ids) > 1 and len(set(codebooks)) == 1 and os.environ.get("SGLANG_EXL3_SLICED", "1") == "1"
                and layer.exl3_marlin_count == 0):
            try:
                group = reference.SlicedGroup(*mats, codebooks[0], max_rows=int(os.environ.get("SGLANG_EXL3_SLICED_MAX_ROWS", "32")))
            except ValueError as e:
                logger.info("%s: no sliced launch (%s)", self.prefix, e)
            else:
                layer.exl3_sliced_keepalive = group
                for j, t in enumerate(group.tables()):
                    layer.register_buffer(f"exl3_sliced_{j}", t, persistent=False)
                layer.exl3_sliced_count = len(group.tables())
                layer.exl3_sliced_min_rows = 3 if max(group.widths) >= 16384 else 1
        layer.exl3_has_dense = False
        if self.prefix.endswith("lm_head") and not self.prefix.startswith("mtp"):
            TARGET_LM_HEADS.append(layer)

    # ---- forward
    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        n = layer.exl3_count
        if n == 0:
            raise RuntimeError(f"{self.prefix}: EXL3 layer has no weights (never loaded)")
        y = ops.linear(x, [getattr(layer, f"exl3_trellis_{i}") for i in range(n)] if layer.exl3_trellis_resident else [],
                       [getattr(layer, f"exl3_suh_{i}") for i in range(n)],
                       [getattr(layer, f"exl3_svh_{i}") for i in range(n)],
                       [],
                       [getattr(layer, f"exl3_sliced_{j}") for j in range(layer.exl3_sliced_count)],
                       layer.exl3_sliced_min_rows,
                       [getattr(layer, f"exl3_marlin_{j}") for j in range(layer.exl3_marlin_count)], layer.exl3_marlin_ends,
                       layer.exl3_codebooks, layer.exl3_widths)
        return y if bias is None else y + bias


class Exl3Dense(torch.nn.Module):
    """Drop-in for a plain `nn.Linear` whose checkpoint tensors are EXL3 (the MTP head's `fc`).
    Returns a tensor, like nn.Linear."""

    def __init__(self, in_features: int, out_features: int, info: tuple, prefix: str, params_dtype=torch.bfloat16):
        super().__init__()
        self.prefix = prefix
        self.method = Exl3LinearMethod(prefix, [info])
        self.method.create_weights(self, in_features, [out_features], in_features, out_features, params_dtype)
        # `weight` placeholder is registered by create_weights (no .weight existed)

    def process_weights_after_loading(self):
        self.method.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.method.apply(self, x)
