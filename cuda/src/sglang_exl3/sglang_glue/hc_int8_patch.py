"""SGLANG_EXL3_HC_INT8: hyper-connection mix weights (GatedResidual input_mix_weight_down / _up, 1.27 GB bf16 on
Qwen3.8-Flash-Next) quantised to int8 right after load_weights; the bf16 copies are freed (before SGLang sizes KV /
the expert cache), decode-size mixes run kernels/hc_mix_int8.fused_hc_mix_int8, larger (prefill) mixes dequantise the
site transiently.

    SGLANG_EXL3_HC_INT8=0 | off   (default) SGLang's bf16 path, unchanged
    SGLANG_EXL3_HC_INT8=row       per-row int8 (0.64 GB; K06: rel err 1.0% vs fp32, 97 sites 1.48 ms vs 2.12 ms bf16)
    SGLANG_EXL3_HC_INT8=g128      W_down per 128-group / W_up per 64-group (0.67 GB; rel err 0.57%, 1.68 ms)
install() is called from plugin.activate(); it patches GatedResidual.mix and wraps load_weights of the qwen4_exp model
classes (main model and MTP). block_inject_weight (combine) and hc_norm stay bf16.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)
_MODES = {"row": (0, 0), "1": (0, 0), "g128": (128, 64)}


def _mode():
    m = os.environ.get("SGLANG_EXL3_HC_INT8", "0").strip().lower()
    return None if m in ("", "0", "off", "none") else m


def quantize_model_hc(model: torch.nn.Module, mode: str) -> dict:
    from sglang.srt.layers.hyperconnection import GatedResidual
    from ..kernels.hc_mix_int8 import HcInt8Weights
    gd, gu = _MODES[mode]
    n, before, after = 0, 0, 0
    for m in model.modules():
        if isinstance(m, GatedResidual) and hasattr(m, "input_mix_weight_down") and getattr(m, "_exl3_hcq", None) is None:
            wd, wu = m.input_mix_weight_down.weight, m.input_mix_weight_up.weight
            if wd.numel() == 0:
                continue
            before += (wd.numel() + wu.numel()) * wd.element_size()
            q = HcInt8Weights(wd.data, wu.data, gd, gu)
            after += q.nbytes()
            m._exl3_hcq = q
            empty = lambda w: torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False)
            m.input_mix_weight_down.weight = empty(wd)
            m.input_mix_weight_up.weight = empty(wu)
            m._mix_up_weight_padded = None
            n += 1
    torch.cuda.empty_cache()
    stats = {"sites": n, "GB_before": before / 1e9, "GB_after": after / 1e9, "mode": mode}
    logger.info("sglang-exl3: HC mix int8 (%s): %d sites, %.2f GB -> %.2f GB", mode, n, before / 1e9, after / 1e9)
    return stats


def _patched_mix(self, hyper_input):
    q = getattr(self, "_exl3_hcq", None)
    if q is None:
        return _ORIG_MIX(self, hyper_input)
    from ..kernels.hc_mix_int8 import MAX_ROWS, fused_hc_mix_int8, hc_mix_int8_torch
    if hyper_input.shape[0] == 0:
        mixed = hyper_input.new_empty((*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype)
        return mixed, (hyper_input, hyper_input)
    if self.config.hc_per_branch_norm:
        normed = self.hc_norm(hyper_input)
    else:
        normed = self.hc_norm(hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))).flatten(-2)
    if normed.dim() == 2 and normed.shape[0] <= MAX_ROWS and normed.is_contiguous() \
            and normed.dtype in (torch.bfloat16, torch.float16):
        mixed = fused_hc_mix_int8(normed, q, self.hc_count, self.hidden_size)
    else:
        mixed = hc_mix_int8_torch(normed.reshape(-1, normed.shape[-1]), q, self.hc_count, self.hidden_size)
        mixed = mixed.view(*normed.shape[:-1], self.hidden_size)
    return mixed.to(self.params_dtype), (hyper_input, normed)


_ORIG_MIX = None


def install() -> bool:
    global _ORIG_MIX
    mode = _mode()
    if mode is None:
        return False
    if mode not in _MODES:
        raise ValueError(f"SGLANG_EXL3_HC_INT8={mode!r}: expected 0 | row | g128")
    try:
        from sglang.srt.layers import hyperconnection as hcm
    except ImportError:  # pragma: no cover
        return False
    if _ORIG_MIX is None:
        _ORIG_MIX = hcm.GatedResidual.mix
        hcm.GatedResidual.mix = _patched_mix
    wrapped = []
    for modname, clsname in (("sglang.srt.models.qwen4_exp", "Qwen4ExpForConditionalGeneration"),
                             ("sglang.srt.models.qwen4_exp", "Qwen4ExpForCausalLM"),
                             ("sglang.srt.models.qwen4_exp_mtp", "Qwen4ExpForCausalLMMTP")):
        try:
            mod = __import__(modname, fromlist=[clsname])
            cls = getattr(mod, clsname)
        except (ImportError, AttributeError):
            continue
        orig = cls.load_weights
        if getattr(orig, "_exl3_hc_wrapped", False):
            continue

        def load_weights(self, weights, _orig=orig):
            r = _orig(self, weights)
            quantize_model_hc(self, mode)
            return r
        load_weights._exl3_hc_wrapped = True
        cls.load_weights = load_weights
        wrapped.append(clsname)
    logger.info("sglang-exl3: HC int8 mix installed (%s), load_weights wrapped: %s", mode, wrapped)
    return True
