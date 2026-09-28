"""Exl3Config: what is quantized, per tensor, from the checkpoint's safetensors headers (never from JSON hints)."""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from trellis_core.format import load_manifest

logger = logging.getLogger(__name__)
_MODEL_PATH_ENV = "SGLANG_EXL3_MODEL_PATH"


class Exl3Config(QuantizationConfig):
    """One instance per model load. `modules` maps a checkpoint module key (HF naming, e.g.
    `model.language_model.layers.3.self_attn.q_proj`, `lm_head`, `mtp.layers.0.mlp.down_proj`) to
    (k, n, 2*K, codebook_id, has_bias, legacy_signs)."""

    def __init__(self, declared: dict[str, Any] | None = None, model_path: str | None = None):
        super().__init__()
        self.declared = {k: v for k, v in (declared or {}).items() if k not in ("hf_config", "packed_modules_mapping")}
        self.model_path = model_path
        self.modules: dict[str, tuple] = {}
        self.warnings: list[str] = []
        if model_path:
            self._scan(model_path)

    # ---- QuantizationConfig contract
    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        hf_config = config.get("hf_config")
        # A draft checkpoint (EXL3 DFlash draft) carries its own manifest: its local directory wins over the env var,
        # which names the target checkpoint.
        hf_path = getattr(hf_config, "_name_or_path", None)
        path = hf_path if hf_path and os.path.isdir(hf_path) else (os.environ.get(_MODEL_PATH_ENV) or hf_path)
        if path and not os.path.isdir(path):
            # HF repo id: resolve the local snapshot without downloading
            try:
                from huggingface_hub import snapshot_download
                path = snapshot_download(path, local_files_only=True)
            except Exception as e:  # pragma: no cover
                raise ValueError(f"EXL3: cannot resolve a local checkpoint directory for {path!r} ({e}); "
                                 f"set {_MODEL_PATH_ENV}") from e
        if not path:
            raise ValueError(f"EXL3: model path unknown (set {_MODEL_PATH_ENV})")
        cfg = cls(config, path)
        if hf_config is not None:
            cfg.packed_modules_mapping = dict(config.get("packed_modules_mapping") or {})
        return cfg

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == "exl3" and user_quant in (None, "exl3"):
            return "exl3"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []

    def can_fuse_shared_expert(self) -> bool:      # shared experts stay ordinary EXL3 linears
        return False

    # ---- checkpoint scan
    def _scan(self, model_path: str) -> None:
        man = load_manifest(model_path)
        self.modules = {k: (m.k, m.n, m.bits.twice, m.codebook.value, m.has_bias, m.legacy_signs)
                        for k, m in man.matrices.items()}
        self.warnings = list(man.warnings)
        for w in man.warnings:
            logger.warning("EXL3 checkpoint: %s", w)
        s = man.summary()
        logger.info("EXL3 checkpoint %s: %d quantized linears %s, %d other tensors", model_path,
                    s["quantized_linears"], s["by_bits_codebook"], s["unquantized_tensors"])

    def __getstate__(self):      # pickled to workers: plain data only
        return self.__dict__.copy()

    # ---- prefix -> checkpoint keys
    _ALIASES = (("model.language_model.", "model."), (".self_attn.", "."))

    def lookup(self, key: str):
        """Checkpoint-key lookup tolerant of SGLang's prefix spellings (`model.` for `model.language_model.`,
        `.self_attn.` dropped or kept)."""
        if key in self.modules:
            return self.modules[key]
        cands = {key}
        for a, b in self._ALIASES:
            for c in list(cands):
                cands.add(c.replace(a, b)); cands.add(c.replace(b, a))
        # the draft (MTP) model is built with prefix "mtp"; its layers are "mtp.layers.0.*"
        for c in list(cands):
            if c.startswith("model.layers.") and ("mtp." + c[len("model."):]) in self.modules:
                cands.add("mtp." + c[len("model."):])
        for c in cands:
            if c in self.modules:
                return self.modules[c]
        return None

    def _sources(self, prefix: str) -> list[str]:
        parent, _, leaf = prefix.rpartition(".")
        packed = (self.packed_modules_mapping or _DEFAULT_PACKED).get(leaf)
        return [f"{parent}.{s}" if parent else s for s in packed] if packed else [prefix]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        from .linear import Exl3LinearMethod
        try:
            from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        except Exception:  # pragma: no cover
            FusedMoE = ()
        if FusedMoE and isinstance(layer, FusedMoE):
            from .moe import Exl3MoEMethod
            experts = {k: v for k, v in self.modules.items() if _expert_of(k, prefix)}
            return Exl3MoEMethod(self, prefix, experts) if experts else None
        if isinstance(layer, ParallelLMHead):
            info = self.lookup(prefix)
            return Exl3LinearMethod(prefix, [info]) if info else None
        if not isinstance(layer, LinearBase):
            return None
        infos = [self.lookup(p) for p in self._sources(prefix)]
        if not any(infos):
            return _padded_unquantized()
        if not all(infos):
            raise ValueError(f"{prefix}: fused module mixes EXL3 and unquantized sources {self._sources(prefix)}")
        return Exl3LinearMethod(prefix, infos)


_DEFAULT_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def _expert_of(key: str, prefix: str) -> bool:
    for a, b in Exl3Config._ALIASES:
        prefix = prefix.replace(a, b)
        key = key.replace(a, b)
    return key.startswith(prefix + ".") and ".experts." in key


_PAD_CLS = None


def _padded_unquantized():
    """Unquantized linears left in bf16 by EXL3 checkpoints (GDN in_proj_a/b = in_proj_ba, 5120 -> 96 on the 27B).
    On sm_86 cuBLAS picks a one-thread-block CUTLASS kernel for 2-7 rows at such shapes (26-60 us instead of ~5 us, i.e.
    ~2.8 ms per MTP round on the 27B). Padding the row count to 8 selects the good kernel; rows are independent, so the
    result rows are identical to the unpadded GEMM's. SGLANG_EXL3_PAD_SMALL_M=0 disables it."""
    global _PAD_CLS
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    if os.environ.get("SGLANG_EXL3_PAD_SMALL_M", "1") != "1":
        return UnquantizedLinearMethod()
    if _PAD_CLS is None:
        class Exl3PaddedUnquantizedLinearMethod(UnquantizedLinearMethod):
            def apply(self, layer, x, bias=None):
                m = x.shape[0] if x.dim() == 2 else -1
                if 1 < m < 8 and layer.weight.shape[0] <= 1024:
                    xp = torch.nn.functional.pad(x, (0, 0, 0, 8 - m))
                    return super().apply(layer, xp, bias)[:m]
                return super().apply(layer, x, bias)
        _PAD_CLS = Exl3PaddedUnquantizedLinearMethod
    return _PAD_CLS()
