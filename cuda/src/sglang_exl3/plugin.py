"""SGLang general plugin (`sglang.srt.plugins` entry point): registers the `exl3` quantization method and installs
the model-side shims EXL3 checkpoints need (MTP fc/embedding sharing, EXL3 DFlash drafts, optional host embedding).
Runs in the launcher, the engine and every scheduler subprocess."""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)
_done = False


def activate() -> None:
    global _done
    if _done:
        return
    _done = True
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    from sglang.srt.arg_groups.choices import QUANTIZATION_CHOICES, add_quantization_method_choices
    from .sglang_glue.config import Exl3Config
    QUANTIZATION_METHODS["exl3"] = Exl3Config
    if "exl3" not in QUANTIZATION_CHOICES:
        add_quantization_method_choices(["exl3"])
    _patch_mtp_fc()
    _patch_dflash_exl3_draft()
    if os.environ.get("SGLANG_EXL3_EMBED_HOST", "0") == "1":
        _patch_host_embedding()
    if os.environ.get("SGLANG_EXL3_VIT_SDPA", "0") == "1":
        from .sglang_glue import vit_attn
        vit_attn.install()
        logger.info("sglang-exl3: vision attention 'triton_attn' served by per-segment SDPA (flash/efficient)")
    if os.environ.get("SGLANG_EXL3_MM_FAST_CPU", "0") == "1":
        from .sglang_glue import vit_attn
        vit_attn.install_fast_processor_cpu()
    if int(os.environ.get("SGLANG_EXL3_VIT_MLP_CHUNK", "0")) > 0:
        from .sglang_glue import vit_attn
        vit_attn.install_mlp_chunk(int(os.environ["SGLANG_EXL3_VIT_MLP_CHUNK"]))
        logger.info("sglang-exl3: vision MLP row-chunked at %s rows", os.environ["SGLANG_EXL3_VIT_MLP_CHUNK"])
    logger.info("sglang-exl3: registered quantization method 'exl3'")


HOST_EMBEDS: list = []


def _patch_host_embedding() -> None:
    """SGLANG_EXL3_EMBED_HOST=1: keep the target's bf16 token embedding (2.5 GB on Qwen3.8-27B) in pinned host memory and
    gather rows over PCIe (SGLang's own pinned-host embedding from qwen4_exp, Triton gather, CUDA-graph safe). Decode reads
    a few 10 KB rows per step; the freed VRAM goes to the KV pool (~80k fp8 tokens on the 27B). The MTP draft shares it."""
    from sglang.srt.models import qwen3_5 as q
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    cls = q.Qwen3_5ForCausalLM
    if getattr(cls, "_exl3_host_embed", False):
        return
    orig = cls._build_embed_tokens

    def _build_embed_tokens(self, config):
        emb = orig(self, config)
        if not isinstance(emb, VocabParallelEmbedding):
            return emb
        if HOST_EMBEDS:              # the MTP draft (built second): reuse the target's host table later
            return emb
        emb.weight_scale = None
        host = Qwen4ExpPinnedHostEmbedding(emb, backend="pinned")
        HOST_EMBEDS.append(host)
        logger.info("sglang-exl3: token embedding %s kept in pinned host memory (%.2f GB)",
                    tuple(host.weight.shape), host.weight.numel() * host.weight.element_size() / 2**30)
        return host

    cls._build_embed_tokens, cls._exl3_host_embed = _build_embed_tokens, True


def _patch_mtp_fc() -> None:
    """The Qwen3.5 MTP draft builds `self.fc = nn.Linear(2h, h)`; EXL3 checkpoints quantize `mtp.fc`. Replace it
    with an EXL3 dense module after construction and finalize it after loading (nn.Linear has no quant hook)."""
    try:
        from sglang.srt.models import qwen3_5_mtp as m
    except Exception as e:  # pragma: no cover
        logger.warning("sglang-exl3: MTP shim not installed (%s)", e)
        return
    cls = m.Qwen3_5ForCausalLMMTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig_init, orig_load = cls.__init__, cls.load_weights

    def __init__(self, config, quant_config=None, prefix="", *a, **k):
        orig_init(self, config, quant_config, prefix, *a, **k)
        from .sglang_glue.config import Exl3Config
        from .sglang_glue.linear import Exl3Dense
        qc = getattr(self, "quant_config", quant_config)
        if isinstance(qc, Exl3Config) and os.environ.get("SGLANG_EXL3_DRAFT_SHARE_EMBED", "1") == "1":
            # The draft receives the target's embed_tokens via set_embed_and_head (init_lm_head); on a 24 GB card the
            # draft's own 2.5 GB bf16 copy would otherwise sit in memory when the KV pool is sized. Keep a 0-row
            # placeholder until the target's tensor is shared.
            emb = self.model.embed_tokens
            w = emb.weight
            emb.weight = torch.nn.Parameter(torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device), requires_grad=False)
            for name, val in vars(w).items():
                if name != "data" and not hasattr(emb.weight, name):
                    setattr(emb.weight, name, val)
            self._exl3_skip_embed = True
            del w
            torch.cuda.empty_cache()
        if isinstance(qc, Exl3Config):
            info = qc.lookup("mtp.fc")
            if info is not None:
                self.fc = Exl3Dense(self.fc.in_features, self.fc.out_features, info, "mtp.fc",
                                    params_dtype=self.fc.weight.dtype)
                logger.info("sglang-exl3: MTP fc replaced by an EXL3 linear (%s)", info[:4])

    def load_weights(self, weights, *a, **k):
        if getattr(self, "_exl3_skip_embed", False):
            weights = ((n, w) for n, w in weights if not n.endswith("embed_tokens.weight"))
        out = orig_load(self, weights, *a, **k)
        fc = getattr(self, "fc", None)
        if fc is not None and hasattr(fc, "process_weights_after_loading") and hasattr(fc, "exl3_shards"):
            fc.process_weights_after_loading()
        return out

    orig_set = cls.set_embed_and_head

    def set_embed_and_head(self, embed, head):
        if embed is not None and not embed.is_cuda and HOST_EMBEDS:
            # target embedding lives in host memory: share the target's pinned-embedding module with the draft
            self.model.embed_tokens = HOST_EMBEDS[0]
            embed = None
        orig_set(self, embed, head)
        if head is not None and head.dim() == 2 and head.shape[1] == 0:
            _install_hot_head(self)

    cls.__init__, cls.load_weights, cls.set_embed_and_head, cls._exl3_patched = __init__, load_weights, set_embed_and_head, True


def _patch_dflash_exl3_draft() -> None:
    """EXL3-quantized DFlash drafts (`--speculative-draft-model-quantization exl3`, e.g. an exllamav3 quant of
    z-lab/Qwen3.8-27B-DFlash2). SGLang builds the DFlash decoder layers with an empty prefix, so every draft linear
    would look up the same checkpoint key, fall back to unquantized, and the EXL3 tensors would be skipped by
    load_weights (a silently random draft). Give the layers their checkpoint prefixes, serve the quantized `fc`
    (a plain nn.Linear upstream) as an EXL3 linear, and fail loudly if any EXL3 layer was left without weights."""
    try:
        from sglang.srt.models import dflash as d
    except Exception as e:  # pragma: no cover
        logger.warning("sglang-exl3: DFlash EXL3 draft shim not installed (%s)", e)
        return
    layer_cls, model_cls = d.DFlashDecoderLayer, d.DFlashDraftModel
    if getattr(model_cls, "_exl3_patched", False):
        return
    orig_layer_init, orig_init, orig_load = layer_cls.__init__, model_cls.__init__, model_cls.load_weights

    def _is_exl3(qc):
        from .sglang_glue.config import Exl3Config
        return isinstance(qc, Exl3Config)

    def layer_init(self, config, layer_id, attention_conv=None, mlp_conv=None, quant_config=None, prefix=""):
        if not prefix and _is_exl3(quant_config):
            prefix = f"layers.{layer_id}"
        orig_layer_init(self, config, layer_id, attention_conv=attention_conv, mlp_conv=mlp_conv,
                        quant_config=quant_config, prefix=prefix)

    def __init__(self, config, quant_config=None, prefix=""):
        orig_init(self, config, quant_config=quant_config, prefix=prefix)
        if _is_exl3(quant_config) and isinstance(self.fc, torch.nn.Linear):
            info = quant_config.lookup("fc")
            if info is not None:
                from .sglang_glue.linear import Exl3Dense
                fin, fout = self.fc.in_features, self.fc.out_features
                self.fc = Exl3Dense(fin, fout, info, "fc", params_dtype=self.fc.weight.dtype)
                self.fc.in_features, self.fc.out_features = fin, fout
                logger.info("sglang-exl3: DFlash fc replaced by an EXL3 linear (%s)", info[:4])

    def load_weights(self, weights, *a, **k):
        out = orig_load(self, weights, *a, **k)
        empty = [n for n, m in self.named_modules() if hasattr(m, "exl3_shards") and not m.exl3_shards.get("trellis")]
        if empty:
            raise RuntimeError(f"sglang-exl3: EXL3 DFlash draft layers received no weights: {empty[:6]}")
        from .sglang_glue.linear import Exl3Dense
        if isinstance(self.fc, Exl3Dense):
            self.fc.process_weights_after_loading()
        return out

    layer_cls.__init__, model_cls.__init__, model_cls.load_weights = layer_init, __init__, load_weights
    model_cls._exl3_patched = True


@torch.no_grad()
def _hot_head_weight(target, hot: torch.Tensor) -> torch.Tensor:
    """Dense bf16 rows of the EXL3 target head for the `hot` token ids: identity blocks pushed through the quantized
    linear, i.e. the exact effective weights the target head applies."""
    k = target.exl3_suh_0.numel()
    w = torch.empty((hot.numel(), k), dtype=torch.bfloat16, device=hot.device)
    step = 256
    for i in range(0, k, step):
        eye = torch.zeros((min(step, k - i), k), dtype=torch.bfloat16, device=hot.device)
        eye[torch.arange(eye.shape[0]), torch.arange(i, i + eye.shape[0])] = 1
        y = target.quant_method.apply(target, eye)          # (block, vocab): rows of the effective weight
        w[:, i:i + eye.shape[0]] = y[:, hot].t()
        del y, eye
    torch.cuda.empty_cache()
    return w


@torch.no_grad()
def _install_hot_head(draft) -> None:
    """--speculative-token-map with an EXL3 target: SGLang sliced the target's zero-width lm_head placeholder, so the
    draft has no weights. Build a dense bf16 head for the hot tokens from the target's EXL3 head and serve it
    unquantized. The target keeps verifying with the full quantized head, so outputs are unchanged; only draft
    acceptance can move."""
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.spec_utils import load_token_map
    from .sglang_glue.linear import TARGET_LM_HEADS
    if not TARGET_LM_HEADS:
        raise RuntimeError("sglang-exl3: token map requested but no quantized target lm_head was registered")
    target = TARGET_LM_HEADS[-1]
    hot = load_token_map(get_spec().speculative_token_map).to(target.exl3_svh_0.device)
    w = _hot_head_weight(target, hot)
    lm = draft.lm_head
    lm.quant_method = UnquantizedEmbeddingMethod()
    lm.weight = torch.nn.Parameter(w, requires_grad=False)
    logger.info("sglang-exl3: draft hot-token head %s built from the EXL3 target head (%.0f MB)", tuple(w.shape),
                w.numel() * 2 / 2**20)
