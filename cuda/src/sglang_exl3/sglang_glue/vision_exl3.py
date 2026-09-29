"""EXL3 vision tower for Qwen4-Exp (Qwen3.8-Flash-Next) and other Qwen3-VL-tower checkpoints (SGLANG_EXL3_VISION=1).

SGLang builds the ViT with `quant_config=None` (plain bf16 linears), but exllamav3 checkpoints quantize its linears
(`vision_bits`, e.g. 5): `attn.{q,k,v}_proj`, `attn.proj`, `mlp.linear_fc{1,2}` of every block and the merger's
`linear_fc{1,2}` arrive as trellis/suh/svh/mul1 + fp16 bias, which SGLang's VL loader cannot place ("Parameter visual.*
not found"), leaving those layers uninitialised (images unusable). The checkpoint also carries a leftover bf16 fused
`attn.qkv.weight` that exllamav3 ignores (its loader prefers the EXL3 q/k/v).

Here the EXL3 tensors are intercepted from the weight stream, each matrix is decoded ONCE at load time with the plugin's
own EXL3 GEMM (identity rows pushed through the quantized linear = exactly the effective weights the kernels apply),
and written into SGLang's dense ViT parameters. The bf16 `qkv` leftover is dropped so the tower matches exllamav3's
(the reference). The ViT then runs on cuBLAS like any bf16 tower. The MLP's padded intermediate (4304 -> 4352 in the
EXL3 checkpoint) is kept at the padded width, as exllamav3 computes it.

VRAM: the dense tower has the same size as the (uninitialised) bf16 tower SGLang allocated before, plus the padding
(27 x 2 x 48 x 1152 x 2 B = 5.7 MiB, allocated at construction); the decode is transient (one matrix at a time).

Also (independent of the checkpoint format): SGLang clamps the multimodal pad ids in `forward_batch.input_ids` to
`vocab-1` before the language model runs; Qwen4-Exp's n-gram PLE hashes those ids, while HF/exllamav3 hash the literal
`image_token_id` at every image position. `_patch_ple_image_ids` restores that.
"""
from __future__ import annotations

import logging
import os
import re
import time

import torch

logger = logging.getLogger(__name__)
_EXL3_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1", "bias")


@torch.no_grad()
def _decode_dense(qc, key: str, tensors: dict, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(out, in) effective weight of one EXL3 matrix + its bias, via the plugin's EXL3 linear."""
    from .linear import Exl3Dense
    k, n, twice, cb, _has_bias, legacy = qc.lookup(key)
    lin = Exl3Dense(k, n, (k, n, twice, cb, False, legacy), key, params_dtype=dtype)
    for suf, t in tensors.items():
        if suf == "bias":
            continue
        getattr(lin, suf).weight_loader(getattr(lin, suf), t)
    lin.process_weights_after_loading()
    dev = torch.device("cuda", torch.cuda.current_device())
    w = torch.empty((n, k), dtype=dtype, device=dev)
    step = 1024
    for a in range(0, k, step):
        b = min(k, a + step)
        eye = torch.zeros((b - a, k), dtype=torch.float16, device=dev)
        eye[torch.arange(b - a, device=dev), torch.arange(a, b, device=dev)] = 1
        y = lin(eye)                                  # (b-a, n) = rows a..b of W^T
        w[:, a:b] = y.t().to(dtype)
        del y, eye
    bias = tensors.get("bias")
    bias = bias.to(dev, dtype) if bias is not None else None
    del lin
    return w, bias


def _set(param: torch.nn.Parameter, value: torch.Tensor) -> None:
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"sglang-exl3: vision parameter {tuple(param.shape)} vs decoded {tuple(value.shape)}")
    param.data.copy_(value.to(param.dtype))


@torch.no_grad()
def _install_visual(model, qc, vis: dict[str, dict]) -> None:
    visual = model.visual
    dtype = next(visual.parameters()).dtype
    t0 = time.time()
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info()[0]
    before = sum(p.numel() * p.element_size() for p in visual.parameters())
    done = 0
    dense = {}
    for key in sorted(vis):
        dense[key] = _decode_dense(qc, key, vis[key], dtype)
        done += 1
    pre = "model.visual."
    nblk = len(visual.blocks)
    for i, blk in enumerate(visual.blocks):
        b = f"{pre}blocks.{i}."
        q, k, v = (dense.pop(b + f"attn.{s}_proj") for s in "qkv")
        qkv = blk.attn.qkv_proj
        _set(qkv.weight, torch.cat([q[0], k[0], v[0]], 0))
        if qkv.bias is not None:
            _set(qkv.bias, torch.cat([q[1], k[1], v[1]], 0))
        for mod, name in ((blk.attn.proj, "attn.proj"), (blk.mlp.linear_fc1, "mlp.linear_fc1"),
                          (blk.mlp.linear_fc2, "mlp.linear_fc2")):
            w, bias = dense.pop(b + name)
            _set(mod.weight, w)
            if bias is not None and mod.bias is not None:
                _set(mod.bias, bias)
    mg = visual.merger
    for mod, name in ((mg.linear_fc1, "merger.linear_fc1"), (mg.linear_fc2, "merger.linear_fc2")):
        w, bias = dense.pop(pre + name)
        _set(mod.weight, w)
        if bias is not None and mod.bias is not None:
            _set(mod.bias, bias)
    if dense:
        raise RuntimeError(f"sglang-exl3: EXL3 vision matrices not placed: {sorted(dense)[:6]}")
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    after = sum(p.numel() * p.element_size() for p in visual.parameters())
    free1 = torch.cuda.mem_get_info()[0]
    logger.info("sglang-exl3: EXL3 vision tower decoded (%d matrices, %d blocks) to %s in %.1f s; visual params %.1f -> "
                "%.1f MiB (delta %+.1f MiB), device free %+.1f MiB", done, nblk, dtype, time.time() - t0,
                before / 2**20, after / 2**20, (after - before) / 2**20, (free1 - free0) / 2**20)


def _to_fp16(visual) -> None:
    """Run the ViT in fp16 like exllamav3 (SGLANG_EXL3_VISION_FP16=1): same bytes; the tower's output is cast back to
    the language model's dtype."""
    out_dtype = visual.dtype
    visual.half()
    orig = visual.forward

    def forward(*a, **k):
        y = orig(*a, **k)
        return y.to(out_dtype) if torch.is_tensor(y) else y
    visual.forward = forward
    logger.info("sglang-exl3: vision tower runs in fp16 (output cast to %s)", out_dtype)


def _dump_hook(visual, path: str) -> None:
    """Debug (SGLANG_EXL3_VISION_DUMP=<dir>): save every ViT output (image embeddings) as <dir>/vit_<n>.pt."""
    orig, n = visual.forward, [0]

    def forward(*a, **k):
        y = orig(*a, **k)
        os.makedirs(path, exist_ok=True)
        torch.save(y.detach().float().cpu(), os.path.join(path, f"vit_{n[0]}.pt")); n[0] += 1
        return y
    visual.forward = forward


def _patch_load(cls) -> None:
    if "_exl3_vision" in cls.__dict__:
        return
    orig_load, orig_init = cls.load_weights, cls.__init__

    def __init__(self, config, quant_config=None, *a, **k):
        # build the MLP at the checkpoint's padded width (4304 -> 4352) so decoded weights drop into the parameters
        # SGLang allocated (re-allocating 54 bigger fc1/fc2 tensors after load left ~0.5 GB of fragmented cache)
        from .config import Exl3Config
        vc = getattr(config, "vision_config", None)
        if isinstance(quant_config, Exl3Config) and vc is not None:
            info = quant_config.lookup("model.visual.blocks.0.mlp.linear_fc1")
            if info is not None and info[1] != vc.intermediate_size:
                logger.info("sglang-exl3: vision MLP width %d -> %d (EXL3 padded)", vc.intermediate_size, info[1])
                vc.intermediate_size = info[1]
        orig_init(self, config, quant_config, *a, **k)

    def load_weights(self, weights, *a, **k):
        from .config import Exl3Config
        qc = getattr(self, "quant_config", None)
        if not isinstance(qc, Exl3Config) or getattr(self, "visual", None) is None:
            return orig_load(self, weights, *a, **k)
        vis: dict[str, dict] = {}
        dropped = []

        def stream():
            for name, w in weights:
                if ".visual." in name or name.startswith("visual."):
                    key = name if name.startswith("model.") else "model." + name
                    mod, _, suf = key.rpartition(".")
                    if suf in _EXL3_SUFFIXES and qc.lookup(mod) is not None:
                        vis.setdefault(mod, {})[suf] = w
                        continue
                    if re.search(r"\.attn\.qkv$", mod) and qc.lookup(mod[:-3] + "q_proj") is not None:
                        dropped.append(name)          # bf16 fused leftover; exllamav3 uses the EXL3 q/k/v
                        continue
                yield name, w
        out = orig_load(self, stream(), *a, **k)
        if vis:
            logger.info("sglang-exl3: %d EXL3 vision matrices intercepted, %d leftover fused qkv tensors dropped",
                        len(vis), len(dropped))
            _install_visual(self, qc, vis)
            if os.environ.get("SGLANG_EXL3_VISION_FP16", "0") == "1":
                _to_fp16(self.visual)
            if os.environ.get("SGLANG_EXL3_VISION_DUMP"):
                _dump_hook(self.visual, os.environ["SGLANG_EXL3_VISION_DUMP"])
        return out

    cls.__init__, cls.load_weights, cls._exl3_vision = __init__, load_weights, True


def _patch_ple_image_ids() -> None:
    from sglang.srt.models import qwen4_exp as q4
    cls = q4.Qwen4ExpVLModel
    if "_exl3_ple_ids" in cls.__dict__:
        return
    orig = cls.forward

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, *a, **k):
        img = getattr(self, "_exl3_image_token_id", None)
        if input_embeds is not None and img is not None and forward_batch.contains_mm_inputs():
            ids = forward_batch.input_ids if input_ids is None else input_ids
            # multimodal pad ids (>= 1e6) were clamped in place to num_embeddings-1 by embed_mm_inputs
            ids.masked_fill_(ids >= self._exl3_pad_floor, img)
        return orig(self, input_ids, positions, forward_batch, input_embeds, *a, **k)

    cls.forward, cls._exl3_ple_ids = forward, True

    top = q4.Qwen4ExpForConditionalGeneration
    orig_init = top.__init__

    def __init__(self, config, *a, **k):
        orig_init(self, config, *a, **k)
        emb = getattr(self.model, "embed_tokens", None)
        n = getattr(emb, "num_embeddings", None) or getattr(getattr(config, "text_config", config), "vocab_size")
        self.model._exl3_image_token_id = getattr(config, "image_token_id", None)
        self.model._exl3_pad_floor = int(n) - 1

    top.__init__ = __init__


def _ln32(norm, x32):
    return torch.nn.functional.layer_norm(x32, norm.normalized_shape, norm.weight.float(),
                                          norm.bias.float() if norm.bias is not None else None, norm.eps)


def install_fp32_residual() -> None:
    """SGLANG_EXL3_VIT_FP32_RESIDUAL=1: keep the ViT residual stream (and its LayerNorms) in fp32 like exllamav3
    (patch/pos embed out_dtype float); linears and attention still run in the tower dtype. +4 B/patch x 1152 x ~2
    transients (~0.6 GB at 4096^2)."""
    from einops import rearrange
    from sglang.srt.models import qwen3_vl as m
    if getattr(m.Qwen3_VisionBlock, "_exl3_fp32", False):
        return

    def block_forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, output_ws=None,
                      forward_metadata=None, max_seqlen=None, sequence_lengths=None):
        dt = self.norm1.weight.dtype
        x32 = x if x.dtype == torch.float32 else x.float()
        h = rearrange(_ln32(self.norm1, x32).to(dt), "s b ... -> b s ...")
        attn = self.attn(h, cu_seqlens=cu_seqlens, rotary_pos_emb_cos=rotary_pos_emb_cos,
                         rotary_pos_emb_sin=rotary_pos_emb_sin, output_ws=output_ws, forward_metadata=forward_metadata,
                         max_seqlen=max_seqlen, sequence_lengths=sequence_lengths)
        x32 = x32 + rearrange(attn, "b s ... -> s b ...").float()
        del attn, h
        x32 += self.mlp(_ln32(self.norm2, x32).to(dt)).float()
        return x32

    orig_merger = m.Qwen3VLMoeVisionPatchMerger.forward

    def merger_forward(self, x):
        if x.dtype != torch.float32:
            return orig_merger(self, x)
        dt = self.norm.weight.dtype
        if self.use_postshuffle_norm:
            x = _ln32(self.norm, x.view(-1, self.hidden_size)).to(dt)
        else:
            x = _ln32(self.norm, x).to(dt).view(-1, self.hidden_size)
        y, _ = self.linear_fc1(x)
        y, _ = self.linear_fc2(self.act_fn(y))
        return y

    m.Qwen3_VisionBlock.forward, m.Qwen3_VisionBlock._exl3_fp32 = block_forward, True
    m.Qwen3VLMoeVisionPatchMerger.forward = merger_forward
    logger.info("sglang-exl3: ViT residual stream in fp32")


def install() -> None:
    try:
        from sglang.srt.models import qwen4_exp as q4
    except Exception as e:  # pragma: no cover
        logger.info("sglang-exl3: vision shim not installed (%s)", e)
        return
    _patch_load(q4.Qwen4ExpForConditionalGeneration)
    if os.environ.get("SGLANG_EXL3_VIT_FP32_RESIDUAL", "0") == "1":
        install_fp32_residual()
    if os.environ.get("SGLANG_EXL3_PLE_IMAGE_IDS", "1") == "1":
        _patch_ple_image_ids()
    logger.info("sglang-exl3: EXL3 vision tower loader installed")
