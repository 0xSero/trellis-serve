"""Vision (image input) for Qwen3.8-Flash-Next EXL3 through SGLang on Intel XPU.

What stock SGLang v0.5.20 does with this checkpoint (qwen4_exp -> Qwen3VLForConditionalGeneration):
  * builds the ViT with quant_config=None, so every EXL3 vision linear (q/k/v/proj, linear_fc1/fc2, merger) is
    "not found" at load and the bf16 parameters stay uninitialised (garbage image embeddings);
  * loads the checkpoint's stray bf16 `attn.qkv.weight/bias` (exllamav3 ignores them: it loads the EXL3 q/k/v first);
  * sizes the MLP at intermediate_size 4304, while exllamav3 quantized it padded to 4352 (fc1 bias is stored padded);
  * hashes the n-gram (PLE) table with SGLang's multimodal pad values at image positions, where exllamav3 / HF hash
    the literal placeholder token (image_token_id).
install() fixes all four without SGLang source edits:
  1. the ViT is built with the EXL3 quant config (Exl3XpuLinearMethod: had_in + trellis reconstruct + oneDNN GEMM +
     had_out, the same XPU kernels as the text linears) and intermediate_size padded to the checkpoint's fc1 width;
  2. q_proj/k_proj/v_proj EXL3 tensors (and their fp16 biases) are routed into the fused qkv_proj with shard ids;
     the unused bf16 qkv tensors are dropped;
  3. Qwen4ExpVLModel.forward maps multimodal pad values back to image/video token ids before the PLE hash;
  4. an XPU vision attention backend "exl3_sdpa": per image (cu_seqlens segment) torch SDPA, no s x s mask, queries
     chunked to bound the transient (SGLang's default xpu_attn needs sgl_kernel flash_attn_varlen_func; the ViT's
     head_dim is 72).
Env: EXL3_VISION=0 disables all of it; EXL3_VIT_ATTN (default exl3_sdpa; xpu_attn / sdpa / triton_attn = SGLang's)
replaces SGLang's XPU default; EXL3_VIT_ATTN_MB (default 512) bounds the attention score transient.
"""
from __future__ import annotations

import copy
import logging
import os
import re

import torch

logger = logging.getLogger(__name__)
_PENDING_QC: list = []
_TOK: dict = {}
_QKV_RE = re.compile(r"visual\.blocks\.(\d+)\.attn\.(q|k|v)_proj\.(\w+)$")
_FUSED_QKV_RE = re.compile(r"visual\.blocks\.\d+\.attn\.qkv\.(weight|bias)$")


def _vision_width(qc) -> int | None:
    info = getattr(qc, "modules", {}).get("visual.blocks.0.mlp.linear_fc1")
    return info[1] if info else None


class VisionChunkedSdpa(torch.nn.Module):
    """SGLang vision qkv backend: q/k/v [T, H, D] (T = all images' patches), cu_seqlens -> out [T, H, D]."""

    def __init__(self, head_dim: int, num_heads: int, num_kv_heads: int, softmax_scale=None, **kwargs):
        super().__init__()
        self.head_dim, self.num_heads, self.num_kv_heads = head_dim, num_heads, num_kv_heads
        self.scale = softmax_scale
        self.budget = int(os.environ.get("EXL3_VIT_ATTN_MB", "512")) << 20

    def forward(self, q, k, v, cu_seqlens=None, bsz: int = 1, seq_len: int | None = None, softmax_scale=None,
                **kwargs):
        T = q.shape[0]
        if cu_seqlens is None:
            bounds = [0, T]
        elif isinstance(cu_seqlens, torch.Tensor):
            bounds = cu_seqlens.tolist()
        elif isinstance(cu_seqlens, (list, tuple)) and cu_seqlens and isinstance(cu_seqlens[0], torch.Tensor):
            bounds = cu_seqlens[0].tolist()
        else:
            bounds = list(getattr(cu_seqlens, "get_data", lambda: cu_seqlens)())
        scale = softmax_scale if softmax_scale is not None else self.scale
        out = torch.empty_like(q)
        H = q.shape[1]
        for s0, s1 in zip(bounds[:-1], bounds[1:]):
            s0, s1 = int(s0), int(s1)
            L = s1 - s0
            if L <= 0:
                continue
            kk = k[s0:s1].transpose(0, 1).unsqueeze(0)          # [1, H, L, D]
            vv = v[s0:s1].transpose(0, 1).unsqueeze(0)
            qc = max(64, self.budget // max(1, H * L * 4))       # fp32 score rows that fit the budget
            for c0 in range(s0, s1, qc):
                c1 = min(s1, c0 + qc)
                qq = q[c0:c1].transpose(0, 1).unsqueeze(0)       # [1, H, c, D]
                o = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=False, scale=scale)
                out[c0:c1] = o[0].transpose(0, 1)
        return out


def install() -> None:
    if os.environ.get("EXL3_VISION", "1") != "1":
        return
    import sglang.srt.models.qwen3_vl as q3
    import sglang.srt.models.qwen4_exp as q4
    import sglang.srt.layers.attention.vision as va

    va.QKV_BACKEND_IMPL.setdefault("exl3_sdpa", VisionChunkedSdpa)
    VA = va.VisionAttention
    if not getattr(VA, "_exl3_patched", False):
        det = VA._determine_attention_backend

        def determine(self, passed_backend):
            b = det(self, passed_backend)
            want = os.environ.get("EXL3_VIT_ATTN", "exl3_sdpa")
            return want if b == "xpu_attn" and want in va.QKV_BACKEND_IMPL else b

        VA._determine_attention_backend, VA._exl3_patched = determine, True

    V = q3.Qwen3VLMoeVisionModel
    if getattr(V, "_exl3_patched", False):
        return
    v_init = V.__init__

    def vis_init(self, vision_config, norm_eps: float = 1e-6, quant_config=None, prefix: str = "",
                 use_data_parallel: bool = False, *a, **k):
        qc = _PENDING_QC[-1] if _PENDING_QC else None
        width = _vision_width(qc) if qc is not None else None
        if quant_config is None and width:
            if width != vision_config.intermediate_size:
                vision_config = copy.copy(vision_config)
                logger.info("exl3xpu vision: intermediate_size %d -> %d (EXL3 padded fc1)",
                            vision_config.intermediate_size, width)
                vision_config.intermediate_size = width
            quant_config = qc
            logger.info("exl3xpu vision: ViT linears on EXL3 (XPU kernels)")
        v_init(self, vision_config, norm_eps, quant_config, prefix, use_data_parallel, *a, **k)

    V.__init__, V._exl3_patched = vis_init, True

    M = q4.Qwen4ExpForConditionalGeneration
    m_init, m_load = M.__init__, M.load_weights

    def model_init(self, config, quant_config=None, *a, **k):
        _PENDING_QC.append(quant_config if hasattr(quant_config, "modules") else None)
        _TOK.update(image=getattr(config, "image_token_id", None), video=getattr(config, "video_token_id", None))
        try:
            m_init(self, config, quant_config, *a, **k)
        finally:
            _PENDING_QC.pop()

    def load_weights(self, weights):
        vis = getattr(self, "visual", None)
        exl3 = vis is not None and hasattr(vis.blocks[0].attn.qkv_proj, "exl3_shards")
        if not exl3:
            return m_load(self, weights)
        params = dict(self.named_parameters())
        n_routed = [0, 0]

        def filt():
            for name, w in weights:
                if "visual" in name:
                    m = _QKV_RE.search(name)
                    if m:
                        p = params[f"visual.blocks.{m.group(1)}.attn.qkv_proj.{m.group(3)}"]
                        p.weight_loader(p, w, m.group(2))
                        n_routed[0] += 1
                        continue
                    if _FUSED_QKV_RE.search(name):
                        n_routed[1] += 1
                        continue
                yield name, w

        out = m_load(self, filt())
        logger.info("exl3xpu vision: %d q/k/v tensors routed into qkv_proj, %d unused bf16 qkv tensors dropped",
                    *n_routed)
        return out

    M.__init__, M.load_weights = model_init, load_weights

    L = q4.Qwen4ExpVLModel
    l_fwd = L.forward

    def vl_forward(self, input_ids, positions, forward_batch, input_embeds=None, *a, **k):
        mm = getattr(forward_batch, "mm_inputs", None)
        if mm and any(x is not None for x in mm):
            ids = input_ids if input_ids is not None else forward_batch.input_ids
            pairs = []
            for x in mm:
                for it in (getattr(x, "mm_items", None) or []) if x is not None else []:
                    mod = str(getattr(it, "modality", "")).upper()
                    tid = _TOK.get("video" if "VIDEO" in mod else "image")
                    if tid is not None and getattr(it, "pad_value", None) is not None:
                        pairs.append((int(it.pad_value), int(tid)))
            if pairs:
                ids = ids.clone()
                for pv, tid in pairs:
                    ids.masked_fill_(ids == pv, tid)
                input_ids = ids
        return l_fwd(self, input_ids, positions, forward_batch, input_embeds, *a, **k)

    L.forward = vl_forward
    logger.info("exl3xpu vision: ViT EXL3 loader, PLE placeholder remap and exl3_sdpa backend installed")
