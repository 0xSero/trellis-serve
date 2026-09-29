"""Vision (image input) for Qwen3.8-Flash-Next EXL3 through SGLang on Intel XPU.

What stock SGLang v0.5.20 does with this checkpoint (qwen4_exp -> Qwen3VLForConditionalGeneration):
  * builds the ViT with quant_config=None, so every EXL3 vision linear (q/k/v/proj, linear_fc1/fc2, merger) is
    "not found" at load and the bf16 parameters stay uninitialised (garbage image embeddings);
  * loads the checkpoint's stray bf16 `attn.qkv.weight/bias` (exllamav3 ignores them: it loads the EXL3 q/k/v first);
  * sizes the MLP at intermediate_size 4304, while exllamav3 quantized it padded to 4352 (fc1 bias is stored padded);
  * hashes the n-gram (PLE) table with SGLang's multimodal pad values at image positions, where exllamav3 / HF hash
    the literal placeholder token (image_token_id).
install() fixes all of it without SGLang source edits:
  1. the ViT is built with the EXL3 quant config (Exl3XpuLinearMethod: had_in + trellis reconstruct + oneDNN GEMM +
     had_out, the same XPU kernels as the text linears) and intermediate_size padded to the checkpoint's fc1 width;
  2. q_proj/k_proj/v_proj EXL3 tensors (and their fp16 biases) are routed into the fused qkv_proj with shard ids;
     the unused bf16 qkv tensors are dropped;
  3. multimodal pad values are mapped back to image/video token ids for the PLE hash (in the outer forward: SGLang's
     mm routine clamps the pads to vocab-1 in place before the language model runs);
  4. an XPU vision attention backend "exl3_sdpa": per image (cu_seqlens segment) torch SDPA (fused on XPU), no s x s
     mask (SGLang's default xpu_attn needs sgl_kernel flash_attn_varlen_func; the ViT's head_dim is 72);
  5. a row-chunked ViT block (bounded peak memory up to 4096^2 images) and an empty torch cache after each ViT call.
Env: EXL3_VISION=0 disables all of it; EXL3_VIT_ATTN (default exl3_sdpa; xpu_attn / sdpa / triton_attn = SGLang's)
replaces SGLang's XPU default; EXL3_VIT_ATTN_ROWS (default 8192) query rows per SDPA call;
EXL3_VIT_ROWS (default 8192) rows per chunk of the ViT block; EXL3_VIT_LOG=1 logs every ViT call (ms, peak memory).
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
_LOGN = [0]
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
        # torch's XPU SDPA is fused (no s x s scores: peak = the output chunk, measured), so the query chunk only
        # bounds the output transient; < 512 rows per call halves the throughput (tests: sdpa_bench, 12:40 CEST)
        self.rows = int(os.environ.get("EXL3_VIT_ATTN_ROWS", "8192"))

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
            qc = max(512, self.rows)
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

    rows = int(os.environ.get("EXL3_VIT_ROWS", "8192"))
    if rows > 0:
        # Row-chunked ViT block (same math as Qwen3_VisionBlock + VisionAttention, per-row ops chunked): the peak is
        # x + q/k/v [S, H, D] + one row chunk of transients instead of ~3 full-width copies of qkv and fc1 per call
        # (a 4096^2 image is 65,536 patches; fc1 alone is 0.57 GB per bf16 copy at that size).
        from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb_native_eager as rope
        B = q3.Qwen3_VisionBlock
        b_fwd = B.forward

        def block_forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, *a, **k):
            attn = self.attn
            if not hasattr(attn.qkv_proj, "exl3_trellis") or not attn.use_qkv_parallel or x.shape[1] != 1:
                return b_fwd(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, *a, **k)
            S = x.shape[0]
            x2 = x.view(S, -1)
            H, Dh = attn.num_attention_heads_per_partition, attn.head_size
            cos, sin = rotary_pos_emb_cos, rotary_pos_emb_sin
            if cos.size(-1) * 2 == Dh:
                cos, sin = torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)
            q = torch.empty(S, H, Dh, dtype=x.dtype, device=x.device)
            kk = torch.empty_like(q)
            vv = torch.empty_like(q)
            for r0 in range(0, S, rows):
                r1 = min(S, r0 + rows)
                qkv, _ = attn.qkv_proj(self.norm1(x2[r0:r1]))
                qc, kc, vc = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
                qc, kc = rope(qc.reshape(-1, H, Dh), kc.reshape(-1, H, Dh), cos[r0:r1], sin[r0:r1])
                q[r0:r1], kk[r0:r1], vv[r0:r1] = qc, kc, vc.reshape(-1, H, Dh)
                del qkv, qc, kc, vc
            o = attn.qkv_backend.forward(q=q, k=kk, v=vv, bsz=1, seq_len=S, cu_seqlens=cu_seqlens,
                                         softmax_scale=attn.softmax_scale)
            del q, kk, vv
            for r0 in range(0, S, rows):
                r1 = min(S, r0 + rows)
                out, _ = attn.proj(o[r0:r1].reshape(r1 - r0, H * Dh))
                x2[r0:r1] += out
            del o
            for r0 in range(0, S, rows):
                r1 = min(S, r0 + rows)
                x2[r0:r1] += self.mlp(self.norm2(x2[r0:r1]))
            return x

        B.forward = block_forward

    if os.environ.get("EXL3_VIT_EMPTY_CACHE", "1") == "1":
        # the ViT's transients (activations of up to 65,536 patches) must not stay in torch's cache: the Level Zero
        # driver needs that headroom for its own allocations (X004: OUT_OF_RESOURCES when torch's cache grows into it)
        v_fwd0 = V.forward

        def vis_forward_ec(self, *a, **k):
            out = v_fwd0(self, *a, **k)
            torch.xpu.empty_cache()
            return out

        V.forward = vis_forward_ec

    if os.environ.get("EXL3_VIT_LOG", "0") == "1":
        # per ViT call: grid, patches, wall ms (host-synchronised) and peak torch allocation above the entry level
        import time
        v_fwd = V.forward

        def vis_forward(self, x, grid_thw, *a, **k):
            torch.xpu.synchronize()
            base = torch.xpu.memory_allocated()
            torch.xpu.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = v_fwd(self, x, grid_thw, *a, **k)
            torch.xpu.synchronize()
            g = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
            logger.info("exl3xpu vit: grid %s patches %d -> %s, %.1f ms, peak +%.3f GiB above %.3f GiB, device free "
                        "%.2f GiB", g, x.shape[0], tuple(out.shape), (time.perf_counter() - t0) * 1e3,
                        (torch.xpu.max_memory_allocated() - base) / 2 ** 30, base / 2 ** 30,
                        torch.xpu.mem_get_info()[0] / 2 ** 30)
            return out

        V.forward = vis_forward

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

    # SGLang's mm routine clamps the pad values in input_ids to vocab-1 IN PLACE and clears forward_batch.mm_inputs
    # before the language model runs, so the remap happens in the outer forward (pads still intact) and the language
    # model picks the remapped copy up for the PLE hash.
    m_fwd = M.forward

    def mm_forward(self, input_ids, positions, forward_batch, *a, **k):
        mm = getattr(forward_batch, "mm_inputs", None)
        if mm and any(x is not None for x in mm) and input_ids is not None:
            pairs = []
            for x in mm:
                for it in (getattr(x, "mm_items", None) or []) if x is not None else []:
                    mod = str(getattr(it, "modality", "")).upper()
                    tid = _TOK.get("video" if "VIDEO" in mod else "image")
                    if tid is not None and getattr(it, "pad_value", None) is not None:
                        pairs.append((int(it.pad_value), int(tid)))
            if pairs:
                ids = input_ids.clone()
                for pv, tid in pairs:
                    ids.masked_fill_(input_ids == pv, tid)
                forward_batch._exl3_ple_ids = ids
                if _LOGN[0] < 3:
                    _LOGN[0] += 1
                    logger.info("exl3xpu vision: PLE ids: %d placeholder positions -> image/video token ids (%s)",
                                int((ids != input_ids).sum()), pairs)
        return m_fwd(self, input_ids, positions, forward_batch, *a, **k)

    M.forward = mm_forward

    L = q4.Qwen4ExpVLModel
    l_fwd = L.forward

    def vl_forward(self, input_ids, positions, forward_batch, input_embeds=None, *a, **k):
        ids = getattr(forward_batch, "_exl3_ple_ids", None)
        if ids is not None:
            forward_batch._exl3_ple_ids = None
            input_ids = ids
        return l_fwd(self, input_ids, positions, forward_batch, input_embeds, *a, **k)

    L.forward = vl_forward
    logger.info("exl3xpu vision: ViT EXL3 loader, PLE placeholder remap and exl3_sdpa backend installed")
