"""QSA (Qwen sparse attention, Qwen4-Exp / Qwen3.8-Flash-Next) decode on sm_86.

SGLang's QSA decode gathers the selected keys/values into packed bf16 scratch, then calls a varlen attention kernel:
FlashInfer's trtllm decode on sm100/sm120, else FA2 `flash_attn`, else the FA4 cute interface. The image has no FA2
wheel and the cute kernels do not build for sm_86 (pack_gqa store_O fails to lower), so decode dies at the first CUDA
graph capture. `install()` substitutes a small graph-safe torch implementation for the decode shape
(max_seqlen_q == 1: one query row per sequence, bottom-right causal = every packed key of that sequence).
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def varlen_decode_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None,
                            causal=True, **_):
    """q [B, Hq, D] (one row per sequence), k/v packed [N, Hkv, D], cu_seqlens_k [B+1] -> [B, Hq, D]."""
    if max_seqlen_q != 1:
        raise NotImplementedError("sm_86 QSA varlen fallback handles decode rows (max_seqlen_q == 1) only")
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    T = int(max_seqlen_k)
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    starts = cu_seqlens_k[:-1].to(torch.long)
    lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.long)
    j = torch.arange(T, device=q.device)
    idx = (starts[:, None] + j[None, :]).clamp_(max=k.shape[0] - 1)
    valid = j[None, :] < lens[:, None]                                     # [B, T]
    # the packed scratch past each sequence's valid keys is uninitialised (may hold NaN bit patterns): zero it, or
    # 0 * NaN poisons the output (seen as NaN hidden states -> the next QSA indexer's top-k never terminates)
    keep = valid[:, :, None, None]
    kk = k.index_select(0, idx.reshape(-1)).view(B, T, Hkv, D).float().masked_fill(~keep, 0.0)
    vv = v.index_select(0, idx.reshape(-1)).view(B, T, Hkv, D).float().masked_fill(~keep, 0.0)
    qq = q.view(B, Hkv, G, D).float()
    s = torch.einsum("bhgd,bthd->bhgt", qq, kk) * scale
    s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)                                       # rows without keys (graph padding)
    o = torch.einsum("bhgt,bthd->bhgd", p, vv)
    return o.reshape(B, Hq, D).to(q.dtype)


def install() -> None:
    try:
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb
    except Exception as e:  # pragma: no cover
        logger.info("sglang-exl3: QSA sm_86 shim not installed (%s)", e)
        return
    if getattr(qb, "_exl3_sm86", False):
        return
    orig = qb._resolve_flash_attn_varlen_func
    import functools

    @functools.lru_cache(maxsize=1)
    def resolve():
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            try:
                from flash_attn import flash_attn_varlen_func  # FA2 wheel, if present, is the better kernel
                return flash_attn_varlen_func
            except ImportError:
                logger.info("sglang-exl3: QSA decode attention on sm_86 via the torch varlen fallback")
                return varlen_decode_attention
        return orig()
    qb._resolve_flash_attn_varlen_func = resolve
    qb._exl3_sm86 = True
