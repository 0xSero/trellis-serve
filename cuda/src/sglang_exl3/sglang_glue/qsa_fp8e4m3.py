"""fp8 e4m3 KV cache for QSA on sm_86.

sm_86 Triton has no fp8e4nv type, so SGLang's QSA kernels that read an e4m3 pool fail to compile. Two readers matter:
  * decode: `qwen_sparse_kv_extraction_compact_triton` (gathers the selected rows into bf16 scratch) -> replaced by a
    copy of its kernel that reads the pool as uint8 and converts through a 256-entry bf16 lookup table (exact: every
    e4m3 value is representable in bf16);
  * chunked prefill with a cached prefix: `sparse_gqa_fwd_interface_triton_ck(q, cat(k_parts), cat(v_parts), ...)`
    -> the gathered fp8 parts are cast to bf16 in torch first.
Writes (set_kv_buffer) are plain torch casts and work unchanged. e5m2 pools keep the stock kernels.
"""
from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_LUT = {}


def _lut(device):
    t = _LUT.get(device)
    if t is None:
        t = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16).to(device)
        _LUT[device] = t
    return t


@triton.jit
def _compact_kv_u8(k, v, lut, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v,
                   topk: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr, req_stride: tl.constexpr,
                   idx_stride: tl.constexpr, pad_cols, BLOCK_TOPK: tl.constexpr, BLOCK_D: tl.constexpr,
                   ZERO_FILL: tl.constexpr):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(req_to_token + req * req_stride + tl.where(valid, positions, 0), mask=valid, other=0)
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols).to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    out_dtype = out_k.dtype.element_ty
    kb = tl.load(k + src, mask=load_mask, other=0).to(tl.int32)
    vb = tl.load(v + src, mask=load_mask, other=0).to(tl.int32)
    kval = tl.where(load_mask, tl.load(lut + kb), 0.0)
    vval = tl.where(load_mask, tl.load(lut + vb), 0.0)
    tl.store(out_k + dst, kval.to(out_dtype), mask=store_mask)
    tl.store(out_v + dst, vval.to(out_dtype), mask=store_mask)


def install() -> None:
    try:
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb
    except Exception as e:  # pragma: no cover
        logger.info("sglang-exl3: e4m3 QSA shim not installed (%s)", e)
        return
    if getattr(qb, "_exl3_e4m3", False):
        return
    orig_extract = qb.qwen_sparse_kv_extraction_compact_triton
    orig_ck = qb.sparse_gqa_fwd_interface_triton_ck

    def extract(k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
                zero_fill_cols: int = 0):
        if k.dtype != torch.float8_e4m3fn:
            return orig_extract(k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
                                zero_fill_cols=zero_fill_cols)
        _, heads, dim = k.shape
        block_topk = 16
        zero_fill = zero_fill_cols > 0
        num_cols = zero_fill_cols if zero_fill else topk
        _compact_kv_u8[(batch, heads, triton.cdiv(num_cols, block_topk))](
            k.view(torch.uint8), v.view(torch.uint8), _lut(k.device), req_to_token, req_indices, indices, seq_lens,
            cu_k, out_k, out_v, topk, heads, dim, req_to_token.stride(0), indices.stride(0), num_cols,
            BLOCK_TOPK=block_topk, BLOCK_D=triton.next_power_of_2(dim), ZERO_FILL=zero_fill, num_warps=8)

    def ck(q, k, v, *a, **kw):
        if k.dtype == torch.float8_e4m3fn:
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        return orig_ck(q, k, v, *a, **kw)

    qb.qwen_sparse_kv_extraction_compact_triton = extract
    qb.sparse_gqa_fwd_interface_triton_ck = ck
    qb._exl3_e4m3 = True
