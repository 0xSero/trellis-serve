"""Split-K (flash-decoding) GQA attention for QSA decode on sm_86: one query row per sequence against that sequence's
packed selected keys (SGLang's `_compact_kv` output, bf16), graph-safe (grid depends only on batch and max keys).

  q [B, Hq, D], k/v [N, Hkv, D] packed back to back per sequence, cu_seqlens_k [B+1] -> o [B, Hq, D]

Pass 1: program (b, kv head, split) runs an online softmax over its BLOCK_SPLIT keys for the G = Hq / Hkv query heads
that share the kv head (padded to 16 rows for tl.dot); pass 2 merges the splits.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_decode_split(Q, K, V, CU, PO, PM, PL, scale,
                      sq_b, sq_h, sk_n, sk_h, NSPLIT,
                      G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BLOCK_SPLIT: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    start = tl.load(CU + b)
    length = tl.load(CU + b + 1) - start
    rows = tl.arange(0, GP)
    dims = tl.arange(0, D)
    qmask = rows < G
    q = tl.load(Q + b * sq_b + (h * G + rows)[:, None] * sq_h + dims[None, :], mask=qmask[:, None], other=0.0)
    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    lo = s * BLOCK_SPLIT
    for n0 in range(0, BLOCK_SPLIT, BN):
        cols = lo + n0 + tl.arange(0, BN)
        kmask = cols < length
        off = (start + cols).to(tl.int64)[:, None] * sk_n + h * sk_h + dims[None, :]
        k = tl.load(K + off, mask=kmask[:, None], other=0.0)
        v = tl.load(V + off, mask=kmask[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(kmask[None, :], sc, float("-inf"))
        m_new = tl.maximum(m, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_safe[:, None])
        alpha = tl.exp(m - m_safe)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    base = (b * tl.num_programs(1) + h) * NSPLIT + s
    tl.store(PO + base * GP * D + rows[:, None] * D + dims[None, :], acc)
    tl.store(PM + base * GP + rows, m)
    tl.store(PL + base * GP + rows, l)


@triton.jit
def _qsa_decode_merge(PO, PM, PL, O, so_b, so_h, NSPLIT,
                      HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, SP: tl.constexpr):
    b = tl.program_id(0)
    hq = tl.program_id(1)
    h = hq // G
    r = hq % G
    sp = tl.arange(0, SP)
    smask = sp < NSPLIT
    base = (b * HKV + h) * NSPLIT + sp
    ms = tl.load(PM + base * GP + r, mask=smask, other=float("-inf"))
    ls = tl.load(PL + base * GP + r, mask=smask, other=0.0)
    mx = tl.max(ms, 0)
    mx_safe = tl.where(mx == float("-inf"), 0.0, mx)
    w = tl.where(smask, tl.exp(ms - mx_safe), 0.0)
    den = tl.sum(w * ls, 0)
    dims = tl.arange(0, D)
    po = tl.load(PO + (base * GP + r)[:, None] * D + dims[None, :], mask=smask[:, None], other=0.0)
    num = tl.sum(po * w[:, None], 0)
    out = tl.where(den > 0, num / den, 0.0)
    tl.store(O + b * so_b + hq * so_h + dims, out.to(O.dtype.element_ty))


def qsa_decode_attention(q, k, v, cu_seqlens_k, max_seqlen_k, scale, block_split=128, bn=32):
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    GP = max(16, triton.next_power_of_2(G))
    nsplit = triton.cdiv(int(max_seqlen_k), block_split)
    po = torch.empty((B, Hkv, nsplit, GP, D), dtype=torch.float32, device=q.device)
    pm = torch.empty((B, Hkv, nsplit, GP), dtype=torch.float32, device=q.device)
    pl = torch.empty((B, Hkv, nsplit, GP), dtype=torch.float32, device=q.device)
    q = q.contiguous()
    _qsa_decode_split[(B, Hkv, nsplit)](q, k, v, cu_seqlens_k, po, pm, pl, float(scale),
                                        q.stride(0), q.stride(1), k.stride(0), k.stride(1), nsplit,
                                        G=G, GP=GP, D=D, BLOCK_SPLIT=block_split, BN=bn, num_warps=4, num_stages=2)
    o = torch.empty_like(q)
    _qsa_decode_merge[(B, Hq)](po, pm, pl, o, o.stride(0), o.stride(1), nsplit,
                               HKV=Hkv, G=G, GP=GP, D=D, SP=triton.next_power_of_2(nsplit), num_warps=4)
    return o
