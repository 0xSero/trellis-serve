"""Low-bit (2-8 bit, default 4) KV cache for the Qwen4-Exp QSA attention layers on sm_86.

`SGLANG_EXL3_KV_BITS=4` (or 3): the full-attention KV pool stores K and V with exllamav3's cache quantization
(`quant_cache_cont`: groups of 32 along head_dim, H32-rotated, per-group fp16 absmax scale, `bits` per value; ~0.56x /
0.44x of fp8 at 4 / 3 bits for head_dim 256) instead of fp8. The freed VRAM goes to the expert cache (lower
`SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB`).

How it plugs into SGLang 0.5.20:
  * `QSATokenToKVPool` gets `Exl3QKVPool` as its full-attention pool class (MHATokenToKVPool subclass: packed int32 +
    fp16 scale buffers; `set_kv_buffer` quantizes the new rows with `quant_cache_cont` and scatters them).
  * `get_key_buffer` / `get_value_buffer` return a `QKVView`: logical shape (slots, heads, head_dim); `.index_select`
    dequantizes the selected rows (used by the chunked-prefill path with a cached prefix).
  * decode: `qwen_sparse_kv_extraction_compact_triton` is replaced for views: a Triton kernel gathers the selected
    rows' packed words + scales into a packed scratch (same compact layout as SGLang's kernel), then
    `dequant_cache_cont` expands the scratch into the bf16 scratch the attention kernel reads.
  * the KV pool sizing (bytes per token) is corrected so `--max-total-tokens` still fits.
All paths are CUDA-graph safe (fixed shapes, no host syncs).
"""
from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
BITS = int(os.environ.get("SGLANG_EXL3_KV_BITS", "0") or 0)


def _ext():
    from exllamav3.ext import exllamav3_ext as ext
    return ext


class QKVView:
    """Logical (slots, heads, head_dim) view of one layer's packed K or V buffer."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, head_dim: int, out_dtype=torch.bfloat16):
        self.packed, self.scales, self.head_dim, self.out_dtype = packed, scales, head_dim, out_dtype
        self.shape = torch.Size((packed.shape[0], packed.shape[1], head_dim))
        self.device = packed.device
        self.dtype = out_dtype

    def dequant(self, packed, scales):
        out = torch.empty((*packed.shape[:-1], self.head_dim), dtype=torch.float16, device=packed.device)
        if packed.numel():
            _ext().dequant_cache_cont(packed.contiguous(), scales.contiguous(), out, 0.0)
        return out.to(self.out_dtype)

    def index_select(self, dim, index):
        assert dim == 0
        return self.dequant(self.packed.index_select(0, index), self.scales.index_select(0, index))

    def size(self, d=None):
        return self.shape if d is None else self.shape[d]


def _make_pool_class():
    from sglang.srt.mem_cache import memory_pool as mp
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    from contextlib import nullcontext

    class Exl3QKVPool(mp.MHATokenToKVPool):
        qkv_bits = BITS

        def _create_buffers(self):
            bits = self.qkv_bits
            if self.head_dim % 32 or self.v_head_dim % 32:
                raise ValueError("Exl3QKVPool needs head dims divisible by 32")
            m = self.size + self.page_size
            wk, gk = self.head_dim // 32 * bits, self.head_dim // 32
            wv, gv = self.v_head_dim // 32 * bits, self.v_head_dim // 32
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.enable_custom_mem_pool else nullcontext()):
                    self.k_buffer = [torch.zeros((m, self.head_num, wk), dtype=torch.int32, device=self.device)
                                     for _ in range(self.layer_num)]
                    self.v_buffer = [torch.zeros((m, self.head_num, wv), dtype=torch.int32, device=self.device)
                                     for _ in range(self.layer_num)]
                    # counted by MHATokenToKVPool.get_kv_size_bytes
                    self.k_scale_buffer = [torch.zeros((m, self.head_num, gk), dtype=torch.float16, device=self.device)
                                           for _ in range(self.layer_num)]
                    self.v_scale_buffer = [torch.zeros((m, self.head_num, gv), dtype=torch.float16, device=self.device)
                                           for _ in range(self.layer_num)]
            self.dq_k_buffer = None
            self.dq_v_buffer = None
            logger.info("sglang-exl3: QSA KV pool %d-bit (exllamav3 cache quant): %d slots x %d layers, %.2f GB",
                        bits, m, self.layer_num, sum(t.numel() * t.element_size() for t in
                        self.k_buffer + self.v_buffer + self.k_scale_buffer + self.v_scale_buffer) / 1e9)

        def _get_key_buffer(self, layer_id):
            l = layer_id - self.start_layer
            return QKVView(self.k_buffer[l], self.k_scale_buffer[l], self.head_dim)

        def _get_value_buffer(self, layer_id):
            l = layer_id - self.start_layer
            return QKVView(self.v_buffer[l], self.v_scale_buffer[l], self.v_head_dim)

        def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
                          layer_id_override=None, dcp_kv_mask=None):
            if dcp_kv_mask is not None:
                raise NotImplementedError("Exl3QKVPool: dcp_kv_mask")
            loc, _, _ = mp.unwrap_write_loc(loc_info)
            layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
            l = layer_id - self.start_layer
            ext = _ext()
            for src, buf, sb, d in ((cache_k, self.k_buffer[l], self.k_scale_buffer[l], self.head_dim),
                                    (cache_v, self.v_buffer[l], self.v_scale_buffer[l], self.v_head_dim)):
                x = src.reshape(-1, self.head_num, d)
                if x.shape[0] == 0:
                    continue
                x16 = x.to(torch.float16).contiguous()
                q = torch.empty((x.shape[0], self.head_num, buf.shape[-1]), dtype=torch.int32, device=x.device)
                s = torch.empty((x.shape[0], self.head_num, sb.shape[-1]), dtype=torch.float16, device=x.device)
                ext.quant_cache_cont(x16, q, s, 0.0)
                idx = loc.long()
                buf[idx] = q
                sb[idx] = s

    return Exl3QKVPool


@triton.jit
def _compact_qkv(kp, ks, vp, vs, req_to_token, req_indices, indices, seq_lens, cu_k, okp, oks, ovp, ovs,
                 topk: tl.constexpr, heads: tl.constexpr, W: tl.constexpr, G: tl.constexpr, req_stride: tl.constexpr,
                 idx_stride: tl.constexpr, pad_cols, BLOCK_TOPK: tl.constexpr, BW: tl.constexpr, BG: tl.constexpr,
                 ZERO_FILL: tl.constexpr):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(req_to_token + req * req_stride + tl.where(valid, positions, 0), mask=valid, other=0).to(tl.int64)
    if ZERO_FILL:
        smask_row = cols < pad_cols
    else:
        smask_row = valid
    dst_row = (pack_start + cols).to(tl.int64)
    w = tl.arange(0, BW)
    g = tl.arange(0, BG)
    lw = valid[:, None] & (w[None, :] < W)
    sw_ = smask_row[:, None] & (w[None, :] < W)
    lg = valid[:, None] & (g[None, :] < G)
    sg = smask_row[:, None] & (g[None, :] < G)
    src_w = slots[:, None] * heads * W + head * W + w[None, :]
    dst_w = dst_row[:, None] * heads * W + head * W + w[None, :]
    src_g = slots[:, None] * heads * G + head * G + g[None, :]
    dst_g = dst_row[:, None] * heads * G + head * G + g[None, :]
    tl.store(okp + dst_w, tl.load(kp + src_w, mask=lw, other=0), mask=sw_)
    tl.store(ovp + dst_w, tl.load(vp + src_w, mask=lw, other=0), mask=sw_)
    tl.store(oks + dst_g, tl.load(ks + src_g, mask=lg, other=0.0), mask=sg)
    tl.store(ovs + dst_g, tl.load(vs + src_g, mask=lg, other=0.0), mask=sg)


def extract_qkv(k: QKVView, v: QKVView, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
                zero_fill_cols: int = 0):
    """Packed counterpart of SGLang's qwen_sparse_kv_extraction_compact_triton (same compact/strided layout)."""
    cap, heads, D = out_k.shape[0], k.packed.shape[1], k.head_dim
    W, G = k.packed.shape[-1], k.scales.shape[-1]
    dev = out_k.device
    okp = torch.zeros((cap, heads, W), dtype=torch.int32, device=dev)
    oks = torch.zeros((cap, heads, G), dtype=torch.float16, device=dev)
    ovp = torch.zeros_like(okp)
    ovs = torch.zeros_like(oks)
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    block_topk = 16
    _compact_qkv[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k.packed, k.scales, v.packed, v.scales, req_to_token, req_indices, indices, seq_lens, cu_k,
        okp, oks, ovp, ovs, topk, heads, W, G, req_to_token.stride(0), indices.stride(0), num_cols,
        BLOCK_TOPK=block_topk, BW=triton.next_power_of_2(W), BG=triton.next_power_of_2(G), ZERO_FILL=zero_fill,
        num_warps=4)
    ext = _ext()
    k16 = torch.empty((cap, heads, D), dtype=torch.float16, device=dev)
    v16 = torch.empty((cap, heads, D), dtype=torch.float16, device=dev)
    ext.dequant_cache_cont(okp, oks, k16, 0.0)
    ext.dequant_cache_cont(ovp, ovs, v16, 0.0)
    out_k.copy_(k16)
    out_v.copy_(v16)


def _patch_cell_size(bits):
    """Bytes per token for the KV pool sizing (DefaultPoolConfigurator, num_layers = full-attention layers):
    fp8 K+V -> packed words + fp16 group scales."""
    from sglang.srt.model_executor import pool_configurator as pc
    cls = pc.DefaultPoolConfigurator
    if getattr(cls, "_exl3_qkv", False):
        return
    orig = cls._compute_cell_size

    def _compute_cell_size(self, kvc, num_layers):
        cell = orig(self, kvc, num_layers)
        try:
            from sglang.srt.runtime_context import get_parallel
            mc = kvc.model_config
            n = mc.get_num_kv_heads(get_parallel().attn_tp_size, get_parallel().attn_dcp_size)
            dk, dv = mc.head_dim, mc.v_head_dim
            fp8 = n * (dk + dv) * num_layers * torch._utils._element_size(kvc.kv_cache_dtype)
            q = n * num_layers * ((dk + dv) // 32 * (bits * 4 + 2))
            if 0 < fp8 <= cell:
                logger.info("sglang-exl3: KV cell size %d -> %d B/token (%d-bit QSA KV)", cell, cell - fp8 + q, bits)
                return cell - fp8 + q
        except Exception as e:  # pragma: no cover
            logger.warning("sglang-exl3: KV cell-size correction skipped (%s)", e)
        return cell
    cls._compute_cell_size, cls._exl3_qkv = _compute_cell_size, True


def install() -> None:
    if not BITS:
        return
    if not 2 <= BITS <= 8:
        raise ValueError(f"SGLANG_EXL3_KV_BITS={BITS}: 2..8")
    from sglang.srt.mem_cache import qsa_kv_pool as qp
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb
    cls = qp.QSATokenToKVPool
    if getattr(cls, "_exl3_qkv", False):
        return
    pool_cls = _make_pool_class()
    orig_init = cls.__init__

    def __init__(self, *a, **k):
        k["full_kv_pool_class"] = pool_cls
        k["quant_method"] = None
        return orig_init(self, *a, **k)
    cls.__init__, cls._exl3_qkv = __init__, True

    prev = qb.qwen_sparse_kv_extraction_compact_triton     # (may already be the e4m3 wrapper)

    def extract(k, v, *a, **kw):
        if isinstance(k, QKVView):
            return extract_qkv(k, v, *a, **kw)
        return prev(k, v, *a, **kw)
    qb.qwen_sparse_kv_extraction_compact_triton = extract
    _patch_cell_size(BITS)
    logger.info("sglang-exl3: %d-bit QSA KV cache installed", BITS)
