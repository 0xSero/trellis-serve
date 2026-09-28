"""Reference backend: ExLlamaV3's own CUDA kernels (`exllamav3_ext`), called directly.

This is the numerical reference: same kernels, same codebooks, same launch
selection as ExLlamaV3. Build it from the pinned, patched tree (third_party/).

Only the compiled module is imported, never the `exllamav3` Python package
(which pulls tokenizers/llguidance and edits PYTORCH_CUDA_ALLOC_CONF).
Environment switches of the extension (EXL3_INT8_GEMV, EXL3_GEMV, ...) are read
once at first use; they are set by runtime settings / the parity harness, not here.
"""
from __future__ import annotations

import importlib

import torch

from trellis_core.format.spec import BLOCK, TILE
from .base import BackendStatus

CODEBOOK_3INST, CODEBOOK_MCG, CODEBOOK_MUL1 = 0, 1, 2
_REQUIRED = ("exl3_gemm", "reconstruct", "reconstruct_slice", "had_r_128")
_ext = None


def probe() -> BackendStatus:
    try:
        ext = _load()
    except ImportError as e:
        return BackendStatus(False, detail=str(e))
    missing = tuple(f for f in _REQUIRED if not hasattr(ext, f))
    return BackendStatus(not missing, missing=missing)


def _load():
    global _ext
    if _ext is None:
        try:
            _ext = importlib.import_module("exllamav3_ext")
        except ImportError:
            _ext = importlib.import_module("exllamav3.exllamav3_ext")  # wheel layout; imports the package too
    return _ext


def _check(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> tuple[int, int]:
    # The extension has most of its shape checks commented out and reads out of bounds instead.
    k, n = trellis.shape[0] * TILE, trellis.shape[1] * TILE
    if trellis.dtype != torch.int16 or trellis.dim() != 3 or not trellis.is_contiguous():
        raise ValueError(f"trellis must be contiguous int16 (k/16, n/16, words), got {trellis.dtype} {tuple(trellis.shape)}")
    if suh.shape != (k,) or svh.shape != (n,) or suh.dtype != torch.float16 or svh.dtype != torch.float16:
        raise ValueError(f"scale vectors do not match trellis ({k}, {n}): suh {tuple(suh.shape)}, svh {tuple(svh.shape)}")
    if x.dim() != 2 or x.shape[1] != k or x.dtype != torch.float16 or not x.is_contiguous():
        raise ValueError(f"x must be contiguous fp16 (rows, {k}), got {x.dtype} {tuple(x.shape)}")
    return k, n


def bits_of(trellis: torch.Tensor) -> float:
    return trellis.shape[2] / TILE


def gemm(x, trellis, suh, svh, codebook: int, out_dtype=torch.float16,
         force_shape_idx: int = -1, force_num_sms: int = 0) -> torch.Tensor:
    """Trellis kernel path. Returns (rows, n). force_* > 0 pins launch geometry (disables autotune and GEMV)."""
    _, n = _check(x, trellis, suh, svh)
    out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
    x_had = torch.empty_like(x)
    _load().exl3_gemm(x, trellis, out, suh, x_had, svh, force_shape_idx,
                      codebook == CODEBOOK_MCG, codebook == CODEBOOK_MUL1, force_num_sms)
    return out


def reconstruct(trellis, codebook: int) -> torch.Tensor:
    """Decoded inner weight W_hat, fp16 (k, n), rotated basis. Bit-exact vs ExLlamaV3 by construction."""
    k, n = trellis.shape[0] * TILE, trellis.shape[1] * TILE
    w = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    _load().reconstruct(w, trellis, bits_of(trellis), codebook == CODEBOOK_MCG, codebook == CODEBOOK_MUL1)
    return w


import os as _os
_DENSE_SLICE_BYTES = int(_os.environ.get("SGLANG_EXL3_DENSE_SLICE_MB", "64")) << 20   # 24 GB cards: bounded transients


def dense_forward(x, trellis, suh, svh, codebook: int, max_weight_bytes: int = _DENSE_SLICE_BYTES) -> torch.Tensor:
    """Reconstruct + cuBLAS path for many rows. The decoded weight is transient and built in
    128-aligned output slices so its footprint stays bounded (lm_head would be 2.5 GB otherwise)."""
    k, n = _check(x, trellis, suh, svh)
    ext = _load()
    mcg, mul1, bits = codebook == CODEBOOK_MCG, codebook == CODEBOOK_MUL1, bits_of(trellis)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y = torch.empty((x.shape[0], n), dtype=torch.float16, device=x.device)
    step = max(BLOCK, (max_weight_bytes // (2 * k)) // BLOCK * BLOCK)
    if step >= n:
        w = torch.empty((k, n), dtype=torch.float16, device=x.device)
        ext.reconstruct(w, trellis, bits, mcg, mul1)
        torch.mm(xh, w, out=y)
    else:
        for start in range(0, n, step):
            width = min(step, n - start)
            w = torch.empty((k, width), dtype=torch.float16, device=x.device)
            ext.reconstruct_slice(w, trellis, bits, mcg, mul1, start)
            y[:, start:start + width] = xh @ w
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def dense_cached(x, w, suh, svh) -> torch.Tensor:
    """Same math as dense_forward with the decoded weight kept resident (bit-identical W, decoded once at load)."""
    ext = _load()
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y = torch.mm(xh, w)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def pointer_table(tensors: list[torch.Tensor]) -> torch.Tensor:
    """int64 device tensor of data pointers, the form exl3_mgemm takes for per-expert matrices.
    The tensors must stay alive and must never move afterwards."""
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device=tensors[0].device)


def moe_mgemm(x, topk_ids, topk_weights, gate, up, down, bits_gate_up: float, bits_down: float,
              intermediate: int, num_experts: int, codebook: int) -> torch.Tensor:
    """Routed experts with SiLU gating via three exl3_mgemm launches over pointer tables
    (calling convention after yeasah/vllm-exl3-plugin, Apache-2.0). gate / up / down are
    (trellis_ptrs, suh_ptrs, svh_ptrs). One K and one codebook per projection group.
    Correctness-first: every (token, expert) slot is one row, so long prefills are slow; M3 replaces it."""
    ext = _load()
    tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    if tokens == 0:
        return x.new_empty((0, hidden))
    mcg, mul1 = codebook == CODEBOOK_MCG, codebook == CODEBOOK_MUL1
    rows = x.to(torch.float16).repeat_interleave(top_k, dim=0).unsqueeze(1).contiguous()   # slot j = (token j // top_k)
    indices = topk_ids.reshape(1, -1).to(torch.long).contiguous()
    weights = topk_weights.reshape(1, -1).to(torch.float16).contiguous()
    scratch = torch.empty_like(rows)               # rotated input; must not alias the input (autotune relaunches)
    inter_g = torch.empty((tokens * top_k, 1, intermediate), dtype=torch.float16, device=x.device)
    inter_u = torch.empty_like(inter_g)
    common = dict(force_shape_idx=-1, mcg=mcg, mul1=mul1, min_index=-1, max_index=num_experts - 1,
                  force_num_sms=0, num_tokens=tokens)
    ext.exl3_mgemm(rows, gate[0], inter_g, gate[1], scratch, gate[2], indices, None, bits_gate_up, **common)
    ext.exl3_mgemm(rows, up[0], inter_u, up[1], scratch, up[2], indices, None, bits_gate_up, **common)
    act = torch.nn.functional.silu(inter_g) * inter_u
    out = torch.empty((tokens * top_k, 1, hidden), dtype=torch.float16, device=x.device)
    # With weights, the kernel scales each slot and sums a token's top_k slots into rows [0, tokens).
    ext.exl3_mgemm(act, down[0], out, down[1], inter_g, down[2], indices, weights, bits_down, **common)
    return out[:tokens].squeeze(1).to(x.dtype)


class SlicedGroup:
    """Several EXL3 matrices that share an input (q/k/v, gate/up, GDN qkv/z) run as ONE exl3_mgemm launch in
    sliced mode: every matrix is cut into equal-width column slices scheduled as concurrent groups, and the input
    Hadamard runs once per source. Calling convention after ExLlamaV3's SlicedMultiLinear (MIT). Needs one K, one
    codebook, one input width; outputs land in persistent per-matrix buffers so the pointer tables are static
    (safe under CUDA graphs)."""

    def __init__(self, trellis, suh, svh, codebook: int, max_rows: int, min_width: int = 256):
        import math
        self.k = trellis[0].shape[0] * TILE
        self.widths = [t.shape[1] * TILE for t in trellis]
        self.bits = bits_of(trellis[0])
        if any(t.shape[0] * TILE != self.k or bits_of(t) != self.bits for t in trellis):
            raise ValueError("sliced group needs one input width and one bitrate")
        self.width = math.gcd(*self.widths)
        if self.width % BLOCK or self.width < min_width:
            raise ValueError(f"unsuitable slice width {self.width} for {self.widths}")
        dev = trellis[0].device
        self.mcg, self.mul1 = codebook == CODEBOOK_MCG, codebook == CODEBOOK_MUL1
        self.max_rows, self.num_src = max_rows, len(trellis)
        self.out = [torch.empty((max_rows, n), dtype=torch.float16, device=dev) for n in self.widths]
        self.x_had = torch.empty((self.num_src, max_rows, self.k), dtype=torch.float16, device=dev)
        t_ptrs, sv_ptrs, c_ptrs, strides, srcs = [], [], [], [], []
        for i, (t, sv, o) in enumerate(zip(trellis, svh, self.out)):
            for n0 in range(0, self.widths[i], self.width):
                t_ptrs.append(t.data_ptr() + (n0 // TILE) * t.shape[2] * t.element_size())
                sv_ptrs.append(sv.data_ptr() + n0 * sv.element_size())
                c_ptrs.append(o.data_ptr() + n0 * o.element_size())
                strides.append(self.widths[i])
                srcs.append(i)
        mk = lambda v, dt: torch.tensor(v, dtype=dt, device=dev)
        self.t_ptrs, self.sv_ptrs, self.c_ptrs = mk(t_ptrs, torch.long), mk(sv_ptrs, torch.long), mk(c_ptrs, torch.long)
        self.su_ptrs = mk([s.data_ptr() for s in suh], torch.long)
        self.size_n = torch.full((len(srcs),), self.width, dtype=torch.int32, device=dev)
        self.n_stride, self.had_src = mk(strides, torch.int32), mk(srcs, torch.int32)
        self.carrier = torch.empty((1, 1, self.width), dtype=torch.float16, device=dev)
        self._keep = (trellis, suh, svh)

    def tables(self) -> list[torch.Tensor]:
        """Everything run_tables needs, as tensors (custom ops cannot carry Python objects)."""
        return [self.t_ptrs, self.sv_ptrs, self.c_ptrs, self.su_ptrs, self.size_n, self.n_stride, self.had_src,
                self.carrier, self.x_had, *self.out]

    def run(self, x: torch.Tensor) -> list[torch.Tensor]:
        return run_sliced(x, self.tables(), self.bits, self.mcg, self.mul1)


def run_sliced(x: torch.Tensor, tables: list[torch.Tensor], bits: float, mcg: bool, mul1: bool) -> list[torch.Tensor]:
    t_ptrs, sv_ptrs, c_ptrs, su_ptrs, size_n, n_stride, had_src, carrier, x_had = tables[:9]
    outs = tables[9:]
    rows, k = x.shape
    if rows > outs[0].shape[0]:
        raise ValueError(f"sliced group was built for {outs[0].shape[0]} rows, got {rows}")
    scratch = x_had.view(-1)[: len(outs) * rows * k].view(len(outs), rows, k)   # one rotated-input slab per source
    _load().exl3_mgemm(x.view(1, rows, k), t_ptrs, carrier.expand(size_n.shape[0], rows, carrier.shape[2]), su_ptrs,
                       scratch, sv_ptrs, None, None, bits, -1, mcg, mul1, -1, -1, 0, 1,
                       size_n, c_ptrs, n_stride, had_src, len(outs))
    return [o[:rows] for o in outs]
