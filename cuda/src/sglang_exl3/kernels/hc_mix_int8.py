"""8-bit hyper-connection low-rank mix (Qwen3.8-Flash-Next GatedResidual) for sm_86 (no fp8 math: int8 weights, scales).

The HC mix of one site (hc = 4 streams, hs = 2560, lowrank = 320):
    t   = silu((x @ W_down^T) / hc)            x: [rows, hc*hs] normed input, W_down: [lowrank, hc*hs]
    g   = sigmoid(t @ W_up^T)                  W_up: [hc*hs, lowrank]
    out = mean_over_streams(g * x)             -> [rows, hs]
96 sites + the final mixer read 1.28 GB of bf16 weights per decode token (SGLang's `_hc_mix_persistent_kernel`,
2.15 ms/token on the 3090). Here the weights are int8 with symmetric scales:
    group = 0 : one fp32 scale per output row (W_down: per lowrank row; W_up: per (stream, hidden) row) -> applied to the
                fp32 dot result (exact rescale; the int8 -> bf16 cast of the codes is exact)
    group = G : one scale per G consecutive inputs of a row, per matrix (group_down, group_up) -> the dot runs per
                G-wide sub-block and each fp32 partial is rescaled (exact, no per-element dequantisation)
`fused_hc_mix_int8` mirrors SGLang's persistent kernel (one CTA per SM, grid barriers, device-scope atomics for the
split-K down projection, last CTA resets the barrier counters so CUDA-graph replays start clean); rows <= 16.
`hc_mix_int8_torch` is the large-row (prefill) path: dequantise one site transiently and run the reference math.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 16


class HcInt8Weights:
    """Quantised W_down / W_up of one GatedResidual site. group_down / group_up: 0 = one scale per output row, else one
    scale per `group` consecutive inputs of a row (applied to the fp32 dot of that sub-block: exact rescale)."""

    def __init__(self, w_down: torch.Tensor, w_up: torch.Tensor, group_down: int = 0, group_up: int = 0):
        self.group_down, self.group_up = group_down, group_up
        self.lowrank, self.k = w_down.shape
        self.n_up = w_up.shape[0]
        self.q_down, self.s_down = quantize(w_down, group_down)
        self.q_up, self.s_up = quantize(w_up, group_up)
        self.dtype = w_down.dtype

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q_down, self.s_down, self.q_up, self.s_up))

    def dequant(self, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        return dequantize(self.q_down, self.s_down, self.group_down, dtype or self.dtype), \
            dequantize(self.q_up, self.s_up, self.group_up, dtype or self.dtype)


def quantize(w: torch.Tensor, group: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8: q = round(w / s), s = max|w| / 127 per row (group 0) or per `group` inputs of a row."""
    wf = w.float()
    if group:
        rows, cols = wf.shape
        if cols % group:
            raise ValueError(f"cols {cols} not a multiple of group {group}")
        g = wf.view(rows, cols // group, group)
        s = g.abs().amax(-1).clamp(min=1e-12) / 127.0
        q = torch.round(g / s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8).view(rows, cols)
        return q.contiguous(), s.contiguous()
    s = wf.abs().amax(-1).clamp(min=1e-12) / 127.0
    q = torch.round(wf / s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), s.contiguous()


def dequantize(q, s, group, dtype):
    if group:
        rows, cols = q.shape
        return (q.float().view(rows, cols // group, group) * s.unsqueeze(-1)).view(rows, cols).to(dtype)
    return (q.float() * s.unsqueeze(-1)).to(dtype)


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _hc_mix_int8_kernel(
    x_ptr, qd_ptr, sd_ptr, qu_ptr, su_ptr, t_raw_ptr, out_ptr, counters_ptr,
    K, LOWRANK, HS, num_rows, num_ctas, inv_hc,
    ROWS: tl.constexpr, HC: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_R: tl.constexpr, GD: tl.constexpr, GU: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    dt = x_ptr.dtype.element_ty

    # t_raw is a persistent per-device accumulator that the LAST CTA of the previous call left zeroed (allocated zeroed):
    # no zero pass + grid barrier at the start (SGLang's kernel spends one barrier per site on it)

    # ---- down projection (split-K over the CTAs, fp32 atomics)
    SUBK: tl.constexpr = BLOCK_K if GD == 0 else GD
    offs_k = tl.arange(0, SUBK)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        mask_n = n < LOWRANK
        acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
        for sb in tl.static_range(BLOCK_K // SUBK):
            k = kc * BLOCK_K + sb * SUBK + offs_k
            xt = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None], other=0.0)
            q = tl.load(qd_ptr + n[:, None] * K + k[None, :], mask=mask_n[:, None], other=0)
            part = tl.dot(xt, tl.trans(q.to(dt)))
            if GD == 0:
                acc += part
            else:
                s = tl.load(sd_ptr + n * (K // GD) + (kc * BLOCK_K + sb * SUBK) // GD, mask=mask_n, other=0.0)
                acc += part * s[None, :]
        if GD == 0:
            s = tl.load(sd_ptr + n, mask=mask_n, other=0.0)
            acc = acc * s[None, :]
        tl.atomic_add(t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :], acc, mask=mask_n[None, :],
                      sem="relaxed", scope="gpu")
    _grid_barrier(counters_ptr + 1, num_ctas)

    # ---- up projection + gate + stream mean
    SUBR: tl.constexpr = BLOCK_R if GU == 0 else GU
    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, SUBR)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,))
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, SUBR):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :], mask=mask_r[None, :], other=0.0)
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(dt)
            q = tl.load(qu_ptr + gj_flat[:, None] * LOWRANK + r[None, :], mask=mask_gj[:, None] & mask_r[None, :], other=0)
            if GU == 0:
                acc = tl.dot(t, tl.trans(q.to(dt)), acc)
            else:
                s = tl.load(su_ptr + gj_flat * (LOWRANK // GU) + r0 // GU, mask=mask_gj, other=0.0)
                acc += tl.dot(t, tl.trans(q.to(dt))) * s[None, :]
        if GU == 0:
            su = tl.load(su_ptr + gj_flat, mask=mask_gj, other=0.0)
            acc = acc * su[None, :]
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(x_ptr + offs_m[:, None, None] * (HC * HS) + offs_g[None, :, None] * HS + j[None, None, :],
                     mask=mask_m[:, None, None] & mask_j[None, None, :], other=0.0).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(out_ptr + offs_m[:, None] * HS + j[None, :], out.to(out_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_j[None, :])

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        # every CTA has finished reading t_raw: leave it zeroed for the next call, reset the barrier counters
        zero_span = ROWS * LOWRANK
        offs_z = tl.arange(0, 1024)
        for z0 in range(0, zero_span, 1024):
            idx = z0 + offs_z
            tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


CFG = {"BLOCK_N": 32, "BLOCK_K": 256, "BLOCK_J": 32, "BLOCK_R": 64, "num_warps": 8}   # tuned in K06 (tools/hc_int8_bench)


_counters: dict = {}
_traw: dict = {}
_nsm: dict = {}


def fused_hc_mix_int8(x: torch.Tensor, w: HcInt8Weights, hc: int, hs: int, cfg: dict | None = None) -> torch.Tensor:
    rows, k = x.shape
    dev = x.device
    if rows > MAX_ROWS:
        raise ValueError("fused_hc_mix_int8: rows > 16, use hc_mix_int8_torch")
    num_ctas = _nsm.get(dev)
    if num_ctas is None:
        num_ctas = _nsm[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    cnt = _counters.get(dev)
    if cnt is None:
        cnt = _counters[dev] = torch.zeros(3, dtype=torch.int32, device=dev)
    t_raw = _traw.get((dev, w.lowrank))
    if t_raw is None:                     # persistent, zeroed once; the kernel's last CTA re-zeroes it after each call
        t_raw = _traw[(dev, w.lowrank)] = torch.zeros((MAX_ROWS, w.lowrank), dtype=torch.float32, device=dev)
    out = torch.empty((rows, hs), dtype=x.dtype, device=dev)
    if rows == 0:
        return out
    c = cfg or CFG
    _hc_mix_int8_kernel[(num_ctas,)](
        x, w.q_down, w.s_down, w.q_up, w.s_up, t_raw, out, cnt, k, w.lowrank, hs, rows, num_ctas, 1.0 / hc,
        ROWS=MAX_ROWS, HC=hc, BLOCK_N=c["BLOCK_N"], BLOCK_K=c["BLOCK_K"], BLOCK_J=c["BLOCK_J"], BLOCK_R=c["BLOCK_R"],
        GD=w.group_down, GU=w.group_up, num_warps=c["num_warps"])
    return out


def hc_mix_reference(x, w_down, w_up, hc, hs, compute_dtype=None):
    """SGLang's `_mix_compute` math (optionally in fp32)."""
    if compute_dtype is not None:
        x, w_down, w_up = x.to(compute_dtype), w_down.to(compute_dtype), w_up.to(compute_dtype)
    t = torch.nn.functional.silu(torch.nn.functional.linear(x, w_down) / hc)
    g = torch.sigmoid(torch.nn.functional.linear(t, w_up)).unflatten(-1, (hc, hs))
    return (g * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def hc_mix_int8_torch(x: torch.Tensor, w: HcInt8Weights, hc: int, hs: int) -> torch.Tensor:
    """Any row count (prefill): transient bf16 dequant of this site (13 MB) + the reference math."""
    wd, wu = w.dequant(x.dtype)
    return hc_mix_reference(x, wd, wu, hc, hs)
