"""Custom ops. One opaque op per layer kind: vLLM traces `apply()` once with Dynamo and freezes it,
so every row-count branch, dtype cast, pad and trim lives in here."""
from __future__ import annotations

import torch

from ..kernels import marlin, reference

_lib = torch.library.Library("sglang_exl3", "FRAGMENT")  # must stay alive at module scope
import os

# Row-count switch points: ExLlamaV3's trellis kernel gets steadily more expensive per step as rows grow, while a
# resident decoded weight + cuBLAS is roughly flat; decoding the weight per call adds about one rows=1 kernel pass.
# ExLlamaV3's own switch point (144) is tuned for bandwidth-starved consumer GPUs. To be replaced by the per-shape
# dispatch plan.
DENSE_ROWS = int(os.environ.get("SGLANG_EXL3_DENSE_ROWS", "144"))   # 3090: kernel wins to ~128 rows (ExLlamaV3 switch 144)                 # transient decoded weight
DENSE_CACHED_ROWS = int(os.environ.get("SGLANG_EXL3_DENSE_CACHED_ROWS", "16"))   # resident decoded weight
# Below the dense switch, matrices that share an input run as one sliced multi-matrix launch (what ExLlamaV3 itself
# does for q/k/v). The fused slice wins from around 8 rows; at 1-2 rows the separate int8 GEMV launches still win
# for wide groups, so each group carries its own minimum row count.


# Switch points of OUR kernel (all 27B linears per forward): the kernel stays ahead of decode + cuBLAS up to a few
# hundred rows (speculative verification batches, chunked-prefill tails), so it serves everything below 256 rows;
# with resident decoded weights the dense path has no reconstruct and wins from ~48.
MARLIN_DENSE_ROWS = int(os.environ.get("SGLANG_EXL3_MARLIN_DENSE_ROWS", "256"))
MARLIN_DENSE_CACHED_ROWS = int(os.environ.get("SGLANG_EXL3_MARLIN_DENSE_CACHED_ROWS", "48"))
_MARLIN_DENSE = os.environ.get("SGLANG_EXL3_MARLIN_DENSE", "1") == "1"
_MARLIN_DENSE_MAX_N = 65536     # lm_head-sized outputs keep the column-sliced reference path (bounded transients)
# Prefill GEMM accumulation. GeForce Ampere runs fp16 tensor-core MMA at 2x the rate with fp16 accumulate (3090, 27B
# linear shapes at 8192 rows: 70 TFLOPS fp32-acc vs 120 TFLOPS fp16-acc, bench/prefill_gemm_ceiling.py). The error
# class (4e-3 rel. vs fp64 at K=17408) is the one ExLlamaV3's own trellis kernels have (2-8e-3). Opt-in.
_PREFILL_ACC16 = os.environ.get("SGLANG_EXL3_PREFILL_FP16_ACC", "0") == "1"


def dense_group(x: torch.Tensor, trellis: list[torch.Tensor], suh_cat: torch.Tensor, svh: list[torch.Tensor],
                codebook: int, out: torch.Tensor, weights: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Many-row (prefill) path of a fused group on our Hadamard launches: x fp16 | bf16 [rows, k] goes in as is, ONE
    batched input transform for all matrices, per matrix decode (ExLlamaV3's reconstruct, or the resident `weights`)
    + cuBLAS fp16 GEMM, and an output transform that writes straight into the matrix's column span of `out` in
    out's dtype. Same arithmetic as ExLlamaV3's reconstruct path and as the explicit-cast path it
    replaces (the conversions are torch's value conversions, kernels/marlin.py), minus 2 casts, 1 copy and
    len(trellis) - 1 input-transform launches per call."""
    rows = x.shape[0]
    xh = marlin.had_in_group(x, suh_cat)
    col0 = 0
    mm = torch.backends.cuda.matmul
    prev = mm.allow_fp16_accumulation
    mm.allow_fp16_accumulation = _PREFILL_ACC16 or prev
    try:
        for i, (t, sv) in enumerate(zip(trellis, svh)):
            w = weights[i] if weights is not None else reference.reconstruct(t, codebook)
            y = torch.mm(xh[i * rows:(i + 1) * rows], w)
            del w
            marlin.had_out_into(y, sv, out, col0)
            col0 += y.shape[1]
    finally:
        mm.allow_fp16_accumulation = prev
    return out


def _linear(x: torch.Tensor, trellis: list[torch.Tensor], suh: list[torch.Tensor], svh: list[torch.Tensor],
            dense: list[torch.Tensor], sliced: list[torch.Tensor], sliced_min_rows: int,
            marlin_w: list[torch.Tensor], marlin_ends: list[int],
            codebooks: list[int], out_widths: list[int]) -> torch.Tensor:
    """Fused EXL3 linear: every matrix reads the same x and writes an adjacent output span.
    Each matrix has its own input scales, bitrate and codebook, so they are launched one by one."""
    k = (trellis[0].shape[0] if trellis else marlin_w[0].shape[0]) * 16
    rows = x.reshape(-1, x.shape[-1])

    def _unpacked():
        # memory-lean layout (3090): only the repacked form is resident; rebuild the int16 trellis per shard on demand.
        # Only for the many-row (prefill) path: never on the decode path (an 80 MB copy per group per step).
        if trellis:
            return trellis
        b = marlin_w[0]
        bounds = [0, *[e // 64 for e in marlin_ends], b.shape[1]]
        return [marlin.unprepare_matrix(b[:, a:c].contiguous()) for a, c in zip(bounds[:-1], bounds[1:])]
    if (marlin_w and rows.shape[0] < (MARLIN_DENSE_CACHED_ROWS if dense else MARLIN_DENSE_ROWS) and rows.shape[1] == k
            and rows.dtype in (torch.float16, torch.bfloat16)):
        # Our Marlin-template kernel: one launch set for the whole fused group, written straight into the fused
        # output. bf16 goes in and comes out as is: the kernel converts bf16 -> fp16 on load and fp16 -> bf16 on
        # store with torch's value conversions, i.e. exactly what `.to(float16)` ... `.to(x.dtype)` did here before
        # (incl. bf16 magnitudes > 65504 -> inf), without the two cast launches and their temporaries.
        y = marlin.run(rows.contiguous(), marlin_w[0], marlin_w[1], marlin_w[2], marlin_ends, codebooks[0])
        return y.view(*x.shape[:-1], y.shape[1])
    if (marlin_w and rows.shape[1] == k and rows.dtype in (torch.float16, torch.bfloat16)
            and max(out_widths) <= _MARLIN_DENSE_MAX_N and _MARLIN_DENSE):
        # Many rows (prefill, big verification batches): decoded weight + cuBLAS between OUR transforms, bf16 in/out,
        # one input-transform launch per fused group, results written straight into the fused output.
        out = torch.empty((rows.shape[0], sum(out_widths)), dtype=x.dtype, device=x.device)
        ends = [0, *marlin_ends, out.shape[1]]
        svs = [marlin_w[2][a:b] for a, b in zip(ends[:-1], ends[1:])]
        dense_group(rows.contiguous(), _unpacked(), marlin_w[1], svs, codebooks[0], out, dense if dense else None)
        return out.view(*x.shape[:-1], out.shape[1])
    trellis = _unpacked()          # remaining paths (padded input width, fp32 caller, huge outputs) need the int16 form
    h = rows.to(torch.float16)
    if h.shape[1] < k:                               # stored input width is padded to 128
        h = torch.nn.functional.pad(h, (0, k - h.shape[1]))
    h = h.contiguous()
    padded = any(w > t.shape[1] * 16 for w, t in zip(out_widths, trellis))
    out = (torch.zeros if padded else torch.empty)((h.shape[0], sum(out_widths)), dtype=x.dtype, device=x.device)
    offset = 0
    n_rows = h.shape[0]
    dense_switch = DENSE_CACHED_ROWS if dense else DENSE_ROWS
    if marlin_w and n_rows < dense_switch:       # padded input width or an fp32 caller: explicit casts
        y = marlin.run(h, marlin_w[0], marlin_w[1], marlin_w[2], marlin_ends, codebooks[0])
        return y.to(x.dtype).view(*x.shape[:-1], y.shape[1])
    if sliced and sliced_min_rows <= n_rows < dense_switch and n_rows <= sliced[9].shape[0]:   # tables[9] = first out buffer
        ys = reference.run_sliced(h, sliced, trellis[0].shape[2] / 16, codebooks[0] == 1, codebooks[0] == 2)
        for y, width in zip(ys, out_widths):
            keep = min(width, y.shape[1])
            out[:, offset:offset + keep] = y[:, :keep]
            offset += width
        return out.view(*x.shape[:-1], out.shape[1])
    for i, (t, su, sv, cb, width) in enumerate(zip(trellis, suh, svh, codebooks, out_widths)):
        if dense and n_rows >= DENSE_CACHED_ROWS:
            y = reference.dense_cached(h, dense[i], su, sv)
        elif n_rows >= DENSE_ROWS:
            y = reference.dense_forward(h, t, su, sv, cb)
        else:
            y = reference.gemm(h, t, su, sv, cb)
        keep = min(width, y.shape[1])                # stored output columns beyond the true width are noise
        out[:, offset:offset + keep] = y[:, :keep]
        offset += width
    return out.view(*x.shape[:-1], out.shape[1])


def _linear_fake(x, trellis, suh, svh, dense, sliced, sliced_min_rows, marlin_w, marlin_ends, codebooks, out_widths):
    return x.new_empty((*x.shape[:-1], sum(out_widths)))


_lib.define("linear(Tensor x, Tensor[] trellis, Tensor[] suh, Tensor[] svh, Tensor[] dense, Tensor[] sliced, int sliced_min_rows, Tensor[] marlin_w, int[] marlin_ends, int[] codebooks, int[] out_widths) -> Tensor")
_lib.impl("linear", _linear, "CUDA")
torch.library.register_fake("sglang_exl3::linear", _linear_fake, lib=_lib)

linear = torch.ops.sglang_exl3.linear
