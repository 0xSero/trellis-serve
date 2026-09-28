"""Marlin backend: our Marlin-template EXL3 kernels (csrc/, module `trellis_exl3_kernels`).

K = 3, K = 4 and K = 6 (lm_head), codebooks MCG and MUL1. Decoded weights are bit-identical to ExLlamaV3's (parity/);
the accumulation order differs. The trellis is repacked once at load: K = 4 a pure permutation of the stream words into
a lane-major layout, K = 6 a lossless bit re-layout (each lane's 64-bit window, 16 bits stored twice), K = 3 the tiles
exactly as stored ([k/16, n/64, 4, 24] int32; the kernel stages them byte-exact, notes/k3-dense-plan.md).  # K3
K = 5 likewise ([k/16, n/64, 4, 40] int32, notes/k5-dense-plan.md).  # K5

Activations may be fp16 or bf16. The GEMM always runs on fp16 (ExLlamaV3 semantics); with bf16 the two conversions
happen inside the Hadamard launches and are exactly torch's `.to(float16)` / `.to(bfloat16)` value conversions
(round to nearest even; bf16 magnitudes above 65504 become +-inf in fp16, as they did with the explicit casts), so
`run(x_bf16) == run(x_bf16.half()).bfloat16()` bit for bit (parity/marlin_parity.py checks it).
"""
from __future__ import annotations

import importlib
import os

import torch

from .base import BackendStatus

_mod = None
# Caller-owned, reused scratch for the rotated input (fp16, shards * rows * k), one per device, sized at load time
# (`prepare`) for MAX_ROWS rows so the serving path never allocates it per call and CUDA graphs see a fixed address.
MAX_ROWS = int(os.environ.get("TRELLIS_EXL3_MARLIN_MAX_ROWS", "64"))
_scratch: dict[int, torch.Tensor] = {}


def _reserve_scratch(device: torch.device, numel: int) -> None:
    cur = _scratch.get(device.index)
    if cur is None or cur.numel() < numel:
        _scratch[device.index] = torch.empty(numel, dtype=torch.float16, device=device)


def _load():
    global _mod
    if _mod is None:
        _mod = importlib.import_module("trellis_exl3_kernels")
    return _mod


def probe() -> BackendStatus:
    try:
        _load()
    except ImportError as e:
        return BackendStatus(False, detail=str(e))
    return BackendStatus(True)


def supports(trellis: list[torch.Tensor], out_widths: list[int], codebooks: list[int]) -> tuple[bool, str]:
    if len({t.shape[2] for t in trellis}) != 1 or trellis[0].shape[2] not in (48, 64, 80, 96):   # K3, K5
        return False, "one bitrate per group, K=3, K=4, K=5 or K=6"
    if any((t.shape[0] * 16) % 128 or (t.shape[1] * 16) % 128 for t in trellis):
        return False, "k, n not multiples of 128"
    if len(set(codebooks)) != 1 or codebooks[0] not in (1, 2):
        return False, "one codebook per group, mcg or mul1"
    if any(w != t.shape[1] * 16 for w, t in zip(out_widths, trellis)):
        return False, "padded output columns"
    if len({t.shape[0] for t in trellis}) != 1:
        return False, "different input widths"
    return True, ""


def prepare(trellis: list[torch.Tensor], suh: list[torch.Tensor], svh: list[torch.Tensor]):
    """-> (b_cat int32 [k/16, n_total/64, 32, 4(, 2)] (K=4, K=6) | [k/16, n_total/64, 4, 24] (K3) | [k/16, n_total/64, 4, 40] (K5), suh_cat [shards, k],
    svh_cat [n_total], shard_ends)"""
    mod = _load()
    mod.init_device(trellis[0].device.index)       # per-device state must exist before any CUDA-graph capture
    b = torch.cat([mod.repack_trellis(t) for t in trellis], dim=1).contiguous()
    ends, total = [], 0
    for t in trellis[:-1]:
        total += t.shape[1] * 16
        ends.append(total)
    _reserve_scratch(b.device, len(trellis) * MAX_ROWS * trellis[0].shape[0] * 16)
    return b, torch.stack(suh).contiguous(), torch.cat(svh).contiguous(), ends


def run(x: torch.Tensor, b: torch.Tensor, suh_cat: torch.Tensor, svh_cat: torch.Tensor, shard_ends: list[int],
        codebook: int) -> torch.Tensor:
    """x fp16 | bf16 [rows, k] contiguous -> y (same dtype) [rows, n_total], one fused launch set for the whole group.
    The rotated-input scratch is the per-device buffer reserved in `prepare`; only y is allocated."""
    mod = _load()
    rows, k = x.shape
    shards = len(shard_ends) + 1
    y = torch.empty((rows, svh_cat.shape[0]), dtype=x.dtype, device=x.device)
    xh = _scratch.get(x.device.index)
    if xh is None or xh.numel() < shards * rows * k:
        xh = torch.empty(shards * rows * k, dtype=torch.float16, device=x.device)
    if shards == 1:
        mod.exl3_linear_marlin_out(x, b, suh_cat.view(-1), svh_cat, codebook, xh, y)
    else:
        mod.exl3_linear_marlin_multi_out(x, b, suh_cat, svh_cat, shard_ends, codebook, xh, y)
    return y


# ---------------------------------------------------------------------------------------------------------------
# Building blocks used by parity/marlin_parity.py and tools/marlin_microbench.py (caller-owned buffers, rotated-basis
# GEMM for reading the decoded weights out, single matrices). The serving path above uses supports/prepare/run.

def had_in_group(x: torch.Tensor, suh_cat: torch.Tensor, xh: torch.Tensor | None = None) -> torch.Tensor:
    """Many-row path: xh[s] = Had128(fp16(x) * suh_cat[s]) for all shards in one launch. x fp16 | bf16 [rows, k]
    -> fp16 [shards * rows, k] (slab s = rows s*rows .. (s+1)*rows-1)."""
    rows, k = x.shape
    if xh is None:
        xh = torch.empty((suh_cat.numel() // k * rows, k), dtype=torch.float16, device=x.device)
    _load().had_in_group_marlin(x, xh, suh_cat)
    return xh


def had_out_into(y: torch.Tensor, svh: torch.Tensor, out: torch.Tensor, col0: int) -> None:
    """Many-row path: out[:, col0:col0+n] = Had128(y) * svh in out's dtype (fp16 | bf16); y fp16 [rows, n]."""
    _load().had_out_into_marlin(y, svh, out, col0)


def set_had_warps(warps: int) -> None:
    """Knob: 128-blocks per thread block of the Hadamard launches at >= 32 rows (8 = default, 1 = ExLlamaV3's geometry)."""
    _load().set_had_warps(int(warps))


def set_out_had_inlaunch(on: bool) -> None:
    """Knob: output Hadamard inside the GEMM launch (default) or as a separate launch. Same bits either way."""
    _load().set_out_had_inlaunch(bool(on))


def set_in_had_inlaunch(on: bool) -> None:
    """Knob (experimental, default off): input Hadamard as the prologue of a cooperative GEMM launch."""
    _load().set_in_had_inlaunch(bool(on))


def supports_matrix(trellis: torch.Tensor) -> tuple[bool, str]:
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    if trellis.shape[2] not in (48, 64, 80, 96):   # K3, K5
        return False, f"K = {trellis.shape[2] / 16:g} (only K = 3, 4, 5 and 6)"
    if k % 128 or n % 128:
        return False, f"k, n = {k}, {n} not multiples of 128"
    return True, ""


def prepare_matrix(trellis: torch.Tensor) -> torch.Tensor:
    """int16 (k/16, n/16, 16K) -> int32 (k/16, n/64, 32, 4) for K=4 (same stream words, lane-major order),
    (k/16, n/64, 32, 4, 2) for K=6 (per lane and tile the 64 stream bits its 8 windows live in) or
    (k/16, n/64, 4, 24) for K=3 (the tiles as stored, K3) or (k/16, n/64, 4, 40) for K=5 (K5)."""
    mod = _load()
    mod.init_device(trellis.device.index)
    return mod.repack_trellis(trellis)


def unprepare_matrix(packed: torch.Tensor) -> torch.Tensor:
    return _load().unpack_trellis(packed)


def gemm_rotated(xh: torch.Tensor, packed: torch.Tensor, codebook: int, out: torch.Tensor | None = None,
                 thread_k: int = -1, thread_n: int = -1) -> torch.Tensor:
    """xh @ W_hat in the rotated basis (no Hadamards). Identity rows of xh read W_hat out exactly."""
    if out is None:
        out = torch.empty((xh.shape[0], packed.shape[1] * 64), dtype=torch.float16, device=xh.device)
    _load().exl3_gemm_marlin(xh, packed, out, codebook, thread_k, thread_n)
    return out


def gemm_rotated_group(xh_all: torch.Tensor, packed: torch.Tensor, shard_ends, codebook: int,
                       out: torch.Tensor | None = None) -> torch.Tensor:
    """Rotated-basis GEMM of a fused group; xh_all is (shards * rows, k), slab s feeds shard s."""
    rows = xh_all.shape[0] // (len(shard_ends) + 1)
    if out is None:
        out = torch.empty((rows, packed.shape[1] * 64), dtype=torch.float16, device=xh_all.device)
    _load().exl3_gemm_marlin_multi(xh_all, packed, out, list(shard_ends), codebook)
    return out


def linear(x: torch.Tensor, packed: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: int,
           xh: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """One matrix: y = Had(x * suh) @ W_hat -> Had -> * svh. With `xh` and `out` given nothing is allocated.
    x / out: fp16 or bf16 (independently); xh: fp16."""
    if xh is None or out is None:
        return _load().exl3_linear_marlin(x, packed, suh, svh, codebook)
    _load().exl3_linear_marlin_out(x, packed, suh, svh, codebook, xh, out)
    return out


def linear_group(x, packed, suh_cat, svh_cat, shard_ends, codebook: int, xh=None, out=None) -> torch.Tensor:
    """Fused group (output of `prepare`) in 3 launches; `out` is (rows, sum(n)), shard s occupies its column span."""
    rows, k = x.shape
    if xh is None:
        xh = torch.empty((suh_cat.shape[0] * rows, k), dtype=torch.float16, device=x.device)
    if out is None:
        out = torch.empty((rows, svh_cat.shape[0]), dtype=x.dtype, device=x.device)
    _load().exl3_linear_marlin_multi_out(x, packed, suh_cat, svh_cat, list(shard_ends), codebook, xh, out)
    return out
