"""Padding and tensor-parallel slicing arithmetic for stored EXL3 matrices.

Why slicing is exact: both Hadamard transforms are
block-diagonal with 128-wide blocks and the trellis is tiled 16x16, so a
128-aligned slice of (trellis, scale vector) is itself a valid EXL3 matrix that
reproduces exactly those rows / columns of the full one. No re-quantisation.
Anything not 128-aligned cannot be split this way and is refused here, with a
reason, so the caller can choose replication instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass

from .spec import BLOCK, TILE, MatrixSpec


def ceil_block(width: int) -> int:
    return -(-width // BLOCK) * BLOCK


class SplitError(ValueError):
    """A requested shard boundary is not representable by slicing."""


@dataclass(frozen=True)
class Narrow:
    dim: int
    start: int
    length: int


@dataclass(frozen=True)
class MatrixSlice:
    """How to cut one rank's share out of the stored tensors. None = keep whole."""
    trellis: Narrow
    suh: Narrow | None
    svh: Narrow | None
    bias: Narrow | None        # column split only
    keep_bias: bool            # row split: bias is added once, on the first shard

    def sliced_dims(self, spec: MatrixSpec) -> tuple[int, int]:
        k = self.suh.length if self.suh else spec.k
        n = self.svh.length if self.svh else spec.n
        return k, n


def _check(spec: MatrixSpec, side: str, width: int, start: int, size: int) -> None:
    if size <= 0 or start < 0 or start + size > width:
        raise SplitError(f"{spec.key}: {side} range [{start}, {start + size}) is outside the stored width {width}")
    if start % BLOCK or size % BLOCK:
        raise SplitError(f"{spec.key}: {side} range [{start}, {start + size}) is not aligned to the "
                         f"{BLOCK}-wide Hadamard blocks; this matrix cannot be sliced there (replicate it instead)")


def column_slice(spec: MatrixSpec, start: int, size: int) -> MatrixSlice:
    """Output-dim split (q/k/v, gate/up, lm_head). Input scales stay whole."""
    _check(spec, "output", spec.n, start, size)
    return MatrixSlice(trellis=Narrow(1, start // TILE, size // TILE), suh=None,
                       svh=Narrow(0, start, size), bias=Narrow(0, start, size), keep_bias=True)


def row_slice(spec: MatrixSpec, start: int, size: int) -> MatrixSlice:
    """Input-dim split (o_proj, down_proj). Output scales stay whole."""
    _check(spec, "input", spec.k, start, size)
    return MatrixSlice(trellis=Narrow(0, start // TILE, size // TILE), suh=Narrow(0, start, size),
                       svh=None, bias=None, keep_bias=start == 0)


def even_shard(width: int, tp_size: int, tp_rank: int) -> tuple[int, int]:
    """vLLM-style even partition of a true width: (start, size) for one rank."""
    if width % tp_size:
        raise SplitError(f"width {width} is not divisible by tp_size {tp_size}")
    size = width // tp_size
    return tp_rank * size, size


def kv_shard_index(tp_size: int, tp_rank: int, num_kv_heads: int) -> tuple[int, int]:
    """(shard index, shard count) for k/v projections; KV heads are replicated when tp_size > num_kv_heads."""
    if tp_size <= num_kv_heads:
        return tp_rank, tp_size
    if tp_size % num_kv_heads:
        raise SplitError(f"tp_size {tp_size} is not a multiple of num_kv_heads {num_kv_heads}")
    return tp_rank // (tp_size // num_kv_heads), num_kv_heads


def admissible_tp_degrees(width: int, candidates=(1, 2, 4, 8, 16)) -> list[int]:
    """TP degrees for which an even split of `width` stays 128-aligned."""
    return [tp for tp in candidates if width % tp == 0 and (tp == 1 or (width // tp) % BLOCK == 0)]
