"""Logical matrices inside a fused runtime module.

vLLM fuses q/k/v, gate/up, MLA's q_a + kv_a, ... into one module. In stock EXL3
every source matrix owns its input scale vector (`suh`) and may have its own K
and codebook, so a fused module is an ordered group of independent matrices
that share an input and write to adjacent output spans. This file only
describes that; how a backend launches it is the backend's business.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Hashable, Sequence

from .spec import MatrixSpec


@dataclass(frozen=True)
class LogicalMatrix:
    shard_id: Hashable      # vLLM's id for this source: "q" / "k" / "v", 0 / 1, None for a plain linear
    spec: MatrixSpec        # as seen by this rank (after TP slicing)
    out_width: int          # true output width on this rank; spec.n - out_width padded columns are trimmed


class FusionKind(enum.Enum):
    SINGLE = "single"               # one matrix, nothing to fuse
    UNIFORM = "uniform"             # same K / codebook / shapes: pointer-table multi-GEMM is possible
    SHARD_MAP = "shard_map"         # same input width, mixed K / codebook / n: needs a per-shard-aware kernel
    SEQUENTIAL = "sequential"       # fallback: one launch per matrix into slices of one output


@dataclass(frozen=True)
class FusedLayout:
    matrices: tuple[LogicalMatrix, ...]

    def __post_init__(self):
        if not self.matrices:
            raise ValueError("a fused layout needs at least one matrix")
        widths = {m.spec.k for m in self.matrices}
        if len(widths) != 1:
            raise ValueError(f"fused matrices must share their input width, got {sorted(widths)}")
        for m in self.matrices:
            if not 0 < m.out_width <= m.spec.n:
                raise ValueError(f"{m.spec.key}: out_width {m.out_width} does not fit stored width {m.spec.n}")

    @property
    def in_width(self) -> int:
        return self.matrices[0].spec.k

    @property
    def out_width(self) -> int:
        return sum(m.out_width for m in self.matrices)

    def spans(self) -> list[tuple[int, int]]:
        """(offset, width) of each matrix in the fused output, in order."""
        out, offset = [], 0
        for m in self.matrices:
            out.append((offset, m.out_width))
            offset += m.out_width
        return out

    @property
    def best_fusion(self) -> FusionKind:
        """The most fused execution this group admits; backends may do less."""
        if len(self.matrices) == 1:
            return FusionKind.SINGLE
        first = self.matrices[0].spec
        if all(m.spec.same_kernel_class(first) and m.out_width == m.spec.n for m in self.matrices):
            return FusionKind.UNIFORM
        return FusionKind.SHARD_MAP


def make_layout(shard_ids: Sequence[Hashable], specs: Sequence[MatrixSpec], out_widths: Sequence[int]) -> FusedLayout:
    if not len(shard_ids) == len(specs) == len(out_widths):
        raise ValueError("shard_ids, specs and out_widths must have the same length")
    return FusedLayout(tuple(LogicalMatrix(s, m, w) for s, m, w in zip(shard_ids, specs, out_widths)))
