"""The contract every kernel backend implements.

Backends see MatrixSpec + tensors + caller-owned workspace. They never see a
vLLM layer, never allocate on the run path, and never change weight values.
Selection happens once per (matrix group, rows bucket) at load time; a backend
that cannot serve a key says so with a reason instead of failing at run time.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from trellis_core.format import FusedLayout, MatrixSpec

if TYPE_CHECKING:  # keep this module importable without torch
    import torch


class Op(enum.Enum):
    LINEAR = "linear"    # one FusedLayout: shared input, adjacent output spans
    MOE = "moe"          # routed experts: gate/up/down groups


class ParityClass(enum.Enum):
    BIT_EXACT = "bit_exact"    # identical bits to pinned ExLlamaV3 at the pinned launch geometry
    NUMERICAL = "numerical"    # same math, different reduction order / precision; judged against measured floors


@dataclass(frozen=True)
class KernelKey:
    op: Op
    layout_class: tuple            # ((k, n, K*2, codebook), ...) per logical matrix (per projection for MOE)
    rows_bucket: int               # upper bound of the row-count bucket this plan entry serves
    in_dtype: str
    out_dtype: str
    device_capability: tuple[int, int]


@dataclass(frozen=True)
class Support:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class BackendStatus:
    available: bool
    version: str = ""
    missing: tuple[str, ...] = ()  # ops / libraries that could not be loaded
    detail: str = ""


@dataclass(frozen=True)
class WorkspaceSpec:
    nbytes: int
    alignment: int = 256


@runtime_checkable
class Backend(Protocol):
    name: str

    def probe(self) -> BackendStatus: ...
    def supports(self, key: KernelKey) -> Support: ...
    def parity(self, key: KernelKey) -> ParityClass: ...
    def graph_safe(self, key: KernelKey) -> bool: ...
    def workspace(self, key: KernelKey, max_rows: int) -> WorkspaceSpec: ...

    def prepare_linear(self, layout: FusedLayout, tensors: Sequence[dict[str, "torch.Tensor"]]) -> Any: ...
    def prepare_moe(self, gate: Sequence[MatrixSpec], up: Sequence[MatrixSpec], down: Sequence[MatrixSpec],
                    tensors: dict[str, Sequence[dict[str, "torch.Tensor"]]]) -> Any: ...

    def warm(self, handle: Any, key: KernelKey, workspace: "torch.Tensor") -> None:
        """Launch every kernel instance this key can reach once and pin its geometry.
        After this returns, run_* for this key must not allocate, synchronise, tune or initialise."""

    def run_linear(self, handle: Any, x: "torch.Tensor", out: "torch.Tensor", workspace: "torch.Tensor") -> None: ...
    def run_moe(self, handle: Any, x: "torch.Tensor", topk_ids: "torch.Tensor", topk_weights: "torch.Tensor",
                out: "torch.Tensor", workspace: "torch.Tensor", expert_map: "torch.Tensor | None") -> None: ...
