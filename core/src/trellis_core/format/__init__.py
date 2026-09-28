"""L0: EXL3 on-disk format, manifest and shard geometry. Pure Python: no torch, no vllm."""
from .geometry import MatrixSlice, SplitError, admissible_tp_degrees, ceil_block, column_slice, row_slice
from .layout import FusedLayout, FusionKind, LogicalMatrix, make_layout
from .manifest import Manifest, build_manifest, load_manifest
from .spec import BitsK, Codebook, FormatError, MatrixSpec, TensorInfo, build_matrix_spec

__all__ = ["BitsK", "Codebook", "FormatError", "FusedLayout", "FusionKind", "LogicalMatrix", "Manifest",
           "MatrixSlice", "MatrixSpec", "SplitError", "TensorInfo", "admissible_tp_degrees", "build_manifest",
           "build_matrix_spec", "ceil_block", "column_slice", "load_manifest", "make_layout", "row_slice"]
