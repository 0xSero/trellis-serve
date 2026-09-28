"""EXL3 tensor-level specification.

Pure Python. Everything here is derived from tensor names, dtypes and shapes
(safetensors headers); nothing is read from quantization_config JSON.
Reference: the EXL3 format as written by ExLlamaV3's convert.py (turboderp-org/exllamav3).
(exllamav3 @ 6b84a21b, v1.5.1).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping

BLOCK = 128  # Hadamard block width; both dims of a stored matrix are padded to it
TILE = 16    # trellis tile edge: one ring of 256 weights per 16x16 tile

MCG_MULTIPLIER = 0xCBAC1FED
MUL1_MULTIPLIER = 0x83DCD12D

# Every suffix the reference reads under a quantized linear's key. Anything
# else under such a key means a derivative format (rank-sliced, shared-H, ...)
# or a future upstream change, and must be rejected rather than ignored.
REQUIRED_TRELLIS = "trellis"
INPUT_SCALE = ("suh", "su")    # fp16 sign*scale vector, or legacy packed sign bits
OUTPUT_SCALE = ("svh", "sv")
MARKERS = ("mcg", "mul1")
KNOWN_SUFFIXES = frozenset({REQUIRED_TRELLIS, *INPUT_SCALE, *OUTPUT_SCALE, *MARKERS, "bias"})


class FormatError(ValueError):
    """The checkpoint violates the EXL3 structure we know how to run."""


class Codebook(enum.Enum):
    INST3 = "3inst"   # no marker tensor
    MCG = "mcg"
    MUL1 = "mul1"


@dataclass(frozen=True, order=True)
class BitsK:
    """Trellis bitrate K. Stored doubled so 1.5 / 2.5 / 3.5 stay exact."""
    twice: int

    @property
    def value(self) -> float:
        return self.twice / 2

    @property
    def is_half(self) -> bool:
        return self.twice % 2 == 1

    @property
    def trellis_words(self) -> int:
        # 16*K for integer K, 16*floor(K)+8 for half K: both are 8 * (2K).
        return 8 * self.twice

    @classmethod
    def from_trellis_words(cls, words: int, codebook: Codebook) -> "BitsK":
        if words % 8 != 0 or not 2 <= words // 8 <= 16:
            raise FormatError(f"trellis last dim {words} is not a valid EXL3 bitrate")
        k = cls(words // 8)
        if k.is_half:
            if k.twice not in (3, 5, 7):
                raise FormatError(f"half-integer K={k.value} is not defined (only 1.5, 2.5, 3.5)")
            if codebook is not Codebook.MUL1:
                raise FormatError(f"half-integer K={k.value} requires the mul1 codebook, got {codebook.value}")
        return k

    def __str__(self) -> str:
        return f"{self.value:g}"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str              # safetensors dtype string: "I16", "F16", "BF16", ...
    shape: tuple[int, ...]
    file: str = ""
    nbytes: int = 0


@dataclass(frozen=True)
class MatrixSpec:
    """One EXL3 linear as stored: y = Had(x * suh) @ W_hat -> Had -> * svh (+ bias).

    k / n are the *stored* (128-padded) input / output widths. True widths are
    not in the tensor files; they come from the model config via arch tables.
    Note the stored trellis is (k/16, n/16, words): input first, transposed
    relative to nn.Linear.weight.
    """
    key: str
    k: int
    n: int
    bits: BitsK
    codebook: Codebook
    has_bias: bool = False
    legacy_signs: bool = False   # su/sv packed bitfields instead of suh/svh
    tensors: Mapping[str, TensorInfo] = field(default_factory=dict, compare=False, repr=False)

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def same_kernel_class(self, other: "MatrixSpec") -> bool:
        """True if both can share one pointer-table / grouped launch."""
        return (self.k, self.n, self.bits, self.codebook) == (other.k, other.n, other.bits, other.codebook)


def _one_of(tensors: Mapping[str, TensorInfo], names: tuple[str, str], key: str) -> tuple[str, TensorInfo]:
    present = [s for s in names if s in tensors]
    if len(present) != 1:
        raise FormatError(f"{key}: expected exactly one of {names}, found {present or 'none'}")
    return present[0], tensors[present[0]]


def build_matrix_spec(key: str, tensors: Mapping[str, TensorInfo]) -> MatrixSpec:
    """Validate the tensors stored under one module key and describe them.

    `tensors` maps suffix -> TensorInfo for everything named `<key>.<suffix>`.
    Rejects on structure, never on the checkpoint's declared version.
    """
    unknown = sorted(set(tensors) - KNOWN_SUFFIXES)
    if unknown:
        raise FormatError(f"{key}: unknown tensors {unknown} under an EXL3 linear "
                          f"(derivative or newer format; refusing to guess)")
    trellis = tensors.get(REQUIRED_TRELLIS)
    if trellis is None:
        raise FormatError(f"{key}: no .trellis tensor")
    if trellis.dtype != "I16" or len(trellis.shape) != 3:
        raise FormatError(f"{key}.trellis: expected 3-D I16, got {trellis.dtype} {trellis.shape}")

    if all(m in tensors for m in MARKERS):
        raise FormatError(f"{key}: both .mcg and .mul1 markers present")
    codebook = Codebook.MCG if "mcg" in tensors else Codebook.MUL1 if "mul1" in tensors else Codebook.INST3
    for m in MARKERS:
        if m in tensors and (tensors[m].dtype != "I32" or tensors[m].shape not in ((), (1,))):
            raise FormatError(f"{key}.{m}: expected I32 scalar, got {tensors[m].dtype} {tensors[m].shape}")

    k, n = trellis.shape[0] * TILE, trellis.shape[1] * TILE
    if k % BLOCK or n % BLOCK:
        raise FormatError(f"{key}: stored dims ({k}, {n}) are not multiples of {BLOCK}")
    bits = BitsK.from_trellis_words(trellis.shape[2], codebook)

    in_name, in_t = _one_of(tensors, INPUT_SCALE, key)
    out_name, out_t = _one_of(tensors, OUTPUT_SCALE, key)
    legacy = in_name == "su" or out_name == "sv"
    if legacy and not (in_name == "su" and out_name == "sv"):
        raise FormatError(f"{key}: mixed legacy/modern scale tensors ({in_name}, {out_name})")
    want = ("I16", (k // TILE,), (n // TILE,)) if legacy else ("F16", (k,), (n,))
    for t, shape in ((in_t, want[1]), (out_t, want[2])):
        if t.dtype != want[0] or t.shape != shape:
            raise FormatError(f"{t.name}: expected {want[0]} {shape}, got {t.dtype} {t.shape}")

    bias = tensors.get("bias")
    if bias is not None and (bias.dtype not in ("F16", "F32") or bias.shape != (n,)):
        raise FormatError(f"{key}.bias: expected F16/F32 ({n},), got {bias.dtype} {bias.shape}")

    return MatrixSpec(key=key, k=k, n=n, bits=bits, codebook=codebook,
                      has_bias=bias is not None, legacy_signs=legacy, tensors=dict(tensors))
