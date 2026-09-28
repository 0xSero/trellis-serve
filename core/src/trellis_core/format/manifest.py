"""Checkpoint manifest built from safetensors headers.

Precedence: tensor headers are the truth about what
is quantized, at which K and with which codebook. quantization_config(.json)
is only cross-checked and reported as warnings; it can be missing or stale.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .safetensors_header import ShardHeader, read_header
from .spec import INPUT_SCALE, OUTPUT_SCALE, REQUIRED_TRELLIS, FormatError, MatrixSpec, TensorInfo, build_matrix_spec

NGRAM_FORMAT = "exl3_ngram_trellis"  # Qwen3.8-Flash-Next table: own container, 2-D ".trellis" tensors


@dataclass(frozen=True)
class NgramTable:
    file: str
    metadata: dict[str, str]
    tensors: dict[str, TensorInfo]

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())


@dataclass
class Manifest:
    matrices: dict[str, MatrixSpec] = field(default_factory=dict)       # module key -> EXL3 linear
    unquantized: dict[str, TensorInfo] = field(default_factory=dict)    # full tensor name -> info
    ngram_tables: list[NgramTable] = field(default_factory=list)
    quant_config: dict = field(default_factory=dict)                    # hint only
    warnings: list[str] = field(default_factory=list)

    def is_quantized(self, key: str) -> bool:
        return key in self.matrices

    def children(self, prefix: str) -> dict[str, MatrixSpec]:
        """EXL3 linears below a module prefix (e.g. all experts of one MoE block)."""
        p = prefix.rstrip(".") + "."
        return {k: m for k, m in self.matrices.items() if k.startswith(p)}

    def summary(self) -> dict:
        by_class = Counter((str(m.bits), m.codebook.value) for m in self.matrices.values())
        return {
            "quantized_linears": len(self.matrices),
            "quantized_bytes": sum(m.nbytes for m in self.matrices.values()),
            "unquantized_tensors": len(self.unquantized),
            "unquantized_bytes": sum(t.nbytes for t in self.unquantized.values()),
            "ngram_tables": len(self.ngram_tables),
            "ngram_bytes": sum(t.nbytes for t in self.ngram_tables),
            "by_bits_codebook": {f"K={k} {cb}": c for (k, cb), c in sorted(by_class.items())},
            "legacy_sign_linears": sum(m.legacy_signs for m in self.matrices.values()),
            "biased_linears": sum(m.has_bias for m in self.matrices.values()),
            "declared": {k: self.quant_config.get(k) for k in
                         ("version", "bits", "head_bits", "mtp_bits", "vision_bits", "codebook", "out_scales")
                         if k in self.quant_config},
            "warnings": list(self.warnings),
        }


def build_manifest(headers: list[ShardHeader], quant_config: dict | None = None,
                   weight_map: dict[str, str] | None = None) -> Manifest:
    """`weight_map` (model.safetensors.index.json) arbitrates tensors stored in more than one shard,
    which happens in the wild when MTP layers are appended by a later conversion step."""
    man = Manifest(quant_config=dict(quant_config or {}))
    weight_map = weight_map or {}
    tensors: dict[str, TensorInfo] = {}
    duplicates = []
    for h in headers:
        if h.metadata.get("format") == NGRAM_FORMAT:
            man.ngram_tables.append(NgramTable(h.file, h.metadata, h.tensors))
            continue
        for name, info in h.tensors.items():
            if name in tensors:
                owner = weight_map.get(name)
                if owner not in (tensors[name].file, h.file):
                    raise FormatError(f"tensor {name} appears in both {tensors[name].file} and {h.file} "
                                      f"and the index does not say which one is live")
                duplicates.append(name)
                if owner != h.file:
                    continue
            tensors[name] = info
    if duplicates:
        man.warnings.append(f"{len(duplicates)} tensors are stored in more than one shard "
                            f"(index decides), e.g. {duplicates[0]}")

    # A key is an EXL3 linear iff it owns a .trellis (the reference's rule also
    # demands the scale vectors; build_matrix_spec turns their absence into an error).
    keys = {name[: -len(REQUIRED_TRELLIS) - 1] for name in tensors if name.endswith("." + REQUIRED_TRELLIS)}
    for key in keys:  # e.g. rank-sliced derivatives: <linear>.rank0.trellis next to <linear>.trellis
        owner = next((k for k in _ancestors(key) if k in keys), None)
        if owner is not None:
            raise FormatError(f"{key}.trellis: unexpected tensor nested under EXL3 linear {owner}")
    grouped: dict[str, dict[str, TensorInfo]] = defaultdict(dict)
    for name, info in tensors.items():
        key, _, suffix = name.rpartition(".")
        if key in keys:
            grouped[key][suffix] = info
            continue
        # Anything nested deeper under a quantized linear is not stock EXL3.
        owner = next((k for k in _ancestors(key) if k in keys), None)
        if owner is not None:
            raise FormatError(f"{name}: unexpected tensor nested under EXL3 linear {owner}")
        man.unquantized[name] = info
    for key in sorted(grouped):
        man.matrices[key] = build_matrix_spec(key, grouped[key])

    _cross_check(man)
    return man


def _ancestors(key: str):
    while "." in key:
        key = key.rpartition(".")[0]
        yield key


def _cross_check(man: Manifest) -> None:
    declared = man.quant_config.get("codebook")
    if declared and man.matrices:
        seen = {m.codebook.value for m in man.matrices.values()}
        if seen != {declared}:
            man.warnings.append(f"quantization_config.codebook={declared!r} but tensors use {sorted(seen)}")
    stray = [n for n in man.unquantized
             if n.rpartition(".")[2] in (*INPUT_SCALE, *OUTPUT_SCALE) and n.rpartition(".")[0] not in man.matrices]
    if stray:
        man.warnings.append(f"{len(stray)} scale tensors without a trellis, e.g. {stray[0]}")


def load_manifest(model_dir: str | os.PathLike) -> Manifest:
    """Manifest of a local checkpoint directory (reads headers only)."""
    model_dir = os.fspath(model_dir)
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FormatError(f"{model_dir}: no .safetensors files")
    index = os.path.join(model_dir, "model.safetensors.index.json")
    headers = [read_header(f) for f in files]
    weight_map = {}
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f).get("weight_map", {})
    man = build_manifest(headers, _read_quant_config(model_dir), weight_map)
    if weight_map:
        have = {n for h in headers if h.metadata.get("format") != NGRAM_FORMAT for n in h.tensors}
        missing = sorted(set(weight_map) - have)
        if missing:
            raise FormatError(f"{len(missing)} tensors listed in the index are in no shard, e.g. {missing[0]}")
        unlisted = have - set(weight_map)
        if unlisted:  # e.g. MTP shards added later by convert_mtp.py
            man.warnings.append(f"{len(unlisted)} tensors are not in model.safetensors.index.json, "
                                f"e.g. {sorted(unlisted)[0]}")
    return man


def _read_quant_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    if qc and qc.get("quant_method") != "exl3":
        raise FormatError(f"{model_dir}: quant_method is {qc.get('quant_method')!r}, not 'exl3'")
    return qc
