"""Header-only safetensors reader. Never touches tensor data.

A safetensors file is: u64 little-endian header length, JSON header, raw data.
The header alone gives name, dtype, shape and byte range of every tensor, which
is all the format layer needs (works on 750B-parameter checkpoints in seconds).
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

from .spec import FormatError, TensorInfo

_MAX_HEADER = 1 << 30  # expert-heavy shards have large headers; still reject garbage


@dataclass(frozen=True)
class ShardHeader:
    file: str
    metadata: dict[str, str]
    tensors: dict[str, TensorInfo]


def parse_header(raw: bytes, file: str = "") -> ShardHeader:
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise FormatError(f"{file}: safetensors header is not valid JSON: {e}") from None
    metadata = doc.pop("__metadata__", None) or {}
    tensors = {}
    for name, entry in doc.items():
        begin, end = entry["data_offsets"]
        tensors[name] = TensorInfo(name=name, dtype=entry["dtype"], shape=tuple(entry["shape"]),
                                   file=file, nbytes=end - begin)
    return ShardHeader(file=file, metadata=metadata, tensors=tensors)


def read_header(path: str | os.PathLike) -> ShardHeader:
    path = os.fspath(path)
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise FormatError(f"{path}: too short to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if not 2 <= length <= _MAX_HEADER:
            raise FormatError(f"{path}: implausible safetensors header length {length}")
        raw = f.read(length)
    if len(raw) != length:
        raise FormatError(f"{path}: truncated safetensors header")
    return parse_header(raw, file=os.path.basename(path))
