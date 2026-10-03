import json
import struct

import pytest

from trellis_core.format import BitsK, Codebook, FormatError, TensorInfo, build_matrix_spec
from trellis_core.format.safetensors_header import parse_header


def _linear(k, n, words, marker="mul1"):
    t = {
        "trellis": TensorInfo("m.trellis", "I16", (k // 16, n // 16, words)),
        "suh": TensorInfo("m.suh", "F16", (k,)),
        "svh": TensorInfo("m.svh", "F16", (n,)),
    }
    if marker:
        t[marker] = TensorInfo(f"m.{marker}", "I32", ())
    return t


def test_spec_reads_bits_and_codebook_from_shapes():
    spec = build_matrix_spec("m", _linear(256, 512, 48))
    assert (spec.k, spec.n, spec.bits.value, spec.codebook) == (256, 512, 3.0, Codebook.MUL1)
    assert build_matrix_spec("m", _linear(256, 512, 64, "mcg")).codebook is Codebook.MCG
    assert build_matrix_spec("m", _linear(256, 512, 64, None)).codebook is Codebook.INST3


def test_half_bitrates_need_mul1():
    assert build_matrix_spec("m", _linear(128, 128, 40)).bits == BitsK(5)          # K = 2.5
    with pytest.raises(FormatError, match="requires the mul1 codebook"):
        build_matrix_spec("m", _linear(128, 128, 40, "mcg"))


def test_rejects_broken_linears():
    t = _linear(128, 128, 32)
    del t["suh"]
    with pytest.raises(FormatError, match="expected exactly one of"):
        build_matrix_spec("m", t)
    with pytest.raises(FormatError, match="not multiples of 128"):
        build_matrix_spec("m", _linear(144, 128, 32))
    t = _linear(128, 128, 32)
    t["extra"] = TensorInfo("m.extra", "F16", (1,))
    with pytest.raises(FormatError, match="unknown tensors"):
        build_matrix_spec("m", t)


def test_header_parser_reads_names_shapes_and_sizes():
    doc = {"__metadata__": {"format": "pt"},
           "m.trellis": {"dtype": "I16", "shape": [8, 8, 48], "data_offsets": [0, 6144]},
           "m.suh": {"dtype": "F16", "shape": [128], "data_offsets": [6144, 6400]}}
    raw = json.dumps(doc).encode()
    h = parse_header(raw, "shard.safetensors")
    assert h.metadata == {"format": "pt"}
    assert h.tensors["m.trellis"].shape == (8, 8, 48) and h.tensors["m.trellis"].nbytes == 6144
    assert struct.calcsize("<Q") == 8


def test_header_parser_rejects_a_reversed_byte_range():
    empty = {"w": {"dtype": "F16", "shape": [0], "data_offsets": [4, 4]}}
    parsed = parse_header(json.dumps(empty).encode(), "shard.safetensors")
    assert parsed.tensors["w"].nbytes == 0
    raw = json.dumps({"w": {"dtype": "F16", "shape": [2], "data_offsets": [10, 4]}}).encode()
    with pytest.raises(FormatError, match="reversed"):
        parse_header(raw, "shard.safetensors")
