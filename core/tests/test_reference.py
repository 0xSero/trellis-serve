import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from trellis_core import reference as ref  # noqa: E402


def _mcg_independent(state: int) -> float:
    """The MCG codebook written out separately from reference.py, one state at a time."""
    x = (state * 0xCBAC1FED) & 0xFFFFFFFF
    x = ((x & 0x8FFF8FFF) ^ 0x3B603B60) & 0xFFFFFFFF
    lo = np.frombuffer(struct.pack("<H", x & 0xFFFF), dtype=np.float16)[0]
    hi = np.frombuffer(struct.pack("<H", x >> 16), dtype=np.float16)[0]
    return float(np.float16(np.float64(lo) + np.float64(hi)))


def test_mcg_codebook_matches_an_independent_implementation():
    lut = ref.codebook_lut(ref.CB_MCG).float().numpy()
    for s in (0, 1, 0xB38D, 0x7FFF, 0xFFFF, 12345):
        assert lut[s] == pytest.approx(_mcg_independent(s), abs=0), hex(s)


@pytest.mark.parametrize("cb", [ref.CB_3INST, ref.CB_MCG, ref.CB_MUL1])
def test_codebooks_are_bell_shaped_and_centred(cb):
    v = ref.codebook_lut(cb).float()
    assert v.numel() == 65536
    assert abs(v.mean().item()) < 0.05
    assert 1.0 < v.std().item() < 1.5       # EXL3 rescales by 1/1.24371 downstream


def test_linear_forward_equals_matmul_with_the_decoded_weight():
    torch.manual_seed(0)
    k, n, K = 256, 128, 3
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16)
    suh = (torch.randn(k) * 0.1).half()
    svh = (torch.randn(n) * 0.1).half()
    x = torch.randn(4, k).half()
    w = ref.weight_orig(trellis, suh, svh, K, ref.CB_MUL1)
    y = ref.linear_forward(x, trellis, suh, svh, K, ref.CB_MUL1)
    assert w.shape == (k, n)
    torch.testing.assert_close(y.float(), (x.float() @ w.float()), rtol=2e-2, atol=2e-2)
