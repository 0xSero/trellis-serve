# The lossless check

Every kernel in this repo decodes the same EXL3 checkpoint that ExLlamaV3 reads. A change only ships if it passes
all four levels below on the GPU it targets.

| Level | Test | Pass means |
| --- | --- | --- |
| 1 | Unpacked weights against ExLlamaV3's `reconstruct` (or `trellis_core.reference`, which matches it) | Bit-identical |
| 2 | Layer output against a float64 reference | Error no worse than ExLlamaV3's own kernel |
| 3 | Captured GPU graph against eager mode, run twice | Bit-equal both times |
| 4 | Whole model: greedy answers and KL divergence against a trusted reference | Within run-to-run noise |

Where the tools are:

- Reference decoder: `core/src/trellis_core/reference.py`. CPU tests in `core/tests` pin its codebooks against an
  independent implementation.
- RTX 3090 (CUDA): `python -m sglang_exl3.parity.marlin_parity <checkpoint>` runs levels 1 and 2 against
  `exllamav3_ext.reconstruct` for every EXL3 tensor class in the checkpoint.
- Intel Arc (XPU): `xpu/tests/test_bitexact_xpu.py` (level 1 on every kernel path), `xpu/tests/gateA3/` (level 4,
  logits against ExLlamaV3 on an NVIDIA card), `xpu/tests/oracle_cuda.py` (the reference decoder against ExLlamaV3's CUDA
  kernels).

Adding a GPU means writing a kernel for it and passing these four levels. The format reader and the reference decoder
in `core/` stay the same.
