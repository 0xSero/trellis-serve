# trellis-serve

EXL3, the trellis quantization format from turboderp's [ExLlamaV3](https://github.com/turboderp-org/exllamav3), served
by stock inference engines on GPUs that ExLlamaV3's own kernels don't cover well. Every kernel here decodes weights
bit-for-bit the same as ExLlamaV3 (see [the lossless check](docs/lossless-check.md)).

| Directory | Package | Engine | GPU | State |
| --- | --- | --- | --- | --- |
| [`core/`](core) | `trellis_core` | any | any | Checkpoint format reader and bit-exact PyTorch reference decoder |
| [`cuda/`](cuda) | `sglang_exl3` | SGLang 0.5.20 (plugin, no source edits) | RTX 3090 (sm_86) | Validated recipes in local-ai-registry |
| [`xpu/`](xpu) | `exl3xpu` | vLLM XPU 0.26, SGLang 0.5.20 XPU (plugins) | Intel Arc Pro B70 | vLLM validated; SGLang candidate |

Both plugins load EXL3 checkpoints as published on Hugging Face, for example
[turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3). Nothing is re-quantized.

## Quick start

### RTX 3090 (SGLang)

The accepted image is public:

```bash
docker pull ghcr.io/0xsero/sglang-exl3:v0.7.0-ampere@sha256:c8922bd7256caf1b273af7ac49221e652be9e6072c1cdec4b959f506fd415d79
```

The launch arguments that were tested are in the registry recipes:
[Qwen3.8-27B](https://github.com/0xSero/local-ai-registry/blob/main/data/registry/recipe/qwen38-27b-exl3-3bpw-mtp-vision-rtx3090-sglang-tp1.json)
and [Qwen3.6-35B-A3B](https://github.com/0xSero/local-ai-registry/blob/main/data/registry/recipe/qwen36-35b-a3b-exl3-3bpw-mtp-vision-rtx3090-sglang-tp1.json).
To build from source, see [`cuda/README.md`](cuda/README.md).

### Intel Arc Pro B70 (vLLM or SGLang)

```bash
IMG=ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8
```

The run command, flags and GPU selection are in [`xpu/README.md`](xpu/README.md), and the tested recipe is
[in the registry](https://github.com/0xSero/local-ai-registry/blob/main/data/registry/recipe/qwen38-27b-exl3-4bpw-arcb70-vllm-exl3xpu-tp1.json).

### Other GPUs

- Most NVIDIA cards: ExLlamaV3 itself, served by [TabbyAPI](https://github.com/theroyallab/tabbyAPI).
- DGX Spark, RTX 5090, RTX PRO 6000: [b12x](https://github.com/local-inference-lab/b12x) by Local Inference Lab.

## Measured

One user, single request, from the recipes and package READMEs:

| GPU | Model | Prose | Code |
| --- | --- | --- | --- |
| RTX 3090 | Qwen3.8-27B EXL3 3.0 bpw, MTP draft | 96.2 tok/s | 141.1 tok/s |
| RTX 3090 | Qwen3.6-35B-A3B EXL3 3.0 bpw, MTP draft | 251.7 tok/s | 357.9 tok/s |
| Arc Pro B70 | Qwen3.8-27B EXL3 4.0 bpw, MTP draft, thinking on | 91.2 tok/s | 66.1 tok/s |

On the B70 that is 3.6 times llama.cpp's Q4_K_M at one user and 6.5 times at sixteen (365.2 against 56.0 tok/s total).

## Layout

```
core/    trellis_core: format reader (no torch) + reference decoder (torch); CPU tests
cuda/    sglang_exl3: SGLang plugin, Marlin-style EXL3 kernels for sm_86 (cuda/csrc), dev Dockerfile, serve scripts
xpu/     exl3xpu: vLLM and SGLang plugins, ESIMD kernels for Intel Xe2 (xpu/csrc), Dockerfile, tests, benchmarks
docs/    the lossless check
```

Docker builds use the repository root as the build context, so both images get `core/`:

```bash
docker build -f cuda/docker/Dockerfile.dev -t sglang-exl3:dev .
docker build -f xpu/docker/Dockerfile -t exl3xpu .
```

CPU tests: `cd core && python -m pytest`.

## Credits

The EXL3 format and its maths are turboderp's ([ExLlamaV3](https://github.com/turboderp-org/exllamav3), MIT). The CUDA
kernel skeleton comes from vLLM's [Marlin](https://github.com/IST-DASLab/marlin) kernels (Apache-2.0). The trellis idea
is from [QTIP](https://arxiv.org/abs/2406.11235). See [NOTICE](NOTICE).

MIT licensed.
