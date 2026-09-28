# sglang-exl3

EXL3 (ExLlamaV3 trellis quantization) inside **stock SGLang**, Ampere (RTX 3090, sm_86) first. No SGLang source edits:
the package registers `--quantization exl3` through SGLang's `sglang.srt.plugins` entry point.

## What it does

- Loads any EXL3 checkpoint (`quantization_config.quant_method == "exl3"`, per-tensor `trellis` / `suh` / `svh` / `mul1|mcg`)
  into SGLang's fused linears with per-shard storage (q/k/v, gate/up and the GDN qkv/z groups keep their own input scales).
- Quantized `lm_head` (K=6 head), quantized MTP head incl. `mtp.fc`, vision tower untouched (bf16), embeddings shared with
  the MTP draft. Works with `--speculative-algorithm NEXTN`, `--kv-cache-dtype fp8_e4m3`, `--enable-multimodal`, the
  `qwen3` reasoning parser and the `qwen3_coder` tool parser.
- Kernels: ExLlamaV3's own `exllamav3_ext` kernels (bit-faithful reference, `SGLANG_EXL3_KERNEL=exllamav3`) and the
  Marlin-template EXL3 kernels in `csrc/` (`SGLANG_EXL3_KERNEL=auto|marlin`; decoded weights bit-identical
  to `exllamav3_ext.reconstruct`, flat cost from 1 to 16 rows). Prefill (>= 144 rows) reconstructs fp16 weight slices and
  runs cuBLAS (bounded transients, `SGLANG_EXL3_DENSE_SLICE_MB`).
- 24 GB layout: only the repacked trellis is resident (`SGLANG_EXL3_KEEP_TRELLIS=0`), the K=6 head stays 6 bits.

## Build / run (docker)

```bash
docker build -f cuda/docker/Dockerfile.dev -t sglang-exl3:dev .     # SGLang v0.5.20 + exllamav3 1.5.1 (sm_86) + this package
docker run --rm --gpus device=0 -v $PWD:/opt/trellis-serve -w /opt/trellis-serve/cuda/csrc --entrypoint bash sglang-exl3:dev -c ./build.sh
cuda/scripts/serve_3090.sh Qwen3.8-27B-EXL3-3.0bpw run1 --context-length 16384 --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --disable-prefill-cuda-graph --mem-fraction-static 0.86 \
  --max-mamba-cache-size 10 --mamba-ssm-dtype bfloat16 --chunked-prefill-size 1024
```

Environment knobs: `SGLANG_EXL3_KERNEL` (auto | exllamav3 | marlin), `SGLANG_EXL3_MARLIN_MAX_N` (65536: wider layers stay on
ExLlamaV3's kernel), `SGLANG_EXL3_DENSE_ROWS` (144), `SGLANG_EXL3_DENSE_SLICE_MB` (64), `SGLANG_EXL3_SLICED` (1),
`SGLANG_EXL3_DRAFT_SHARE_EMBED` (1), `SGLANG_EXL3_MODEL_PATH` (override the checkpoint dir for the header scan).

## Measured (one RTX 3090)

Accepted recipes on the published image `ghcr.io/0xsero/sglang-exl3:v0.7.0-ampere`: Qwen3.8-27B EXL3 3.0 bpw with its MTP draft
writes 96.2 tok/s on prose and 141.1 on code for one user; Qwen3.6-35B-A3B EXL3 3.0 bpw writes 251.7 and 357.9. The same
engine with AWQ-INT4 and no speculative decoding writes 45 tok/s on the 27B. Launch arguments are in the recipes linked
from the top-level README.

## Layout

`src/sglang_exl3/{kernels,runtime}` (engine-independent; the checkpoint format reader is `trellis_core.format` in `../core`), `sglang_glue/` (the only SGLang imports),
`plugin.py` (entry point), `csrc/` (Marlin-template EXL3 kernels + MoE kernels), `parity/`, `tools/`, `docker/`, `scripts/`.
