"""Offload tiers for models whose routed experts / n-gram tables do not fit the card (Qwen3.8-Flash-Next on a 24 GB
RTX 3090): routed experts on exllamav3's persistent CPU worker (cpu_moe), the EXL3 n-gram table in pinned host
memory (ngram_host). Kernels: csrc/offload/offload_kernels.cu, JIT-built on first use (_ext.load)."""
