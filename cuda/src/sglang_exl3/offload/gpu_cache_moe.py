"""GPU expert-cache offload for EXL3 routed experts (kernel lane K05): the drop-in quant method for SGLang's FusedMoE.

    from sglang_exl3.offload.gpu_cache_moe import Exl3OffloadMoEMethod, is_offloaded_expert_key, offload_stats

Implementation: sglang_glue/offload_moe_method.py (method), kernels/offload_runtime.py (decode + staged prefill),
kernels/offload_store.py (pinned host store), kernels/offload_moe.py (pointer-table kernels, ExpertCache).
Enable in Exl3Config.get_quant_method with SGLANG_EXL3_MOE_OFFLOAD=gpu_cache (or 1). See REPORT.md "K05 API".
"""
from ..sglang_glue.offload_moe_method import (  # noqa: F401
    Exl3OffloadMoEMethod, get_runtime, is_offloaded_expert_key, offload_enabled, offload_stats)
