"""Bound the transient memory of SGLang's QSA indexer prefill on XPU.

`QSAIndexer.select_prefill_tokens` splits query rows so that the [rows, keys] fp32 logits stay under
`_QSA_PREFILL_LOGITS_BUDGET_BYTES` (128 MiB). On XPU `qsa_mqa_prefill` falls back to `torch_qsa_mqa_prefill`, whose
real peak is far larger than the logits: it materialises `scores[rows, keys, heads]` in fp32, a second tensor of the
same size for `relu`, then the summed logits, the scaled logits, the column/validity masks and the masked result.
With Flash-Next's 4 indexer heads a 2048-row chunk at a 64K context (~16K compressed keys) peaks near 1.3 GB instead of
128 MiB, which exceeded a pipeline stage's free memory serving Flash-Next on two B70s (observed: OOM on a
124 MiB logits allocation during a 64K prefill with a concurrent request).

This replaces `_qsa_prefill_row_chunk_size` with one that budgets the whole fallback: per row,
keys * 4 bytes * (2 * heads + TEMPS). The computation per row is unchanged (rows are independent), so only the
number of rows per chunk changes; whether results are bit-identical on XPU (matmul algorithm selection can depend on
the row count) is checked by tests/test_qsa_prefill_mem.py, not assumed.

Env: EXL3_QSA_PREFILL_BUDGET_BYTES (default 128 MiB, the upstream constant, now applied to the true peak);
     EXL3_QSA_PREFILL_BUDGET=0 disables the patch.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_BUDGET = int(os.environ.get("EXL3_QSA_PREFILL_BUDGET_BYTES", str(128 << 20)))
if _BUDGET <= 0:
    raise ValueError("EXL3_QSA_PREFILL_BUDGET_BYTES must be positive")
# Logits-sized fp32 allowance besides the two [rows, keys, heads] tensors (scores and relu output). Not an exact peak
# model: the summed, scaled and masked logits do not all coexist, and the bool masks are 1 byte each, so 4 is a
# conservative allowance. Not covered: the fp32 copies of q (rows x heads x 128) and k (keys x 128), backend
# workspace, and tensors held by the caller; at the minimum chunk (block_q rows) the budget may be exceeded.
_TEMPS = 4


def row_chunk_size(rows: int, keys: int, heads: int) -> int:
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    block_q = max(1, 128 // heads)                       # same padding granularity as upstream
    bytes_per_row = keys * torch.float32.itemsize * (2 * heads + _TEMPS)   # heads = runtime q.shape[1] (padded)
    max_rows = max(block_q, _BUDGET // bytes_per_row)
    max_rows = max(block_q, max_rows // block_q * block_q)
    return min(rows, max_rows)


def install() -> bool:
    """Patch only where the torch fallback runs (XPU without CUDA/TileLang)."""
    if os.environ.get("EXL3_QSA_PREFILL_BUDGET", "1") == "0":
        return False
    if torch.cuda.is_available() or not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    from sglang.srt.layers.attention.qsa import qsa_indexer as qi
    if not hasattr(qi, "_qsa_prefill_row_chunk_size"):
        logger.warning("exl3xpu: this SGLang has no _qsa_prefill_row_chunk_size; row-chunk budget not installed")
        return False
    if getattr(qi._qsa_prefill_row_chunk_size, "_exl3", False):
        return True
    row_chunk_size._exl3 = True
    qi._qsa_prefill_row_chunk_size = row_chunk_size
    logger.info("exl3xpu: QSA prefill row chunks budget the full torch fallback peak (%d MiB)", _BUDGET >> 20)
    return True
