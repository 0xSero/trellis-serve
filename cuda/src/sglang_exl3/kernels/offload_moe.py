"""Offload MoE: the grouped EXL3 Marlin MoE kernels with experts addressed through a per-expert POINTER TABLE.

An expert can live anywhere the GPU can dereference: a slot of a device cache arena, a device staging buffer (prefill
streaming), or pinned + mapped host memory (zero-copy over PCIe, UVA address). The runtime owns the table and rewrites
it in place; the kernels read it on the device, so a captured CUDA graph stays valid while experts move.

Expert RECORD (one contiguous blob per expert, identical in every tier, so moving an expert is one memcpy):

    field 0 w13    int32 [H/16, 2I/64, 4, 24] (K=3) | [H/16, 2I/64, 32, 4] (K=4)   gate | up along n (pack layout)
    field 1 w2     int32 [I/16, H/64, 4, 24]  (K=3) | [I/16, H/64, 32, 4]  (K=4)
    field 2 suh13  fp16  [2, H]     gate suh, up suh
    field 3 svh13  fp16  [2I]       gate svh | up svh
    field 4 suh2   fp16  [I]
    field 5 svh2   fp16  [H]

Every field starts on a 256-byte boundary (record_bytes is a multiple of 256). For Qwen3.8-Flash-Next (H 2560, I 640,
K=3): 1,228,800 + 614,400 + 10,240 + 2,560 + 1,280 + 5,120 = 1,862,400 B per expert (the checkpoint's 1,862,412 B
minus the three 4-byte mul1 markers), 953,548,800 B per layer.

POINTER TABLE: int64 CUDA tensor [E, 6], row e = the six field addresses of LOGICAL expert e (router id). Build it from
per-expert record base addresses with `fill_table_` (device op, graph-capturable), e.g. from a slot map:
    base[e] = arena_base + slot[e] * record_bytes    if slot[e] >= 0   (GPU cache / staging slot)
            = host_base  + e       * record_bytes    otherwise         (zero-copy from the host bank)
Routing is unchanged: moe_align_block_size(topk_ids, block, E) with logical ids; the table does the mapping.

One MoE layer = moe_had_in_ptr -> moe_gemm_ptr (gate+up, svh in launch) -> moe_glu_had_in_ptr -> moe_gemm_ptr (down)
-> moe_combine: the same five launches and the same arithmetic as marlin_moe.run (bit-identical outputs).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import marlin_moe

FIELDS = ("w13", "w2", "suh13", "svh13", "suh2", "svh2")
F_W13, F_W2, F_SUH13, F_SVH13, F_SUH2, F_SVH2 = range(6)
ALIGN = 256


def _mod():
    return marlin_moe._load()


@dataclass(frozen=True)
class RecordLayout:
    hidden: int
    inter: int
    bits: int
    sizes: tuple
    offsets: tuple
    record_bytes: int

    def offsets_tensor(self, device) -> torch.Tensor:
        return torch.tensor(self.offsets, dtype=torch.int64, device=device)


def layout(hidden: int, inter: int, bits: int) -> RecordLayout:
    if bits not in (3, 4) or hidden % 128 or inter % 128:
        raise ValueError("K must be 3 or 4 and hidden / inter multiples of 128")
    w = lambda k, n: k * n * bits // 8
    sizes = (w(hidden, 2 * inter), w(inter, hidden), 2 * hidden * 2, 2 * inter * 2, inter * 2, hidden * 2)
    offs, o = [], 0
    for s in sizes:
        offs.append(o)
        o += (s + ALIGN - 1) // ALIGN * ALIGN
    return RecordLayout(hidden, inter, bits, sizes, tuple(offs), o)


def expert_record(gate, up, down, e: int, lay: RecordLayout, out: torch.Tensor | None = None) -> torch.Tensor:
    """gate / up / down = (trellis list, suh list, svh list) by expert (as marlin_moe.prepare). -> uint8 [record_bytes]
    on the tensors' device (or written into `out`, any device). Pure re-layout of expert e."""
    w13 = torch.cat([marlin_moe.stack_repacked([gate[0][e]]), marlin_moe.stack_repacked([up[0][e]])], dim=2)
    w2 = marlin_moe.stack_repacked([down[0][e]])
    if marlin_moe.bits_of_stack(w13) != lay.bits:
        raise ValueError("expert K does not match the layout")
    parts = (w13, w2, torch.stack([gate[1][e], up[1][e]]), torch.cat([gate[2][e], up[2][e]]), down[1][e], down[2][e])
    rec = torch.zeros((lay.record_bytes,), dtype=torch.uint8, device=w13.device) if out is None else out
    for p, off, size in zip(parts, lay.offsets, lay.sizes):
        b = p.contiguous().view(-1).view(torch.uint8)
        if b.numel() != size:
            raise ValueError(f"field size {b.numel()} != {size}")
        rec[off:off + size].copy_(b)
    return rec


def build_bank(gate, up, down, lay: RecordLayout, where: str = "host", device=None) -> torch.Tensor:
    """All experts of a layer as uint8 [E, record_bytes]: where='host' -> pinned + mapped host memory (exact-size
    cudaHostAlloc), 'device' -> a CUDA tensor (a cache arena / staging buffer holding every expert)."""
    e = len(gate[0])
    if where == "host":
        bank = _mod().host_alloc_mapped(e * lay.record_bytes).view(e, lay.record_bytes)
    else:
        bank = torch.empty((e, lay.record_bytes), dtype=torch.uint8, device=device or "cuda")
    for i in range(e):
        bank[i].copy_(expert_record(gate, up, down, i, lay))
    return bank


def base_address(t: torch.Tensor) -> int:
    """Kernel-visible address of a tensor: device pointer, or the UVA address of pinned + mapped host memory."""
    return t.data_ptr() if t.is_cuda else int(_mod().host_device_ptr(t))


def new_table(num_experts: int, device=None) -> torch.Tensor:
    return torch.zeros((num_experts, len(FIELDS)), dtype=torch.int64, device=device or "cuda")


def fill_table_(table: torch.Tensor, bases: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """table[e, f] = bases[e] + offsets[f], in place (device op, CUDA-graph capturable)."""
    torch.add(bases.unsqueeze(1), offsets.unsqueeze(0), out=table)
    return table


def slot_bases(slot_of_expert: torch.Tensor, arena_base: int, host_base: int, record_bytes: int) -> torch.Tensor:
    """Per-expert record base from a slot map (int32/int64 [E], -1 = not resident -> host bank row e)."""
    s = slot_of_expert.to(torch.int64)
    e = torch.arange(s.numel(), dtype=torch.int64, device=s.device)
    return torch.where(s >= 0, arena_base + s * record_bytes, host_base + e * record_bytes)


def run(x, topk_weights, topk_ids, sorted_ids, expert_ids, num_post_padded, block: int, table: torch.Tensor,
        lay: RecordLayout, codebook: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """x fp16 | bf16 [T, H]; topk_weights fp32 [T, top_k]; topk_ids int32 | int64 [T, top_k] (logical ids);
    sorted_ids / expert_ids / num_post_padded = moe_align_block_size(topk_ids, block, E); table int64 [E, 6]."""
    mod = _mod()
    tokens, hidden = x.shape
    top_k, inter = topk_ids.shape[1], lay.inter
    slots = tokens * top_k
    y = out if out is not None else torch.empty_like(x)
    if tokens == 0:
        return y
    f16 = dict(dtype=torch.float16, device=x.device)
    xh = torch.empty((2 * slots, hidden), **f16)
    mod.moe_had_in_ptr(x, table, F_SUH13, 2, topk_ids, xh)
    gu = torch.empty((slots, 2 * inter), **f16)
    cb = codebook
    mod.moe_gemm_ptr(xh, gu, table, F_W13, F_SVH13, lay.bits, sorted_ids, expert_ids, num_post_padded, block, inter, cb)
    act, xd = torch.empty((slots, inter), **f16), torch.empty((slots, inter), **f16)
    mod.moe_glu_had_in_ptr(gu, table, F_SUH2, topk_ids, act, xd)
    yd = xh[:slots]
    mod.moe_gemm_ptr(xd, yd, table, F_W2, F_SVH2, lay.bits, sorted_ids, expert_ids, num_post_padded, block, 0, cb)
    mod.moe_combine(yd, topk_weights, topk_ids, table.shape[0], y)
    return y
