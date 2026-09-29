"""Per-launch time breakdown of one pointer-table MoE layer at decode sizes (CUDA events around each launch, eager,
after warm-up; median of reps). python -m sglang_exl3.tools.offload_moe_breakdown <model> [--tokens 1] [--where device]"""
from __future__ import annotations

import argparse
import statistics

import torch

from ..kernels import marlin_moe, offload_moe as om
from .offload_moe_bench import Layer, align, uniform_routing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--where", default="device")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(0)
    L = Layer(a.model_dir, a.layer)
    mod = om._mod()
    T, lay = a.tokens, L.lay
    x = (torch.randn((T, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    slots, I, H = T * 10, lay.inter, lay.hidden
    f16 = dict(dtype=torch.float16, device="cuda")
    xh, gu = torch.empty((2 * slots, H), **f16), torch.empty((slots, 2 * I), **f16)
    act, xd, y = torch.empty((slots, I), **f16), torch.empty((slots, I), **f16), torch.empty_like(x)
    block = marlin_moe.moe_block_size(T, 10, L.e)
    L.set_table(a.where)
    names = ["align", "had_in", "gemm_gate_up", "glu_had_in", "gemm_down", "combine"]
    ids, w = uniform_routing(T, 10, L.e, gen)
    st = {}

    def ops(k):
        if k >= 1: st["a"] = align(ids, block, L.e)
        s, e, p = st["a"]
        if k >= 2: mod.moe_had_in_ptr(x, L.table, om.F_SUH13, 2, ids, xh)
        if k >= 3: mod.moe_gemm_ptr(xh, gu, L.table, om.F_W13, om.F_SVH13, lay.bits, s, e, p, block, I, L.cb)
        if k >= 4: mod.moe_glu_had_in_ptr(gu, L.table, om.F_SUH2, ids, act, xd)
        if k >= 5: mod.moe_gemm_ptr(xd, xh[:slots], L.table, om.F_W2, om.F_SVH2, lay.bits, s, e, p, block, 0, L.cb)
        if k >= 6: mod.moe_combine(xh[:slots], w, ids, L.e, y)

    cum = []
    routs = [uniform_routing(T, 10, L.e, gen) for _ in range(a.reps + 5)]
    for k in range(1, 7):
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            ops(6); ops(6)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ops(k)
        ts = []
        for r, (ri, rw) in enumerate(routs):
            ids.copy_(ri); w.copy_(rw); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            if r >= 5:
                ts.append(e0.elapsed_time(e1) * 1000)
        cum.append(statistics.median(ts))
    acc = {n: [cum[i] - (cum[i - 1] if i else 0.0)] for i, n in enumerate(names)}
    out = {n: round(statistics.median(v), 1) for n, v in acc.items()}
    out["graph_total"] = round(cum[-1], 1)
    print("breakdown_us", {"tokens": T, "where": a.where, **out}, flush=True)


if __name__ == "__main__":
    main()
