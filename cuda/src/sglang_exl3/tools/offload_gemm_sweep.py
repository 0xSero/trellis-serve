"""Launch-config sweep of the pointer-table grouped GEMMs at decode sizes (device- or host-resident experts).
Each config: one moe_gemm_ptr captured in a CUDA graph, median of reps replays (fresh routing per rep for host).
python -m sglang_exl3.tools.offload_gemm_sweep <model> [--where device] [--tokens 1,4,16]"""
from __future__ import annotations

import argparse
import statistics

import torch

from ..kernels import marlin_moe, offload_moe as om
from .offload_moe_bench import Layer, align, uniform_routing

CFGS = [(-1, -1), (128, 128), (64, 128), (128, 64), (64, 256)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", default="1,4,16")
    ap.add_argument("--where", default="device")
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(0)
    L = Layer(a.model_dir, a.layer)
    mod = om._mod()
    lay = L.lay
    L.set_table(a.where)
    for T in [int(t) for t in a.tokens.split(",")]:
        slots, I, H = T * 10, lay.inter, lay.hidden
        block = marlin_moe.moe_block_size(T, 10, L.e)
        ids, _ = uniform_routing(T, 10, L.e, gen)
        s, e, p = align(ids, block, L.e)
        xh = (torch.randn((2 * slots, H), generator=gen) * 0.3).half().cuda()
        xd = (torch.randn((slots, I), generator=gen) * 0.3).half().cuda()
        gu, yd = torch.empty((slots, 2 * I), dtype=torch.half, device="cuda"), torch.empty((slots, H), dtype=torch.half, device="cuda")
        routs = [align(uniform_routing(T, 10, L.e, gen)[0], block, L.e) for _ in range(a.reps + 5)]
        for which in ("gate_up", "down"):
            res = []
            for tk, tn in CFGS:
                for bps in (-1, 1, 2, 3, 4):
                    mod.moe_set_blocks_per_sm(bps)
                    if which == "gate_up":
                        fn = lambda: mod.moe_gemm_ptr(xh, gu, L.table, om.F_W13, om.F_SVH13, lay.bits, s, e, p, block, I, L.cb, tk, tn)
                    else:
                        fn = lambda: mod.moe_gemm_ptr(xd, yd, L.table, om.F_W2, om.F_SVH2, lay.bits, s, e, p, block, 0, L.cb, tk, tn)
                    try:
                        fn(); torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            fn()
                    except Exception as ex:   # invalid config for this shape
                        continue
                    ts = []
                    for r, (rs, re_, rp) in enumerate(routs):
                        s.copy_(rs[: s.numel()]) if rs.numel() == s.numel() else None
                        e.copy_(re_[: e.numel()]) if re_.numel() == e.numel() else None
                        p.copy_(rp)
                        torch.cuda.synchronize()
                        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        t0.record(); g.replay(); t1.record(); torch.cuda.synchronize()
                        if r >= 5:
                            ts.append(t0.elapsed_time(t1) * 1000)
                    res.append((round(statistics.median(ts), 1), tk, tn, bps))
            mod.moe_set_blocks_per_sm(-1)
            res.sort()
            default = [r for r in res if r[1] == -1 and r[3] == -1]
            print(f"sweep {a.where} T={T} {which}: default {default[0][0] if default else None} us; best {res[:5]}; worst {res[-2:]}", flush=True)


if __name__ == "__main__":
    main()
