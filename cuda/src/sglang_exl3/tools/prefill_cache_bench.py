"""Prefill of one real layer with part of its experts resident in the ExpertCache: only the missing experts are
copied into a staging buffer (record layout), the table points resident experts at their slots (no D2D gather).
    python -m sglang_exl3.tools.prefill_cache_bench <model> [--resident 0.38] [--tokens 2048,8192]
Measures (median of reps): GEMM alone; copy of the misses alone (DMA whole layer, copy_records full grid,
copy_records_persistent 1 block); copy overlapped with a concurrent GEMM of the same size (= the previous layer).
Correctness: prefill output == stacked path."""
from __future__ import annotations

import argparse
import statistics

import torch

from ..kernels import marlin_moe, offload_moe as om
from .offload_moe_bench import Layer, align, eq16, uniform_routing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--resident", type=float, default=0.38)
    ap.add_argument("--tokens", default="2048,8192")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(0)
    L = Layer(a.model_dir, a.layer)
    mod, E, rec = om._mod(), L.e, L.lay.record_bytes
    nres = int(round(a.resident * E))
    cache = om.ExpertCache(1, E, L.lay, [L.host_base], nres)
    cache.preload(0, torch.randperm(E, generator=gen)[:nres].tolist())
    staging = torch.empty((E, rec), dtype=torch.uint8, device="cuda")
    table = om.new_table(E)
    src, dst = torch.empty(E, dtype=torch.int64, device="cuda"), torch.empty(E, dtype=torch.int64, device="cuda")
    cache.prefill_plan(0, staging.data_ptr(), table, src, dst)
    miss_bytes = int((src != 0).sum()) * rec
    E_ = lambda: torch.cuda.Event(enable_timing=True)
    copies = {
        "dma_whole_layer": lambda: staging.copy_(L.host, non_blocking=True),
        "copy_records_misses": lambda: mod.copy_records(src, dst, rec),
        "persistent1_misses": lambda: mod.copy_records_persistent(src, dst, rec, blocks=1),
    }
    miss_idx = (src != 0).nonzero().flatten().tolist()                      # host-known miss list (prefill snapshot)
    copies["dma_misses"] = lambda: [staging[e].copy_(L.host[e], non_blocking=True) for e in miss_idx]
    for T in [int(t) for t in a.tokens.split(",")]:
        ids, w = uniform_routing(T, 10, E, gen)
        x = (torch.randn((T, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
        block = marlin_moe.moe_block_size(T, 10, E)
        al = align(ids, block, E)
        copies["copy_records_misses"](); torch.cuda.synchronize()
        y = om.run(x, w, ids, *al, block, table, L.lay, L.cb)
        ok = eq16(y, marlin_moe.run(x, w, ids, *al, block, L.pack))   # same align output (see REPORT E10)
        gemm = lambda: om.run(x, w, ids, *al, block, table, L.lay, L.cb)
        ts = []
        for _ in range(a.reps):
            s0, s1 = E_(), E_(); s0.record(); gemm(); s1.record(); torch.cuda.synchronize(); ts.append(s0.elapsed_time(s1))
        out = {"tokens": T, "eq_stacked": ok, "gemm_ms": round(statistics.median(ts), 2),
               "resident_frac": round(nres / E, 3), "miss_MB": round(miss_bytes / 1e6, 1)}
        side = torch.cuda.Stream()
        for name, cp in copies.items():
            alone, wall, cms, gms = [], [], [], []
            for _ in range(a.reps):
                with torch.cuda.stream(side):
                    c0, c1 = E_(), E_(); c0.record(side); cp(); c1.record(side)
                torch.cuda.synchronize(); alone.append(c0.elapsed_time(c1))
                st = E_(); st.record(); side.wait_event(st)
                with torch.cuda.stream(side):
                    c0, c1 = E_(), E_(); c0.record(side); cp(); c1.record(side)
                g0, g1 = E_(), E_(); g0.record(); gemm(); g1.record()
                en = E_(); torch.cuda.current_stream().wait_event(c1); en.record(); torch.cuda.synchronize()
                wall.append(st.elapsed_time(en)); cms.append(c0.elapsed_time(c1)); gms.append(g0.elapsed_time(g1))
            m = statistics.median
            out[name] = {"copy_alone_ms": round(m(alone), 2), "overlap_wall_ms": round(m(wall), 2),
                         "overlap_copy_ms": round(m(cms), 2), "overlap_gemm_ms": round(m(gms), 2)}
        print("prefill_cache", out, flush=True)


if __name__ == "__main__":
    main()
