"""Graph-safe record copy (csrc copy_records) host bank -> device slots: correctness + GB/s sweep vs DMA.
python -m sglang_exl3.tools.copy_records_bench <model> [--layer 3]"""
from __future__ import annotations

import argparse
import statistics

import torch

from ..kernels import offload_moe as om
from .offload_moe_bench import Layer


def timed(fn, reps):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(0)
    L = Layer(a.model_dir, a.layer)
    mod = om._mod()
    rec = L.lay.record_bytes
    arena = torch.zeros((L.e, rec), dtype=torch.uint8, device="cuda")
    ar = torch.arange(L.e, dtype=torch.int64, device="cuda")
    # correctness: a random permutation of all records
    perm = torch.randperm(L.e, generator=gen).cuda()
    mod.copy_records(L.host_base + perm * rec, arena.data_ptr() + ar * rec, rec)
    torch.cuda.synchronize()
    ok = torch.equal(arena.cpu(), L.host[perm.cpu()])
    # count-limited: only the first 7 entries
    arena.zero_(); cnt = torch.tensor([7], dtype=torch.int32, device="cuda")
    mod.copy_records(L.host_base + perm * rec, arena.data_ptr() + ar * rec, rec, count=cnt)
    torch.cuda.synchronize()
    ok_cnt = torch.equal(arena[:7].cpu(), L.host[perm[:7].cpu()]) and bool((arena[7:] == 0).all())
    print("copy_records correctness", ok, "count-limited", ok_cnt, flush=True)
    for n in (10, 40, 512):
        nb = n * rec
        row = []
        for kb in (16, 64, 256, 1024):
            mod.set_copy_chunk_kb(kb)
            for unroll in (1, 2, 4, 8):
                src = L.host_base + torch.randperm(L.e, generator=gen)[:n].cuda() * rec
                dst = arena.data_ptr() + ar[:n] * rec
                us = timed(lambda: mod.copy_records(src, dst, rec, unroll=unroll), a.reps)
                row.append((round(nb / us / 1e3, 2), kb, unroll, round(us, 1)))
        best = max(row)
        idx = torch.randperm(L.e, generator=gen)[:n].tolist()
        dma = timed(lambda: [arena[j].copy_(L.host[e], non_blocking=True) for j, e in enumerate(idx)], max(5, a.reps // 3))
        print(f"n={n} records ({nb/1e6:.1f} MB): best copy_records {best[0]} GB/s (chunk {best[1]} KB, unroll {best[2]}, "
              f"{best[3]} us); DMA per-record loop {nb/dma/1e3:.2f} GB/s ({dma:.0f} us); sweep {sorted(row, reverse=True)[:6]}", flush=True)
    # overlap: prefetch copy on a side stream while the device-resident decode layer (graph) replays on the main stream
    from .offload_moe_bench import uniform_routing
    L.set_table("device")
    x = (torch.randn((1, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    ids, w = uniform_routing(1, 10, L.e, gen)
    side0 = torch.cuda.Stream(); side0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side0):
        L.run_ptr(x, ids, w); L.run_ptr(x, ids, w)
    torch.cuda.current_stream().wait_stream(side0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        L.run_ptr(x, ids, w)
    side = torch.cuda.Stream()
    E = lambda: torch.cuda.Event(enable_timing=True)
    # persistent small-footprint copy: correctness + GB/s by block count
    arena.zero_()
    mod.copy_records_persistent(L.host_base + perm * rec, arena.data_ptr() + ar * rec, rec, blocks=16)
    torch.cuda.synchronize()
    print("persistent correctness", torch.equal(arena.cpu(), L.host[perm.cpu()]), flush=True)
    for nb in (1, 2, 4, 8, 16, 32, 82):
        src = L.host_base + torch.randperm(L.e, generator=gen)[:10].cuda() * rec
        dst = arena.data_ptr() + ar[:10] * rec
        us = timed(lambda: mod.copy_records_persistent(src, dst, rec, blocks=nb), a.reps)
        print(f"persistent blocks={nb}: 10 records {us:.1f} us = {10 * rec / us / 1e3:.2f} GB/s", flush=True)
    import os
    configs = ((1024, "copy_records", 0, 0), (1024, "dma", 0, 0), (0, "persistent", 4, 0), (0, "persistent", 1, 0),
               (0, "persistent", 1, 162), (0, "persistent", 2, 160), (0, "persistent", 1, 160))
    if os.environ.get("COPY_OVERLAP_QUICK"):
        configs = configs[3:]
    for kb, method, pb, glim in configs:
        if kb:
            mod.set_copy_chunk_kb(kb)
        mod.moe_set_grid_limit(glim)
        side0 = torch.cuda.Stream(); side0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side0):
            L.run_ptr(x, ids, w)
        torch.cuda.current_stream().wait_stream(side0)
        g = torch.cuda.CUDAGraph()        # re-capture: the grid size is baked into the graph
        with torch.cuda.graph(g):
            L.run_ptr(x, ids, w)
        if True:
            n = 10
            idx = torch.randperm(L.e, generator=gen)[:n]
            src = L.host_base + idx.cuda() * rec
            dst = arena.data_ptr() + ar[:n] * rec
            if method == "copy_records":
                copy = lambda: mod.copy_records(src, dst, rec)
            elif method == "persistent":
                copy = lambda: mod.copy_records_persistent(src, dst, rec, blocks=pb)
            else:
                copy = lambda: [arena[j].copy_(L.host[int(e)], non_blocking=True) for j, e in enumerate(idx)]
            layers = 12
            res = {"alone_layers_us": [], "alone_copy_us": [], "both_layers_us": [], "both_copy_us": []}
            for _ in range(15):
                a, b = E(), E(); a.record()
                for _ in range(layers): g.replay()
                b.record(); torch.cuda.synchronize(); res["alone_layers_us"].append(a.elapsed_time(b) * 1000)
                with torch.cuda.stream(side):
                    a, b = E(), E(); a.record(side); copy(); b.record(side)
                torch.cuda.synchronize(); res["alone_copy_us"].append(a.elapsed_time(b) * 1000)
                st = E(); st.record(); side.wait_event(st)
                with torch.cuda.stream(side):
                    ca, cb = E(), E(); ca.record(side); copy(); cb.record(side)
                la, lb = E(), E(); la.record()
                for _ in range(layers): g.replay()
                lb.record(); torch.cuda.synchronize()
                res["both_layers_us"].append(la.elapsed_time(lb) * 1000); res["both_copy_us"].append(ca.elapsed_time(cb) * 1000)
            print("overlap", method, f"chunk {kb} KB blocks {pb} marlin_grid_limit {glim}", {k: round(statistics.median(v), 1) for k, v in res.items()},
                  f"({layers} device decode layers vs {n}-record prefetch)", flush=True)
    mod.set_copy_chunk_kb(64)
    mod.moe_set_grid_limit(0)


if __name__ == "__main__":
    main()
