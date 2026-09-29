"""K04: hybrid miss split for ONE real MoE layer: part of the routed experts computed on the GPU (device hits + zero-copy
misses, sglang_exl3 offload_moe) and the rest concurrently on the CPU by exllamav3's MoeCpuHost worker (job ring,
submit_issue / submit_collect, -mcl 48), plus the contention between PCIe zero-copy reads and CPU DRAM bandwidth.

Run in sglang-exl3:dev on GPU 0 with the trellis-serve working copy on PYTHONPATH (see run_k04.sh).
Per configuration (1 token, 10 routed experts, fresh random routing per rep, median of reps, CUDA events on the stream):
  cpu_only(n)         : n experts on the CPU worker (issue -> collect), the others dropped
  gpu_zc(n)           : n experts zero-copy on the GPU (host table), the others dropped (sentinel E)
  split(h, F, C)      : h device hits + F zero-copy misses on the GPU while C misses run on the CPU; wall = issue ->
                        GPU work -> collect -> add
Correctness: split output (h, F, C) vs float64 of the same decoded weights over all 10 experts (exllamav3's CPU kernels
use their own arithmetic: tolerance, not bit-exactness).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

import torch

sys.path.insert(0, "/opt/exllamav3-src")
from exllamav3 import model_init  # noqa: E402

from sglang_exl3.kernels import marlin_moe, offload_moe as om  # noqa: E402
from sglang_exl3.kernels import reference  # noqa: E402
from sglang_exl3.tools.moe_parity_sm86 import exact_layer  # noqa: E402
from sglang_exl3.tools.offload_moe_bench import Layer, align  # noqa: E402

_PROJ = ("gate_proj", "up_proj", "down_proj")


def find_mlp(model, layer):
    seen, stack = set(), [model]
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        if type(m).__name__ == "BlockSparseMLP" and re.search(rf"layers\.{layer}\.", m.key) and not m.key.startswith("mtp"):
            return m
        stack.extend(getattr(m, "modules", []) or [])
    raise RuntimeError("layer not found")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    model_init.add_args(ap, cache=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    model, config, cache, tok = model_init.init(a)
    mlp = find_mlp(model, a.layer)
    host, lidx = mlp.cpu_host, mlp.cpu_layer_idx
    print("exllamav3 layer", mlp.key, "cpu_layer_idx", lidx, flush=True)
    L = Layer(a.model_dir, a.layer)
    E, H = L.e, L.hidden
    gen = torch.Generator().manual_seed(0)
    dev = torch.device("cuda")
    ev = lambda: torch.cuda.Event(enable_timing=True)

    # hits live in a device copy (L.dev), misses in the host bank: per-rep table = device rows for the h "hit" experts
    def table_for(hit_ids):
        mask = torch.ones(E, dtype=torch.bool, device=dev)
        mask[hit_ids] = False
        L.set_table("mixed", mask)                 # host where mask

    def one(h, F, C, x, ids, w):
        """ids [1, 10]: first h = hits, next F = GPU zero-copy misses, next C = CPU. -> (y fp32 [1, H], wall ms)."""
        gpu_ids = ids.clone(); gpu_ids[0, h + F:] = E             # dropped on the GPU
        cpu_sel = ids.to(torch.long).clone(); cpu_sel[0, : h + F] = -1
        cpu_sel[0, h + F + C:] = -1
        table_for(ids[0, :h].long())
        block = marlin_moe.moe_block_size(1, 10, E)
        al = align(gpu_ids, block, E)
        torch.cuda.synchronize()
        e0, e1 = ev(), ev()
        e0.record()
        hd = host.submit_issue(lidx, x.half(), cpu_sel, w.half()) if C > 0 else None
        y = om.run(x, w, gpu_ids, *al, block, L.table, L.lay, L.cb).float() if h + F > 0 else torch.zeros((1, H), device=dev)
        if hd is not None:
            y = y + host.submit_collect(hd)
        e1.record()
        torch.cuda.synchronize()
        return y, e0.elapsed_time(e1)

    def rand_route():
        ids = torch.randperm(E, generator=gen)[:10].view(1, 10).to(torch.int32).to(dev)
        w = torch.softmax(torch.randn((1, 10), generator=gen), -1).to(dev)
        x = (torch.randn((1, H), generator=gen) * 0.5).half().to(dev)
        return x, ids, w

    def med(h, F, C, reps):
        ts = []
        for r in range(reps + 3):
            x, ids, w = rand_route()
            _, t = one(h, F, C, x, ids, w)
            if r >= 3:
                ts.append(t * 1000)
        return round(statistics.median(ts), 1), round(sorted(ts)[len(ts) // 10], 1), round(sorted(ts)[len(ts) * 9 // 10], 1)

    out = {"layer": a.layer, "threads": a.moe_cpu_threads, "reps": a.reps, "rows": []}
    # correctness
    t_gpu = {p: tuple([t.cuda() for t in g] for g in L.t_cpu[p]) for p in _PROJ}
    ref13 = torch.stack([torch.cat([reference.reconstruct(g, L.cb), reference.reconstruct(u, L.cb)], dim=1)
                         for g, u in zip(t_gpu["gate_proj"][0], t_gpu["up_proj"][0])])
    ref2 = torch.stack([reference.reconstruct(d, L.cb) for d in t_gpu["down_proj"][0]])
    for (h, F, C) in ((10, 0, 0), (0, 10, 0), (0, 0, 10), (4, 3, 3), (0, 5, 5)):
        x, ids, w = rand_route()
        y, _ = one(h, F, C, x, ids, w)
        ex = exact_layer(x, ids, w, t_gpu, ref13, ref2, L.inter)
        err = float((y.double() - ex).abs().mean() / ex.abs().mean())
        out.setdefault("correctness", []).append({"h": h, "F": F, "C": C, "err64": err})
        print("correct", h, F, C, "err64", round(err, 5), flush=True)
    del t_gpu, ref13, ref2
    torch.cuda.empty_cache()
    # timing grid: all misses (h = 0), and 50% / 80% hit cases
    grid = [(0, 0, n) for n in (1, 2, 3, 5, 10)] + [(0, n, 0) for n in (1, 2, 3, 5, 10)] + [(10, 0, 0)]
    grid += [(0, f, 10 - f) for f in range(1, 10)]
    grid += [(5, f, 5 - f) for f in range(0, 6)] + [(8, f, 2 - f) for f in range(0, 3)]
    for h, F, C in grid:
        m, p10, p90 = med(h, F, C, a.reps)
        row = {"h": h, "F": F, "C": C, "median_us": m, "p10_us": p10, "p90_us": p90,
               "model_us": max(5.5 * h + 73 * F, 65 * C)}
        out["rows"].append(row)
        print("split", json.dumps(row), flush=True)
        json.dump(out, open(a.json, "w"), indent=1)
    # contention: CPU worker on all 10 experts of one layer, repeated, while the GPU streams the whole layer bank
    # zero-copy (copy_records_persistent of 512 records from the host bank, 953 MB, ~36 ms), and vice versa
    mod = om._mod()
    stage = torch.empty_like(L.dev)
    ar = torch.arange(E, dtype=torch.int64, device=dev)
    src, dst = L.host_base + ar * L.lay.record_bytes, stage.data_ptr() + ar * L.lay.record_bytes
    side = torch.cuda.Stream()
    for blocks in (1, 82):
        # PCIe alone
        torch.cuda.synchronize(); e0, e1 = ev(), ev(); e0.record()
        mod.copy_records_persistent(src, dst, L.lay.record_bytes, blocks=blocks); e1.record(); torch.cuda.synchronize()
        pcie_alone = e0.elapsed_time(e1)
        # CPU alone: n_jobs back-to-back 10-expert layers
        x, ids, w = rand_route()
        jobs = 40
        torch.cuda.synchronize(); e0, e1 = ev(), ev(); e0.record()
        for _ in range(jobs):
            host.submit_collect(host.submit_issue(lidx, x, ids.long(), w.half()))
            torch.cuda.current_stream().synchronize()
        e1.record(); torch.cuda.synchronize()
        cpu_alone = e0.elapsed_time(e1) / jobs
        # both
        torch.cuda.synchronize()
        with torch.cuda.stream(side):
            s0, s1 = ev(), ev(); s0.record(side)
            mod.copy_records_persistent(src, dst, L.lay.record_bytes, blocks=blocks); s1.record(side)
        e0, e1 = ev(), ev(); e0.record()
        n = 0
        while not s1.query() or n < 5:
            host.submit_collect(host.submit_issue(lidx, x, ids.long(), w.half())); n += 1
            torch.cuda.current_stream().synchronize()
        e1.record(); torch.cuda.synchronize()
        both_cpu = e0.elapsed_time(e1) / n
        both_pcie = s0.elapsed_time(s1)
        row = {"copy_blocks": blocks, "pcie_alone_ms": round(pcie_alone, 2),
               "pcie_GBps_alone": round(E * L.lay.record_bytes / pcie_alone / 1e6, 2),
               "pcie_ms_with_cpu": round(both_pcie, 2), "pcie_GBps_with_cpu": round(E * L.lay.record_bytes / both_pcie / 1e6, 2),
               "cpu_layer10_us_alone": round(cpu_alone * 1000, 1), "cpu_layer10_us_with_pcie": round(both_cpu * 1000, 1),
               "cpu_jobs_during_copy": n}
        out.setdefault("contention", []).append(row)
        print("contention", json.dumps(row), flush=True)
    json.dump(out, open(a.json, "w"), indent=1)
    print("K04_DONE", flush=True)


if __name__ == "__main__":
    main()
