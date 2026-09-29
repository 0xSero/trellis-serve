"""GPU check of the n-gram NVMe tier (GPU 0 only): bit-exactness vs the pinned host table, CUDA-graph replay, and the
per-decode-step stall with the SGLang PLE pattern (gather on a side stream overlapping a layer-0 proxy on the main
stream).

  python3 tools/ngram_nvme_gpu_test.py --model /models/... --out /w/runs/... [--ram-gb 4] [--io aio] [--layer0-us 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ngram_nvme_bench import EOS, build_streams  # noqa: E402


def busy_kernel(x, us):
    """Main-stream proxy for layer 0: a matmul chain sized to ~us microseconds (calibrated)."""
    for _ in range(us):
        x = x @ x
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workloads", default="/w/kernels/cuda_sm86/traces/workloads.json")
    ap.add_argument("--k02", default="/w/runs/2026-09-29-K02-traces")
    ap.add_argument("--ram-gb", type=float, default=4.0)
    ap.add_argument("--io", default="aio")
    ap.add_argument("--layer0-us", default="0,250,500")
    ap.add_argument("--decode-steps", type=int, default=2000)
    ap.add_argument("--no-host", action="store_true", help="skip loading the 32.6 GB pinned reference table")
    ap.add_argument("--hint-stress", action="store_true",
                    help="tiny cache + a thread warming random rows during graph replays (hint/service eviction race)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from sglang_exl3.offload.ngram_nvme import Exl3NgramNvmeTable
    from sglang_exl3.offload.ngram_host import Exl3NgramHostTable
    res = {}
    streams = build_streams(a.model, a.workloads, a.k02)
    t0 = time.time()
    nv = Exl3NgramNvmeTable(a.model, ram_gb=a.ram_gb, io=a.io, max_tokens=16384)
    res["nvme_init_s"] = time.time() - t0
    host = None
    if not a.no_host:
        t0 = time.time()
        host = Exl3NgramHostTable(a.model)
        res["host_init_s"] = time.time() - t0

    if a.hint_stress:
        import threading
        stop = threading.Event()
        warmed = [0]

        def spam():
            g = torch.Generator().manual_seed(3)
            while not stop.is_set():
                nv.store.warm(torch.randint(0, nv.num_rows, (4096,), generator=g))
                warmed[0] += 4096
        ids_static = torch.zeros((1, 16), dtype=torch.long, device=dev)
        out_static = torch.empty((1, 16, 160), dtype=torch.bfloat16, device=dev)
        ref = torch.empty_like(out_static)
        side = torch.cuda.Stream()
        nv.gather(ids_static, out=out_static); torch.cuda.synchronize()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            nv.gather(ids_static, out=out_static)
        th = threading.Thread(target=spam, daemon=True); th.start()
        gen = torch.Generator().manual_seed(11)
        bad = 0
        for i in range(a.decode_steps):
            ids = torch.randint(0, nv.num_rows, (1, 16), generator=gen)
            if i % 2:   # half the steps reuse recently warmed-looking rows: repeat the previous ids
                ids = last
            last = ids
            ids_static.copy_(ids.to(dev))
            g2.replay()
            host.gather(ids_static, out=ref)
            torch.cuda.synchronize()
            bad += int(not torch.equal(ref, out_static))
        stop.set(); th.join()
        res.update(hint_stress_steps=a.decode_steps, hint_stress_mismatch=bad, warmed_rows=warmed[0], stats=nv.stats())
        print(json.dumps(res), flush=True)
        json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
        nv.release(); host.release()
        print("DONE", flush=True)
        return

    # --- 1. eager bit-exactness on real ids: a long prompt (prefill chunks) + decode-shaped lookups
    s = [x for x in streams if x["name"].startswith("L1")][0]
    ids_p = nv.hash_tokens(s["prompt"], eos=EOS)
    ids_d = nv.hash_tokens(s["completion"], history=s["prompt"][-2:], eos=EOS)
    mism = 0
    checked = 0
    pre_ms = []
    for a0 in range(0, ids_p.shape[0], 16384):
        x = ids_p[a0:a0 + 16384].to(dev)
        torch.cuda.synchronize()
        t = time.perf_counter()
        o = nv.gather(x)
        torch.cuda.synchronize()
        pre_ms.append((time.perf_counter() - t) * 1e3)
        if host is not None:
            mism += int((o.view(-1, 160) != host.gather(x).view(-1, 160)).any(-1).sum())
        checked += x.numel()
    res["eager_prefill_ms"] = pre_ms
    # host table timing on the same chunk (the pinned-tier baseline)
    if host is not None:
        x = ids_p[:16384].to(dev)
        host.gather(x); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            host.gather(x)
        torch.cuda.synchronize()
        res["pinned_16k_ms"] = (time.perf_counter() - t) * 1e3 / 5
    for b in (1, 3):
        for i in range(0, 300, b):
            x = ids_d[i:i + b].to(dev)
            o = nv.gather(x)
            if host is not None:
                mism += int((o.view(-1, 160) != host.gather(x).view(-1, 160)).any(-1).sum())
            checked += x.numel()
    torch.cuda.synchronize()
    res.update(eager_rows_checked=checked, eager_rows_mismatch=mism)
    print("eager", res, flush=True)

    # --- 2. CUDA graph: decode-shaped lookup captured once, replayed on fresh ids; PLE side-stream pattern
    side = torch.cuda.Stream()
    main_s = torch.cuda.Stream()
    ids_static = torch.zeros((1, 16), dtype=torch.long, device=dev)
    out_static = torch.empty((1, 16, 160), dtype=torch.bfloat16, device=dev)
    ref_out = torch.empty_like(out_static)
    mat = torch.randn(256, 256, device=dev, dtype=torch.float16) * 0.01
    # calibrate one 256^3 matmul
    with torch.cuda.stream(main_s):
        for _ in range(10):
            mat @ mat
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); y = mat
        for _ in range(200):
            y = mat @ mat
        e1.record(); torch.cuda.synchronize()
        mm_us = e0.elapsed_time(e1) * 1e3 / 200
    res["matmul_us"] = mm_us

    def build_graph(table, layer0_us):
        n_mm = int(round(layer0_us / mm_us))
        g = torch.cuda.CUDAGraph()
        # warm (eager) once on the capture streams
        with torch.cuda.stream(main_s):
            side.wait_stream(main_s)
            with torch.cuda.stream(side):
                table.gather(ids_static, out=out_static)
            z = mat
            for _ in range(n_mm):
                z = mat @ mat
            main_s.wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=main_s):
            side.wait_stream(main_s)
            with torch.cuda.stream(side):
                table.gather(ids_static, out=out_static)
            z = mat
            for _ in range(n_mm):
                z = mat @ mat
            main_s.wait_stream(side)
            y = out_static.float().sum()
        return g

    # decode stream: a different request, its prompt prefilled first (cache state like a server)
    s2 = [x for x in streams if x["name"] == "C2"][0]
    p2 = nv.hash_tokens(s2["prompt"], eos=EOS)
    d2 = nv.hash_tokens(s2["completion"], history=s2["prompt"][-2:], eos=EOS)
    nv.gather(p2.to(dev)); torch.cuda.synchronize()
    graph_res = []
    step = 0
    for l0 in [int(x) for x in a.layer0_us.split(",")]:
        for name, table in (("nvme", nv), ("pinned", host)):
            if table is None:
                continue
            g = build_graph(table, l0)
            nv.reset_stats()
            times, bad = [], 0
            n = min(a.decode_steps, d2.shape[0] - step) if name == "nvme" else min(a.decode_steps, 500)
            base = step
            for i in range(n):
                ids_static.copy_(d2[base + i].view(1, 16).to(dev))
                torch.cuda.synchronize()
                t = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t) * 1e6)
                if host is not None and name == "nvme" and i % 5 == 0:
                    host.gather(ids_static, out=ref_out); torch.cuda.synchronize()
                    bad += int(not torch.equal(ref_out, out_static))
            if name == "nvme":
                step += n
            st = nv.stats() if name == "nvme" else {}
            r = dict(table=name, layer0_us=l0, steps=n, replay_us_mean=float(np.mean(times)),
                     replay_us_p50=float(np.percentile(times, 50)), replay_us_p90=float(np.percentile(times, 90)),
                     replay_us_p99=float(np.percentile(times, 99)), replay_mismatch=bad)
            if st:
                r.update(lookup_hit=st["lookup_hit_rate"], gpu_wait_us_mean=st["gpu_wait_ns_total"] / max(st["gpu_waits"], 1) / 1e3,
                         gpu_wait_us_max=st["gpu_wait_ns_max"] / 1e3, error=st["error"])
            graph_res.append(r)
            print(json.dumps(r), flush=True)
            del g
    res["graph"] = graph_res
    res["final_stats"] = nv.stats()
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    nv.release()
    if host is not None:
        host.release()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
