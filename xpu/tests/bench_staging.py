"""
X002: prefill layer streaming on the B70. One MoE layer of routed experts (512 x 1.862 MB = 0.954 GB) lives in USM
host memory; it is copied H2D into one of two device staging buffers on a second XPU queue while the previous layer
computes from the other buffer (pointer table -> staging buffer). Measures:
  - H2D copy alone (queue.memcpy from host USM), GB/s
  - prefill compute alone (M rows), ms/layer
  - overlapped pipeline over N layers, ms/layer, vs max(copy, compute)
  - zero-copy prefill (pointer table -> host USM, no staging) for reference

  python3 tests/bench_staging.py --m 2048,8192 --layers 6
"""
import os, sys, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv, _argv = [sys.argv[0]], sys.argv
exec(open(os.path.join(HERE, "test_moe.py")).read().split("res = {")[0])   # helpers: load_layer, pack, routing, ...
sys.argv = _argv
ap2 = argparse.ArgumentParser()
ap2.add_argument("--m", default="2048,8192")
ap2.add_argument("--layers", type=int, default=6)
ap2.add_argument("--out", default="")
a2 = ap2.parse_args()

host, devarena = load_layer(0)
LB = E * BLOB
stage = [torch.empty(LB, dtype=torch.uint8, device=dev) for _ in range(2)]
hp = s64(host.data_ptr())
ptr_stage = [(s64(st.data_ptr()) + torch.arange(E, dtype=torch.int64) * BLOB).to(dev) for st in stage]
ptr_host = ((hp + torch.arange(E, dtype=torch.int64) * BLOB)).to(dev)
cs = torch.xpu.current_stream()
ce = torch.xpu.Stream()   # second queue for copies
out = {"copy": [], "compute": [], "pipeline": []}


def copy_layer(i):
    X.memcpy_async(s64(stage[i % 2].data_ptr()), hp, LB)


# 1) copy alone
for _ in range(2):
    with torch.xpu.stream(ce):
        copy_layer(0)
torch.xpu.synchronize()
for rep in range(3):
    t = time.perf_counter()
    with torch.xpu.stream(ce):
        for i in range(4):
            copy_layer(i)
    torch.xpu.synchronize()
    dt_ = (time.perf_counter() - t) / 4
    r = {"what": "H2D copy one layer (host USM -> device), side queue", "ms": round(dt_ * 1e3, 2), "GBps": round(LB / dt_ / 1e9, 2)}
    out["copy"].append(r)
    print(r, flush=True)

for M in [int(v) for v in a2.m.split(",")]:
    x = (torch.randn((M, H), device=dev) * 0.5).to(dt)
    ids, w = routing(M, torch.Generator().manual_seed(M))
    # stage buffers hold real weights
    for i in range(2):
        stage[i].copy_(devarena.view(-1))
    torch.xpu.synchronize()
    # 2) compute alone
    for _ in range(2):
        X.moe_forward(x, ids, w, ptr_stage[0], I, K, E)
    torch.xpu.synchronize()
    t = time.perf_counter()
    for i in range(a2.layers):
        X.moe_forward(x, ids, w, ptr_stage[i % 2], I, K, E)
    torch.xpu.synchronize()
    t_comp = (time.perf_counter() - t) / a2.layers
    # zero-copy prefill
    X.moe_forward(x, ids, w, ptr_host, I, K, E)
    torch.xpu.synchronize()
    t = time.perf_counter()
    X.moe_forward(x, ids, w, ptr_host, I, K, E)
    torch.xpu.synchronize()
    t_zc = time.perf_counter() - t
    rc = {"M": M, "compute_ms": round(t_comp * 1e3, 2), "zero_copy_prefill_ms": round(t_zc * 1e3, 2)}
    out["compute"].append(rc)
    print(rc, flush=True)
    # 3) pipeline: copy layer i+1 on ce while computing layer i on cs
    for rep in range(2):
        torch.xpu.synchronize()
        t = time.perf_counter()
        ev_copied = [torch.xpu.Event() for _ in range(a2.layers + 1)]
        ev_done = [torch.xpu.Event() for _ in range(a2.layers + 1)]
        with torch.xpu.stream(ce):
            copy_layer(0)
            ev_copied[0].record(ce)
        for i in range(a2.layers):
            if i + 1 < a2.layers:
                with torch.xpu.stream(ce):
                    if i >= 1:
                        ce.wait_event(ev_done[i - 1])      # buffer (i+1)%2 was used by layer i-1
                    copy_layer(i + 1)
                    ev_copied[i + 1].record(ce)
            cs.wait_event(ev_copied[i])
            X.moe_forward(x, ids, w, ptr_stage[i % 2], I, K, E)
            ev_done[i].record(cs)
        torch.xpu.synchronize()
        t_pipe = (time.perf_counter() - t) / a2.layers
        copy_ms = out["copy"][-1]["ms"]
        rp = {"M": M, "layers": a2.layers, "pipeline_ms_per_layer": round(t_pipe * 1e3, 2),
              "bound_max_copy_compute_ms": round(max(copy_ms, t_comp * 1e3), 2),
              "serial_sum_ms": round(copy_ms + t_comp * 1e3, 2),
              "tok_per_s_48_layers_moe_only": round(M / (t_pipe * 48), 0)}
        out["pipeline"].append(rp)
        print(rp, flush=True)

if a2.out:
    import json
    json.dump(out, open(a2.out, "w"), indent=1)
