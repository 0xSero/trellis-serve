"""Cost of the device cache's ensure step vs slot-pool size S (argmin per miss). Synthetic blobs (content irrelevant
for timing): L layers x 512 experts in host USM, S device slots, uniform routing (worst case: most calls miss)."""
import os, sys, time, argparse, json
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from exl3xpu.moe_offload import ExpertStore
ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=24)
ap.add_argument("--slots", default="1024,4096,12000")
ap.add_argument("--out", default="")
a = ap.parse_args()
H, I, K, E = 2560, 640, 3, 512
dev = torch.device("xpu", 0)
res = []
for S in [int(v) for v in a.slots.split(",")]:
    st = ExpertStore(H, I, K, E, S, dev, max_layers=a.layers)
    for L in range(a.layers):
        h = st.add_layer(L)
        h.zero_()      # fp16 zeros: valid (all-zero) experts
    for M in (1, 16):
        x = torch.randn((M, H), device=dev).to(torch.bfloat16)
        rts = []
        for i in range(64):
            g = torch.Generator().manual_seed(i)
            w, ids = torch.topk(torch.rand((M, E), generator=g), 10, -1)
            rts.append((ids.to(torch.int32).to(dev), (w / w.sum(-1, keepdim=True)).float().to(dev)))
        for rep, fill, off in (("fill_all", E, 0), ("fill_none", 0, 7), ("repeat_hits", 0, 0)):
            for i in range(10):
                st.forward_cached(i % a.layers, x, *rts[(i + off) % 64], max_fill=fill)
            torch.xpu.synchronize()
            N = 3 * a.layers
            t = time.perf_counter()
            for i in range(N):
                st.forward_cached(i % a.layers, x, *rts[(i * 5 + off) % 64] if rep != "repeat_hits" else rts[(i % a.layers) % 64],
                                  max_fill=fill)
            torch.xpu.synchronize()
            us = (time.perf_counter() - t) / N * 1e6
            r = {"S": S, "M": M, "phase": rep, "max_fill": fill, "us_per_layer": round(us, 1)}
            res.append(r)
            print(r, flush=True)
    del st
    torch.xpu.empty_cache()
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
