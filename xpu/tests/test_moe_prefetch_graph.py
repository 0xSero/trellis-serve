"""
X006b: copy-engine prefetch measured GPU-bound: one token (L layers of [non-MoE GEMMs -> forward_cached -> plan next
layer]) captured in an XPU graph and replayed per token with fresh routing written into static tensors, so host launch
cost does not hide the device timeline (the eager harness was CPU-bound: 3.6 ms wall vs 0.85 ms GPU per layer).

  python3 tests/test_moe_prefetch_graph.py --layers 8 --slots 1024 --gemms 8
"""
import os, sys, json, time, argparse
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from safetensors import safe_open
from exl3xpu.moe_offload import ExpertStore, pack_expert

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "/model"))
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--slots", type=int, default=1024)
ap.add_argument("--tokens", type=int, default=40)
ap.add_argument("--gemms", default="0,4,8,16", help="1024^2 bf16 GEMMs of non-MoE work per layer (~24 us each)")
ap.add_argument("--zipf", type=float, default=0.5)
ap.add_argument("--out", default="")
a = ap.parse_args()
H, I, K, E, TOPK = 2560, 640, 3, 512, 10
dev = torch.device("xpu", 0)
idx = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
handles = {}


def get(n):
    fn = idx[n]
    if fn not in handles:
        handles[fn] = safe_open(f"{a.model}/{fn}", "pt", device="cpu")
    return handles[fn].get_tensor(n)


store = ExpertStore(H, I, K, E, a.slots, dev, max_layers=a.layers)
for L in range(a.layers):
    h = store.add_layer(L)
    for e in range(E):
        p = f"model.language_model.layers.{L}.mlp.experts.{e}."
        t = {pr: {s: get(p + pr + "." + s) for s in ("trellis", "suh", "svh")} for pr in ("gate_proj", "up_proj", "down_proj")}
        pack_expert(t["gate_proj"], t["up_proj"], t["down_proj"], K, out=h[e])
host_tab = {L: store._host_ptr[L].to(dev) for L in range(a.layers)}
pop = {L: -a.zipf * torch.log1p(torch.randperm(E, generator=torch.Generator().manual_seed(100 + L)).float()) for L in range(a.layers)}
A = torch.randn((1024, 1024), device=dev, dtype=torch.bfloat16)


def routes(tok):
    ids, ws = [], []
    for L in range(a.layers):
        g = torch.Generator().manual_seed(tok * 1000 + L)
        w, i = torch.topk(torch.softmax(torch.randn((1, E), generator=g) + pop[L], -1), TOPK, -1)
        ids.append(i.to(torch.int32)); ws.append((w / w.sum(-1, keepdim=True)).float())
    return torch.stack(ids), torch.stack(ws)


def predict(ids, p, tok):
    g = torch.Generator().manual_seed(tok * 7919)
    keep = torch.rand(ids.shape, generator=g) < p
    return torch.where(keep, ids, torch.randint(0, E, ids.shape, generator=g, dtype=torch.int32))


res = []
for ng in [int(v) for v in a.gemms.split(",")]:
    for mode, p in (("none", None), ("copy_engine", 0.7), ("copy_engine", 1.0)):
        store.reset_cache()
        if mode == "copy_engine":
            store.start_copy_prefetch()
        x = (torch.randn((1, H), device=dev) * 0.5).to(torch.bfloat16)
        ids_s = torch.zeros((a.layers, 1, TOPK), dtype=torch.int32, device=dev)
        w_s = torch.zeros((a.layers, 1, TOPK), dtype=torch.float32, device=dev)
        pred_s = torch.zeros((a.layers, 1, TOPK), dtype=torch.int32, device=dev)   # pred_s[L] = prediction for layer L
        outs = [None] * a.layers

        def token():
            for L in range(a.layers):
                for _ in range(ng):
                    A @ A
                outs[L] = store.forward_cached(L, x, ids_s[L], w_s[L])
                if mode == "copy_engine" and L + 1 < a.layers:
                    store.plan_prefetch(L + 1, pred_s[L + 1])

        def load(tok):
            i, w = routes(tok)
            ids_s.copy_(i); w_s.copy_(w)
            if p is not None:
                pred_s.copy_(predict(i, p, tok))       # prediction of this token's layer-L routing (L >= 1 used)

        load(0)
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            for _ in range(2):
                token()
        torch.xpu.current_stream().wait_stream(s)
        torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            token()
        torch.xpu.synchronize()
        bad = 0
        warm = a.tokens // 4
        t0 = None
        for tok in range(1, a.tokens + 1):
            if tok == warm:
                torch.xpu.synchronize()
                t0 = time.perf_counter()
            load(tok)
            g.replay()
            if tok % 10 == 0:
                torch.xpu.synchronize()
                for L in range(a.layers):
                    ref = store.forward(L, x, ids_s[L], w_s[L], ptrs=host_tab[L])
                    bad += int(not torch.equal(outs[L], ref))
        torch.xpu.synchronize()
        dt_ = (time.perf_counter() - t0) / ((a.tokens + 1 - warm) * a.layers) * 1e6
        stats = store.stop_copy_prefetch() if mode == "copy_engine" else None
        rec = {"gemms_per_layer": ng, "mode": mode, "prefetch_accuracy": p, "us_per_layer": round(dt_, 1),
               "worker_plans_copies": stats, "unequal_outputs": bad}
        res.append(rec)
        print(rec, flush=True)
        del g
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
sys.exit(1 if any(r["unequal_outputs"] for r in res) else 0)
