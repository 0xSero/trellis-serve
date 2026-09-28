"""
XPU graph capture of the MoE ops (what SGLang-XPU does in decode): capture N layers of moe_forward (and
moe_forward_cached) with static inputs, replay with new routing written into the static tensors, compare with eager
(zero-copy reference, bitwise), time replay vs eager.

  python3 tests/test_moe_graph.py --layers 0,1,2,3 --slots 1024
"""
import os, sys, json, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from safetensors import safe_open
from exl3xpu.moe_offload import ExpertStore, pack_expert

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "/model"))
ap.add_argument("--layers", default="0,1,2,3")
ap.add_argument("--slots", type=int, default=1024)
ap.add_argument("--out", default="")
args = ap.parse_args()
H, I, K, E, TOPK = 2560, 640, 3, 512, 10
dev = torch.device("xpu", 0)
idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
handles = {}


def get(n):
    fn = idx[n]
    if fn not in handles:
        handles[fn] = safe_open(f"{args.model}/{fn}", "pt", device="cpu")
    return handles[fn].get_tensor(n)


layers = [int(v) for v in args.layers.split(",")]
store = ExpertStore(H, I, K, E, args.slots, dev, max_layers=len(layers))
keys = []
for L in layers:
    key = f"layers.{L}.mlp.experts"
    keys.append(key)
    h = store.add_layer(key)
    for e in range(E):
        p = f"model.language_model.layers.{L}.mlp.experts.{e}."
        t = {pr: {s: get(p + pr + "." + s) for s in ("trellis", "suh", "svh")} for pr in ("gate_proj", "up_proj", "down_proj")}
        pack_expert(t["gate_proj"], t["up_proj"], t["down_proj"], K, out=h[e])
    store.make_resident(key, list(range(0, E, 3)))          # a static third resident, the cache handles the rest
host_tab = {k: store._host_ptr[k].to(dev) for k in keys}
res = []


def rout(M, seed):
    g = torch.Generator().manual_seed(seed)
    w, ids = torch.topk(torch.softmax(torch.randn((M, E), generator=g) - 0.7 * torch.log1p(torch.arange(E).float()), -1), TOPK, -1)
    return ids.to(torch.int32), (w / w.sum(-1, keepdim=True)).float()


for cached in (False, True):
    for M in (1, 4, 16):
        xs = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)
        ids_s = torch.zeros((len(keys), M, TOPK), dtype=torch.int32, device=dev)
        w_s = torch.zeros((len(keys), M, TOPK), dtype=torch.float32, device=dev)
        outs = [None] * len(keys)

        def body():
            for li, k in enumerate(keys):
                f = store.forward_cached if cached else store.forward
                outs[li] = f(k, xs, ids_s[li], w_s[li])

        def load_inputs(seed):
            xs.copy_((torch.randn((M, H), generator=torch.Generator().manual_seed(seed)) * 0.5).to(torch.bfloat16))
            for li in range(len(keys)):
                i, w = rout(M, seed * 100 + li)
                ids_s[li].copy_(i)
                w_s[li].copy_(w)

        load_inputs(1)
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            for _ in range(3):
                body()
        torch.xpu.current_stream().wait_stream(s)
        torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            body()
        torch.xpu.synchronize()
        bad = 0
        for seed in range(2, 12):
            load_inputs(seed)
            g.replay()
            torch.xpu.synchronize()
            for li, k in enumerate(keys):
                ref = store.forward(k, xs, ids_s[li], w_s[li], ptrs=host_tab[k])
                if not torch.equal(outs[li], ref):
                    bad += 1
        torch.xpu.synchronize()
        N = 200
        t = time.perf_counter()
        for _ in range(N):
            g.replay()
        torch.xpu.synchronize()
        t_graph = (time.perf_counter() - t) / N / len(keys) * 1e6
        t = time.perf_counter()
        for _ in range(N):
            body()
        torch.xpu.synchronize()
        t_eager = (time.perf_counter() - t) / N / len(keys) * 1e6
        rec = {"cached": cached, "M": M, "layers_in_graph": len(keys), "replay_mismatches": bad,
               "graph_us_per_layer": round(t_graph, 1), "eager_us_per_layer": round(t_eager, 1)}
        res.append(rec)
        print(rec, flush=True)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
sys.exit(1 if any(r["replay_mismatches"] for r in res) else 0)
