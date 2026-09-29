"""forward_prefill_staged (copy-engine layer streaming) must equal the zero-copy forward bitwise."""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from safetensors import safe_open
from exl3xpu.moe_offload import ExpertStore, pack_expert
H, I, K, E = 2560, 640, 3, 512
dev = torch.device("xpu", 0)
M_ = "/model"
idx = json.load(open(f"{M_}/model.safetensors.index.json"))["weight_map"]
hd = {}
def get(n):
    f = idx[n]
    if f not in hd: hd[f] = safe_open(f"{M_}/{f}", "pt", device="cpu")
    return hd[f].get_tensor(n)
L = 3
st = ExpertStore(H, I, K, E, 600, dev, max_layers=L)
for l in range(L):
    h = st.add_layer(f"layers.{l}.mlp.experts")
    for e in range(E):
        p = f"model.language_model.layers.{l}.mlp.experts.{e}."
        t = {pr: {s: get(p + pr + "." + s) for s in ("trellis", "suh", "svh")} for pr in ("gate_proj", "up_proj", "down_proj")}
        pack_expert(t["gate_proj"], t["up_proj"], t["down_proj"], K, out=h[e])
    st.make_resident(f"layers.{l}.mlp.experts", list(range(l, E, 3)))
host_tab = {f"layers.{l}.mlp.experts": st._host_ptr[f"layers.{l}.mlp.experts"].to(dev) for l in range(L)}
bad = 0
# phase 2 (server-like): device-cache decode steps (write-through fills, LRU evictions) and eager small prefills
# interleaved with staged prefill forwards over all layers in order (wrap-around pre-staging)
st.reset_cache()
def route(M, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((M, H), generator=g) * 0.5).to(torch.bfloat16).to(dev)
    w, ids = torch.topk(torch.softmax(torch.randn((M, E), generator=g), -1), 10, -1)
    return x, ids.to(torch.int32).to(dev), (w / w.sum(-1, keepdim=True)).float().to(dev)
seed = 0
for rnd in range(4):
    for step in range(30):                       # decode through the device cache
        for l in range(L):
            seed += 1
            x, ids, w = route(1 if step % 5 else 40, seed)
            key = f"layers.{l}.mlp.experts"
            yc = st.forward_cached(key, x, ids, w)
            bad += not torch.equal(yc, st.forward(key, x, ids, w, ptrs=host_tab[key]))
    for chunk in range(2):                       # a staged prefill forward (all layers in order)
        for l in range(L):
            seed += 1
            key = f"layers.{l}.mlp.experts"
            x, ids, w = route(1024, seed)
            ys = st.forward_prefill_staged(key, x, ids, w)
            yz = st.forward(key, x, ids, w, ptrs=host_tab[key])
            eq = torch.equal(ys, yz); bad += not eq
            print("server-like", rnd, chunk, l, "equal", eq, (ys.float() - yz.float()).abs().max().item(), flush=True)
st.reset_cache()
st._pf = None; del st._pf
for l in range(L):
    st.make_resident(f"layers.{l}.mlp.experts", list(range(l, E, 3)))
for rep in range(3):
    for M in (600, 2048):
        for l in range(L):
            key = f"layers.{l}.mlp.experts"
            g = torch.Generator().manual_seed(rep * 100 + M + l)
            x = (torch.randn((M, H), generator=g) * 0.5).to(torch.bfloat16).to(dev)
            w, ids = torch.topk(torch.softmax(torch.randn((M, E), generator=g), -1), 10, -1)
            ids = ids.to(torch.int32).to(dev); w = (w / w.sum(-1, keepdim=True)).float().to(dev)
            ys = st.forward_prefill_staged(key, x, ids, w)
            yz = st.forward(key, x, ids, w, ptrs=host_tab[key])
            eq = torch.equal(ys, yz)
            bad += not eq
            print(rep, M, l, "equal", eq, (ys.float() - yz.float()).abs().max().item(), flush=True)
sys.exit(bad)
