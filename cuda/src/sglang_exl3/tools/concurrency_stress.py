"""SGLang capture-mode pattern: routed experts (grouped MoE kernel, alt_stream) concurrent with the shared expert (dense
EXL3 Marlin kernel, main stream). Stream A: CUDA graph of 12 decode MoE layers (layer 3 pack, T = 2); stream B: CUDA
graph of 12 shared-expert MLPs (layer 3 shared_expert gate/up/down, dense trellis_exl3_kernels, T = 2). `rounds`
concurrent replays; then outputs == the serial replay. A hang = deadlock (the caller's timeout kills the process).
    python -m sglang_exl3.tools.concurrency_stress <model> --rounds 3000"""
import argparse, time, sys
import torch
from safetensors import safe_open
import json, os
from ..kernels import marlin_moe, marlin
from .moe_parity_sm86 import load_layer
from .offload_moe_bench import align


def shared_expert(model, layer):
    idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.shared_expert."
    get = lambda k: safe_open(os.path.join(model, idx[k]), "pt").get_tensor(k)
    out = {}
    for p in ("gate_proj", "up_proj", "down_proj"):
        out[p] = [get(pre + p + s).cuda() for s in (".trellis", ".suh", ".svh")]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--rounds", type=int, default=3000)
    a = ap.parse_args()
    _, t, cb = load_layer(a.model, 3, 512, "cpu")
    pack = marlin_moe.prepare(t["gate_proj"], t["up_proj"], t["down_proj"], cb)
    se = shared_expert(a.model, 3)
    gu_packed = marlin.prepare([se["gate_proj"][0], se["up_proj"][0]], [se["gate_proj"][1], se["up_proj"][1]],
                               [se["gate_proj"][2], se["up_proj"][2]])
    dn_packed = marlin.prepare([se["down_proj"][0]], [se["down_proj"][1]], [se["down_proj"][2]])
    print("prepared", type(gu_packed), flush=True)
    gen = torch.Generator().manual_seed(0)
    ids = torch.stack([torch.randperm(512, generator=gen)[:10] for _ in range(2)]).int().cuda()
    w = torch.softmax(torch.randn((2, 10), generator=gen), -1).cuda()
    x = (torch.randn((2, 2560), generator=gen) * 0.5).half().cuda()
    moe_fn = lambda: [marlin_moe.run(x, w, ids, *align(ids, 8, 512), 8, pack) for _ in range(12)]

    xa = (torch.randn((2, 640), generator=gen) * 0.5).half().cuda()

    def dense_fn():
        outs = []
        for _ in range(12):
            outs.append(marlin.run(x, *gu_packed, 2))       # shared expert gate|up (5-bit dense Marlin, 2 shards)
            outs.append(marlin.run(xa, *dn_packed, 2))      # shared expert down
        return outs
    fns = [moe_fn, dense_fn]
    graphs, ys = [], []
    for fn in fns:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fn()
        graphs.append(g); ys.append(y)
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    serial = [[v.clone() for v in y] for y in ys]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    t0 = time.time()
    for r in range(a.rounds):
        for i in range(2):
            with torch.cuda.stream(streams[i]):
                graphs[i].replay()
        if r % 500 == 0:
            torch.cuda.synchronize(); print("round", r, round(time.time() - t0, 2), flush=True)
    torch.cuda.synchronize()
    eq = all(torch.equal(p, q) for y, sy in zip(ys, serial) for p, q in zip(y, sy))
    print("STRESS", {"rounds": a.rounds, "seconds": round(time.time() - t0, 2), "equal_serial": eq}, flush=True)


if __name__ == "__main__":
    main()
