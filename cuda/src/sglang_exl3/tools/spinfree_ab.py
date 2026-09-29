"""A/B of the decode-family reduction (TRELLIS_SPINFREE_M8 0 vs 1) on a real layer + a concurrency stress test.
    python -m sglang_exl3.tools.spinfree_ab <model> --save /runs/x_spin0.pt            (build with SPINFREE_M8=0)
    python -m sglang_exl3.tools.spinfree_ab <model> --save /runs/x_spin1.pt --compare /runs/x_spin0.pt --stress
Outputs: decode layers (T = 1, 2, 4, 8; 100 fixed routings each; stacked grouped path) saved; compare = bit equality.
Stress: two streams each replaying a CUDA graph of 12 decode MoE layers, 1,000 rounds concurrently (the pattern of
SGLang's alt_stream routed experts next to a main-stream Marlin launch); must finish and match the serial outputs."""
import argparse, time
import torch
from ..kernels import marlin_moe
from .moe_parity_sm86 import load_layer
from .offload_moe_bench import align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--save", required=True); ap.add_argument("--compare")
    ap.add_argument("--stress", action="store_true"); ap.add_argument("--rounds", type=int, default=1000)
    a = ap.parse_args()
    _, t, cb = load_layer(a.model, 3, 512, "cpu")
    pack = marlin_moe.prepare(t["gate_proj"], t["up_proj"], t["down_proj"], cb)
    gen = torch.Generator().manual_seed(0)
    outs = []
    for T in (1, 2, 4, 8):
        for r in range(100):
            ids = torch.stack([torch.randperm(512, generator=gen)[:10] for _ in range(T)]).int().cuda()
            w = torch.softmax(torch.randn((T, 10), generator=gen), -1).cuda()
            x = (torch.randn((T, 2560), generator=gen) * 0.5).bfloat16().cuda()
            blk = marlin_moe.moe_block_size(T, 10, 512)
            outs.append(marlin_moe.run(x, w, ids, *align(ids, blk, 512), blk, pack).cpu())
    torch.save(outs, a.save)
    res = {}
    if a.compare:
        ref = torch.load(a.compare)
        res["bit_equal_all"] = all(torch.equal(p.view(torch.int16), q.view(torch.int16)) for p, q in zip(outs, ref))
        res["n"] = len(outs)
    if a.stress:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        graphs, ys, bufs = [], [], []
        for si in range(2):
            ids = torch.stack([torch.randperm(512, generator=gen)[:10] for _ in range(2)]).int().cuda()
            w = torch.softmax(torch.randn((2, 10), generator=gen), -1).cuda()
            x = (torch.randn((2, 2560), generator=gen) * 0.5).bfloat16().cuda()
            fn = lambda x=x, w=w, ids=ids, si=si: [marlin_moe.run(x, w, ids, *align(ids, 8, 512), 8, pack, scratch=si)
                                                   for _ in range(12)]
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = fn()
            graphs.append(g); ys.append(y); bufs.append((x, w, ids))
        for g in graphs:
            g.replay()
        torch.cuda.synchronize()
        serial = [[t_.clone() for t_ in y] for y in ys]
        t0 = time.time()
        for r in range(a.rounds):
            for si in range(2):
                with torch.cuda.stream(streams[si]):
                    graphs[si].replay()
        torch.cuda.synchronize()
        res["stress_rounds"] = a.rounds
        res["stress_s"] = round(time.time() - t0, 2)
        res["stress_equal_serial"] = all(torch.equal(p, q) for y, sy in zip(ys, serial) for p, q in zip(y, sy))
    print("SPINFREE_AB", res, flush=True)


if __name__ == "__main__":
    main()
