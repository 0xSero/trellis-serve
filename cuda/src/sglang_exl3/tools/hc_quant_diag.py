"""K06a diagnosis: which HC matrix / granularity drives the int8 error (fp32 fake-quant math, real weights, 97 sites).
python -m sglang_exl3.tools.hc_quant_diag <model>"""
import sys, statistics, json
import torch
from ..kernels import hc_mix_int8 as hq
from .hc_int8_bench import load_sites, normed_input, HC, HS


def fq(w, g):
    q, s = hq.quantize(w, g)
    return hq.dequantize(q, s, g, torch.float32)


def main():
    sites = load_sites(sys.argv[1])
    gen = torch.Generator().manual_seed(0)
    xs = [normed_input(4, s["norm"], gen).float() for s in sites]
    refs = [hq.hc_mix_reference(x, s["down"].float(), s["up"].float(), HC, HS) for x, s in zip(xs, sites)]
    cfgs = [("bf16 weights", None, None)] + [(f"down g{gd} / up g{gu}", gd, gu) for gd, gu in
            [(0, None), (None, 0), (0, 0), (256, 0), (128, 0), (64, 0), (32, 0), (128, 64), (128, 32), (64, 32), (32, 32)]]
    for name, gd, gu in cfgs:
        es = []
        for x, s, ref in zip(xs, sites, refs):
            wd, wu = s["down"].float(), s["up"].float()
            if name.startswith("bf16"):
                wd, wu = s["down"].bfloat16().float(), s["up"].bfloat16().float()
            else:
                wd = fq(s["down"].bfloat16(), gd) if gd is not None else wd
                wu = fq(s["up"].bfloat16(), gu) if gu is not None else wu
            y = hq.hc_mix_reference(x, wd, wu, HC, HS)
            es.append(float((y - ref).norm() / ref.norm()))
        print(f"DIAG {name:28s} rel_l2 mean {statistics.mean(es):.5f} max {max(es):.5f}", flush=True)


if __name__ == "__main__":
    main()
