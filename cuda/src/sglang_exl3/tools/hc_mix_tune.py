"""Tune the launch of SGLang's persistent hyper-connection mix kernel (sglang.srt.layers.hc_mix_triton) for the 3090
at decode size (rows 1..3, K = 4 x 2560, low rank 320). Same kernel, different tile sizes / warps / stages; results
checked against the default launch (fp32 atomics -> compare with a tolerance)."""
import itertools
import time

import torch
from sglang.srt.layers import hc_mix_triton as hm


def launch(x, wd, wu, hc, hs, BN, BK, BJ, BR, warps, stages):
    rows, k = x.shape
    lowrank = wd.shape[0]
    num_ctas = torch.cuda.get_device_properties(x.device).multi_processor_count
    t_raw = torch.empty((16, lowrank), dtype=torch.float32, device=x.device)
    out = torch.empty((rows, hs), dtype=x.dtype, device=x.device)
    hm._hc_mix_persistent_kernel[(num_ctas,)](x, wd, wu, t_raw, out, hm._get_counters(x.device), k, lowrank, hs, rows,
                                               num_ctas, 1.0 / hc, ROWS=16, HC=hc, BLOCK_N=BN, BLOCK_K=BK, BLOCK_J=BJ,
                                               BLOCK_R=BR, num_warps=warps, num_stages=stages)
    return out


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    hc, hs, lr = 4, 2560, 320
    K = hc * hs
    torch.manual_seed(0)
    # 8 independent weight sets so the loop does not hit L2 (the model has 97 of them)
    sets = [((torch.randn(lr, K, device="cuda") * 0.02).to(torch.bfloat16),
             (torch.randn(K, lr, device="cuda") * 0.02).to(torch.bfloat16)) for _ in range(8)]
    x = torch.randn(1, K, device="cuda").to(torch.bfloat16)
    ref = [hm.fused_hc_mix(x, wd, wu, hc, hs).float() for wd, wu in sets]
    i = {"n": 0}

    def run_default():
        wd, wu = sets[i["n"] % 8]; i["n"] += 1
        return hm.fused_hc_mix(x, wd, wu, hc, hs)
    base = bench(run_default)
    print(f"default (BN32 BK256 BJ32 BR64 w8): {base:.1f} us/call ({13.1e6 / base / 1e3:.0f} GB/s)", flush=True)
    results = []
    for BN, BK, BJ, BR, warps, stages in itertools.product((32, 64), (128, 256, 512), (16, 32), (64,), (4, 8), (1, 2, 3)):
        try:
            outs = [launch(x, wd, wu, hc, hs, BN, BK, BJ, BR, warps, stages).float() for wd, wu in sets]
            err = max((o - r).abs().max().item() for o, r in zip(outs, ref))

            def run():
                wd, wu = sets[i["n"] % 8]; i["n"] += 1
                return launch(x, wd, wu, hc, hs, BN, BK, BJ, BR, warps, stages)
            us = bench(run)
        except Exception as e:  # noqa: BLE001
            print(f"BN{BN} BK{BK} BJ{BJ} BR{BR} w{warps} s{stages}: FAIL {str(e)[:80]}", flush=True)
            continue
        results.append((us, BN, BK, BJ, BR, warps, stages, err))
        print(f"BN{BN} BK{BK} BJ{BJ} BR{BR} w{warps} s{stages}: {us:.1f} us, max err {err:.2e}", flush=True)
    results.sort()
    print("BEST", results[:5])


if __name__ == "__main__":
    main()
