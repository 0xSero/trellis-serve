"""Triton QSA decode kernel vs the fp32 torch reference (random packed K/V, ragged lengths incl. 0 and full), graph
replay == eager, timing at B=1, T=2048 (the Flash-Next budget)."""
import time
import torch
from sglang_exl3.sglang_glue.qsa_sm86 import varlen_decode_attention_torch
from sglang_exl3.sglang_glue.qsa_decode_triton import qsa_decode_attention


def case(B, T, lens, Hq=24, Hkv=2, D=256, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((B, Hq, D), generator=g, device="cuda").to(torch.bfloat16)
    N = B * T
    k = torch.randn((N, Hkv, D), generator=g, device="cuda").to(torch.bfloat16)
    v = torch.randn((N, Hkv, D), generator=g, device="cuda").to(torch.bfloat16)
    # poison the unused tail of every row with NaN (like uninitialised scratch)
    cu = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    used = int(cu[-1])
    k[used:] = float("nan"); v[used:] = float("nan")
    return q, k, v, cu


ok = True
for B, T, lens in ((1, 2048, [2048]), (1, 2048, [1]), (3, 2048, [5, 0, 2048]), (2, 2048, [1000, 77]), (4, 512, [512, 3, 400, 1])):
    q, k, v, cu = case(B, T, lens)
    ref = varlen_decode_attention_torch(q, k, v, cu, T, 256 ** -0.5).float()
    out = qsa_decode_attention(q, k, v, cu, T, 256 ** -0.5).float()
    err = (out - ref).abs().max().item()
    fin = torch.isfinite(out).all().item()
    print(f"B={B} T={T} lens={lens}: max abs err {err:.2e}, finite {fin}")
    ok &= fin and err < 2e-2
# graph
q, k, v, cu = case(1, 2048, [2048])
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    qsa_decode_attention(q, k, v, cu, 2048, 0.0625)
torch.cuda.current_stream().wait_stream(s)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    go = qsa_decode_attention(q, k, v, cu, 2048, 0.0625)
for n in (2048, 700, 1):
    cu[1] = n
    q.normal_()
    gr.replay()
    ok &= torch.equal(go, qsa_decode_attention(q, k, v, cu, 2048, 0.0625))
print("graph==eager", ok)
for fn, name in ((qsa_decode_attention, "triton"), (varlen_decode_attention_torch, "torch")):
    for _ in range(5):
        fn(q, k, v, cu, 2048, 0.0625)
    cu[1] = 2048
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100):
        fn(q, k, v, cu, 2048, 0.0625)
    torch.cuda.synchronize()
    print(f"{name}: {(time.perf_counter() - t0) / 100 * 1e6:.1f} us per call (B=1, 2048 keys, eager)")
print("QSA_DECODE_TEST", "PASS" if ok else "FAIL")
