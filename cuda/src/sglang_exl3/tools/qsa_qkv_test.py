"""Low-bit QSA KV: quant->dequant error (vs bf16 and vs fp8 e4m3) and extract_qkv == torch reference."""
import sys, torch
from exllamav3.ext import exllamav3_ext as ext
from sglang_exl3.sglang_glue.qsa_qkv import QKVView, extract_qkv
dev = "cuda"; H, D = 2, 256
torch.manual_seed(0)
for bits in (8, 6, 5, 4, 3):
    x = (torch.randn(4096, H, D, device=dev) * 2).to(torch.bfloat16)
    x[:, :, :8] *= 8   # a few large channels, like real K
    q = torch.empty(4096, H, D // 32 * bits, dtype=torch.int32, device=dev)
    s = torch.empty(4096, H, D // 32, dtype=torch.float16, device=dev)
    ext.quant_cache_cont(x.half().contiguous(), q, s, 0.0)
    y = QKVView(q, s, D).index_select(0, torch.arange(4096, device=dev))
    rel = ((y.float() - x.float()).norm() / x.float().norm()).item()
    f8 = x.to(torch.float8_e4m3fn).to(torch.bfloat16)
    rel8 = ((f8.float() - x.float()).norm() / x.float().norm()).item()
    print(f"bits {bits}: rel err {rel:.4f} (fp8 e4m3 {rel8:.4f})")
# extraction vs reference
bits = 4; P = 3000; B = 3; topk = 2048
kx = (torch.randn(P, H, D, device=dev)).half(); vx = (torch.randn(P, H, D, device=dev)).half()
def qz(x):
    q = torch.empty(P, H, D // 32 * bits, dtype=torch.int32, device=dev); s = torch.empty(P, H, D // 32, dtype=torch.float16, device=dev)
    ext.quant_cache_cont(x.contiguous(), q, s, 0.0); return QKVView(q, s, D)
K, V = qz(kx), qz(vx)
rtt = torch.randint(0, P, (8, 4096), device=dev, dtype=torch.int32)
ri = torch.tensor([2, 5, 7], device=dev); sl = torch.tensor([3000, 17, 2048], device=dev, dtype=torch.int32)
cnt = [2048, 17, 1000]
idx = torch.full((B, topk), -1, device=dev, dtype=torch.int32)
for b in range(B): idx[b, :cnt[b]] = torch.randperm(int(sl[b]), device=dev)[:cnt[b]].to(torch.int32)
cu = torch.zeros(B + 1, device=dev, dtype=torch.int32); cu[1:] = torch.tensor(cnt, device=dev).cumsum(0)
ok = torch.full((B * topk, H, D), float("nan"), device=dev, dtype=torch.bfloat16); ov = ok.clone()
extract_qkv(K, V, rtt, ri, idx, sl, cu, ok, ov, B, topk)
good = True
for b in range(B):
    n = cnt[b]; slots = rtt[ri[b], idx[b, :n].long()].long(); a, e = int(cu[b]), int(cu[b + 1])
    good &= torch.equal(ok[a:e], K.index_select(0, slots)) and torch.equal(ov[a:e], V.index_select(0, slots))
print("QSA_QKV_TEST", "PASS" if good else "FAIL")
