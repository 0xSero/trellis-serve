"""e4m3 KV extraction (uint8 + LUT Triton kernel) vs torch gather + cast; bit-exact expected."""
import torch
from sglang_exl3.sglang_glue import qsa_fp8e4m3
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb
qsa_fp8e4m3.install()
dev = "cuda"
P, H, D, B, topk = 5000, 2, 256, 3, 2048
kf = (torch.randn((P, H, D), device=dev) * 3).to(torch.float8_e4m3fn)
vf = (torch.randn((P, H, D), device=dev) * 3).to(torch.float8_e4m3fn)
req_to_token = torch.randint(0, P, (8, 4096), device=dev, dtype=torch.int32)
req_idx = torch.tensor([2, 5, 7], device=dev, dtype=torch.int64)
seq = torch.tensor([3000, 17, 2048], device=dev, dtype=torch.int32)
counts = [2048, 17, 1000]
idx = torch.full((B, topk), -1, device=dev, dtype=torch.int32)
for b in range(B):
    idx[b, :counts[b]] = torch.randperm(int(seq[b]), device=dev)[:counts[b]].to(torch.int32)
cu = torch.zeros(B + 1, device=dev, dtype=torch.int32); cu[1:] = torch.tensor(counts, device=dev).cumsum(0)
ok_ = torch.empty((B * topk, H, D), device=dev, dtype=torch.bfloat16).fill_(float("nan"))
ov_ = ok_.clone()
qb.qwen_sparse_kv_extraction_compact_triton(kf, vf, req_to_token, req_idx, idx, seq, cu, ok_, ov_, B, topk)
good = True
for b in range(B):
    n = counts[b]
    slots = req_to_token[req_idx[b], idx[b, :n].long()].long()
    rk = kf[slots].to(torch.bfloat16); rv = vf[slots].to(torch.bfloat16)
    a, e = int(cu[b]), int(cu[b + 1])
    good &= torch.equal(ok_[a:e], rk) and torch.equal(ov_[a:e], rv)
print("QSA_E4M3_TEST", "PASS" if good else "FAIL")
