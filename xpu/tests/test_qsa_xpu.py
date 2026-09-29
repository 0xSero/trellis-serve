"""exl3xpu/qsa_xpu.py vs SGLang's per-row reference implementations (random inputs, XPU)."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sglang.srt.layers.attention.qsa.kernel as K
ref_topk, ref_attn = K.qsa_fast_topk, K.qsa_sparse_attention_reference
from exl3xpu import qsa_xpu
dev = torch.device("xpu", 0)
torch.manual_seed(0)
bad = 0
for R, C, topk in ((5, 3000, 512), (3, 700, 512), (4, 9000, 2048)):
    logits = torch.randn(R, C, device=dev)
    starts = torch.randint(0, 50, (R,), device=dev, dtype=torch.int32)
    ends = torch.minimum(starts + torch.randint(0, C, (R,), device=dev, dtype=torch.int32), torch.tensor(C, device=dev))
    ends[0] = starts[0]                                  # an empty row
    a = ref_topk(logits, starts, ends, topk); b = qsa_xpu.qsa_fast_topk(logits, starts, ends, topk)
    same = all(set(a[r][a[r] >= 0].tolist()) == set(b[r][b[r] >= 0].tolist()) for r in range(R))
    bad += not same
    print("topk", R, C, topk, "same sets:", same)
N, Hk, G, D = 5000, 2, 12, 256
kc = torch.randn(N, Hk, D, device=dev).to(torch.float8_e4m3fn); vc = torch.randn(N, Hk, D, device=dev).to(torch.float8_e4m3fn)
q = torch.randn(7, Hk * G, D, device=dev, dtype=torch.bfloat16)
slots = torch.randint(-1, N, (7, 2051), device=dev, dtype=torch.int32); slots[3] = -1
a = ref_attn(q, kc, vc, slots, 0.0625); b = qsa_xpu.qsa_sparse_attention_reference(q, kc, vc, slots, 0.0625)
err = ((a.float() - b.float()).abs().max() / a.float().abs().max()).item()
print("attn rel max err", err, "empty row zero:", b[3].abs().max().item() == 0)
bad += err > 2e-2
for R in (7, 300):
    qR = torch.randn(R, Hk * G, D, device=dev, dtype=torch.bfloat16)
    # QSA rows select DISTINCT positions (expanded complete blocks + disjoint tail); -1 padding at the end
    sl = torch.stack([torch.randperm(N, device=dev)[:2051] for _ in range(R)]).to(torch.int32)
    sl[:, 1800:] = -1; sl[3] = -1
    a = ref_attn(qR, kc, vc, sl, 0.0625); c = qsa_xpu.qsa_sparse_attention_union(qR, kc, vc, sl, 0.0625)
    e2 = ((a.float() - c.float()).abs().max() / a.float().abs().max()).item()
    print("union attn R", R, "rel max err", e2, "empty row zero:", c[3].abs().max().item() == 0)
    bad += e2 > 2e-2
# fused decode selection vs torch MQA decode + top-k (same selected sets)
from sglang.srt.layers.attention.qsa import mqa as MQ
from exl3xpu.moe_offload import ops
X = ops()
for L_, pages in ((100, 1024), (3000, 1024), (60000, 1024)):
    ps = 64
    cache = torch.randn(pages + 8, ps, 1, 128, device=dev).to(torch.bfloat16)
    pt = torch.randperm(pages, device=dev)[:pages].to(torch.int32).view(1, -1)
    lens = torch.tensor([L_], device=dev, dtype=torch.int32)
    qd = torch.zeros(1, 8, 128, device=dev, dtype=torch.bfloat16); qd[:, :4] = torch.randn(1, 4, 128, device=dev)
    lg = MQ.torch_qsa_mqa_decode(qd, cache, pt, lens, pages * ps)
    a = qsa_xpu.qsa_fast_topk(lg, torch.zeros_like(lens), lens, 512)
    scratch = torch.empty(1, pages * ps, device=dev); o = torch.empty(1, 512, dtype=torch.int32, device=dev)
    X.qsa_decode_select(qd, cache, pt, lens, scratch, o, 512)
    sa, so = set(a[0][a[0] >= 0].tolist()), set(o[0][o[0] >= 0].tolist())
    print("decode select len", L_, "same set:", sa == so, len(sa), len(so), "sym diff", len(sa ^ so))
    bad += len(sa ^ so) > 2          # bf16 score ties at the boundary may swap one entry
sys.exit(bad)
