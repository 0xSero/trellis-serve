"""Gate-spike hunt: is the EXL3 linear (large-M prefill path, as SGLang calls it) run-to-run deterministic?
Same input N times through ops.exl3_linear (all-C++ op) and exl3_linear_impl (python path); report rows that differ."""
import sys, os, json, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exl3xpu import ops
MODEL = os.environ.get("MODEL", "/model")
dev = torch.device("xpu:0")
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{MODEL}/{idx[n]}", "pt", device="cpu") as f:
        return f.get_tensor(n)
names = sys.argv[1:] or ["lm_head"]
N = int(os.environ.get("N", "30"))
for base in names:
    tr = get(base + ".trellis").to(dev); suh = get(base + ".suh").to(dev).unsqueeze(0); svh = get(base + ".svh").to(dev)
    K = tr.shape[2] // 16
    cb = 2 if base + ".mul1" in idx else 1 if base + ".mcg" in idx else 0
    n = svh.shape[0]
    shard = torch.zeros(n // 128, dtype=torch.int32, device=dev); bounds = [0, n]
    for M in [int(m) for m in os.environ.get("MS", "575,4096").split(",")]:
        torch.manual_seed(0)
        x = (torch.randn(M, suh.shape[1], device=dev) * 0.5).to(torch.bfloat16)
        for fn_name, fn in (("linear", ops.exl3_linear), ("impl", ops.exl3_linear_impl)):
            y0 = fn(x, tr, suh, svh, shard, bounds, K, cb).clone(); torch.xpu.synchronize()
            bad_runs = 0; bad_rows = set(); maxd = 0.0
            for r in range(N):
                y = fn(x, tr, suh, svh, shard, bounds, K, cb)
                d = (y.float() - y0.float()).abs().amax(1)
                rows = torch.nonzero(d > 0).flatten().tolist()
                if rows:
                    bad_runs += 1; bad_rows.update(rows[:20]); maxd = max(maxd, d.max().item())
            print(f"{base} K={K} cb={cb} n={n} M={M} {fn_name}: {bad_runs}/{N} runs differ from run 0, rows {sorted(bad_rows)[:12]}, max |d| {maxd:.3g}", flush=True)
