"""X008: NVMe n-gram tier (XPU publish/wait + RowStore) must return bit-identical rows to the USM-host table, incl. an
XPU-graph-captured lookup; timings for a decode token and a 4k/16k-token lookup (cold and warm cache)."""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exl3xpu.ngram_host import Exl3NgramHostTable
from exl3xpu.ngram_nvme import Exl3NgramNvmeTable
dev = torch.device("xpu", 0)
M = "/model"
nv = Exl3NgramNvmeTable(M, ram_gb=float(os.environ.get("RAM_GB", "8")), device=dev)
ho = Exl3NgramHostTable(M)
bad = 0
g = torch.Generator().manual_seed(0)
for T in (1, 7, 4096, 16384, 4096):
    ids = torch.randint(0, nv.num_rows, (T, 16), generator=g, dtype=torch.int64).to(dev)
    if T == 7:
        ids[0, 0] = 0; ids[0, 1] = nv.num_rows - 1
    torch.xpu.synchronize(); t = time.perf_counter()
    a = nv.gather(ids); torch.xpu.synchronize(); dt = (time.perf_counter() - t) * 1e3
    b = ho.gather(ids)
    eq = torch.equal(a.view(torch.int16), b.view(torch.int16)); bad += not eq
    print(f"T={T}: bit-identical {eq}  nvme gather {dt:.2f} ms (cold rows)", flush=True)
    torch.xpu.synchronize(); t = time.perf_counter()
    a2 = nv.gather(ids); torch.xpu.synchronize(); dt2 = (time.perf_counter() - t) * 1e3
    bad += not torch.equal(a2.view(torch.int16), b.view(torch.int16))
    print(f"      warm {dt2:.2f} ms", flush=True)
# graph capture of a decode lookup
ids_s = torch.randint(0, nv.num_rows, (1, 16), generator=g, dtype=torch.int64).to(dev)
out_s = torch.empty((1, 16, 160), dtype=torch.bfloat16, device=dev)
s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
with torch.xpu.stream(s):
    nv.gather(ids_s, out=out_s)
torch.xpu.current_stream().wait_stream(s); torch.xpu.synchronize()
gr = torch.xpu.XPUGraph()
with torch.xpu.graph(gr):
    nv.gather(ids_s, out=out_s)
for k in range(5):
    ids_s.copy_(torch.randint(0, nv.num_rows, (1, 16), generator=g, dtype=torch.int64).to(dev))
    gr.replay(); torch.xpu.synchronize()
    eq = torch.equal(out_s.view(torch.int16), ho.gather(ids_s).view(torch.int16)); bad += not eq
    print("graph replay", k, "bit-identical", eq, flush=True)
print("stats", {k: v for k, v in nv.stats().items() if k in ("lookups", "misses", "lookup_hit_rate", "bytes", "error", "gpu_waits")})
nv.release()
sys.exit(bad)
