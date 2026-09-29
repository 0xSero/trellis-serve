"""Cost of SGLang's torch QSA decode indexer path on XPU at graph shapes (page table sized for the max context)."""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sglang.srt.layers.attention.qsa import mqa as M
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from exl3xpu import qsa_xpu
dev = torch.device("xpu", 0)
for dt in (torch.bfloat16,):
    for pages in (1024, 4096):
        ps = 64
        cache = torch.randn(pages + 8, ps, 1, 128, device=dev).to(dt)
        pt = torch.arange(pages, device=dev, dtype=torch.int32).view(1, -1)
        lens = torch.tensor([100], device=dev, dtype=torch.int32)
        q = torch.randn(1, 8, 128, device=dev, dtype=dt)
        def step():
            lg = M.torch_qsa_mqa_decode(q, cache, pt, lens, pages * ps)
            bi = qsa_xpu.qsa_fast_topk(lg, torch.zeros_like(lens), lens, 512)
            return expand_qsa_block_indices(bi, torch.tensor([399], device=dev), torch.tensor([400], device=dev), 4, 2048)
        for _ in range(5): step()
        torch.xpu.synchronize(); t = time.perf_counter()
        for _ in range(50): step()
        torch.xpu.synchronize(); dt_ = (time.perf_counter() - t) / 50 * 1e3
        print(f"page table {pages} x {ps} = {pages*ps} compressed keys: {dt_:.3f} ms per QSA layer (eager, incl. launches)")
