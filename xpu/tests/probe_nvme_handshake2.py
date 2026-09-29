"""X008 debug 2: which direction of the in-kernel handshake is not visible (no host sync in between)?"""
import ctypes, sys, os, time, threading, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exl3xpu.ngram_nvme import Exl3NgramNvmeTable
dev = torch.device("xpu", 0)
nv = Exl3NgramNvmeTable("/model", ram_gb=1.0, device=dev, start_service=False)
ctrl = (ctypes.c_uint32 * 8).from_address(nv.ctrl_dev & ((1 << 64) - 1))
ids = torch.randint(0, nv.num_rows, (1, 16), dtype=torch.int64, device=dev)
# A: GPU -> host visibility without a sync
nv.ext.nvme_publish(ids.reshape(-1), nv.req_dev, nv.ctrl_dev, nv.dseq)
for k in range(3):
    time.sleep(0.1)
    print("A t=%.1fs ctrl %s" % (0.1 * (k + 1), list(ctrl)[:4]), flush=True)
# B: host -> GPU visibility while the wait kernel spins (host writes done 0.5 s after launch)
slots = nv.slots_dev[:16]
ctrl[3] = 0
t = time.perf_counter()
nv.ext.nvme_wait(nv.ctrl_dev, nv.dseq, nv.stall, nv.resp_dev, slots, 16, int(float(os.environ.get("TO_NS", "2e8"))))
time.sleep(0.5)
ctrl[2] = 1
if os.environ.get("FLUSH", "1") == "1":
    nv.ext.host_flush(nv.ctrl_dev + 8, 4)
print("B host wrote done=1 at %.3f s" % (time.perf_counter() - t), flush=True)
torch.xpu.synchronize()
print("B wait returned after %.3f s stall %s err %d" % (time.perf_counter() - t, nv.stall.tolist(), ctrl[3]), flush=True)
os._exit(0)
