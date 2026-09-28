import torch, time, sys
sys.path.insert(0, "/ft/kernels/xpu_bmg/trellis-serve/xpu")
from exl3xpu.moe_offload import ops, s64
X = ops()
dev = torch.device("xpu", 0)
A = torch.randn((2048, 2048), device=dev, dtype=torch.bfloat16)
def gemms(n):
    for _ in range(n): A @ A
host = X.host_alloc(512 * 2**20)
dst = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
src_dev = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
def pcie_kernel():   # a kernel-driven host read: gather-copy via torch (device kernel reading host USM is not available in torch) -> use our zero-copy read through copy_ of a host tensor? use memcpy on compute stream instead
    X.memcpy_async(s64(dst.data_ptr()), s64(host.data_ptr()), host.numel())
def dev_kernel():
    dst.copy_(src_dev)          # device->device copy kernel/engine
for _ in range(3): gemms(20); pcie_kernel(); dev_kernel()
torch.xpu.synchronize()
def t(f):
    torch.xpu.synchronize(); t0 = time.perf_counter(); f(); torch.xpu.synchronize(); return (time.perf_counter() - t0) * 1e3
side = torch.xpu.Stream()
g = t(lambda: gemms(40))
m = t(pcie_kernel)
def both_memcpy():
    with torch.xpu.stream(side): pcie_kernel()
    gemms(40)
b = t(both_memcpy)
d = t(dev_kernel)
def both_dev():
    with torch.xpu.stream(side):
        for _ in range(4): dev_kernel()
    gemms(40)
bd = t(both_dev)
print(f"GEMMs alone {g:.2f} ms; H2D memcpy 512MiB alone {m:.2f} ms; concurrent (memcpy on side queue) {b:.2f} ms")
print(f"D2D copy_ 512MiB x4 alone {4*d:.2f} ms; concurrent with GEMMs (side queue) {bd:.2f} ms  (sum {g+4*d:.2f})")
