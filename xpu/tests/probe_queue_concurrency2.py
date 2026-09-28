import torch, time
dev = torch.device("xpu", 0)
a = torch.randn((256, 256), device=dev, dtype=torch.bfloat16)
b = torch.randn((256, 256), device=dev, dtype=torch.bfloat16)
def loop(x, n=400):
    for _ in range(n): x @ x
for _ in range(3): loop(a, 50); loop(b, 50)
torch.xpu.synchronize()
def t(f):
    torch.xpu.synchronize(); t0 = time.perf_counter(); f(); torch.xpu.synchronize(); return (time.perf_counter() - t0) * 1e3
s2 = torch.xpu.Stream()
one = t(lambda: loop(a))
def two_serial(): loop(a); loop(b)
ser = t(two_serial)
def two_conc():
    with torch.xpu.stream(s2): loop(b)
    loop(a)
conc = t(two_conc)
print(f"small GEMM loop x400: one stream {one:.2f} ms; two loops same stream {ser:.2f} ms; two streams {conc:.2f} ms")
