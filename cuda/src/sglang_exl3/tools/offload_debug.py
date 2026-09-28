"""Debug the device-driven CPU MoE handoff step by step (1 layer), printing the shared control words with bounded waits."""
import os
import sys
import time
import types

import numpy as np
import torch


def main():
    model = sys.argv[1]
    from exllamav3.model import moe_cpu_host as mch
    from exllamav3.model.moe_cpu_host import MoeCpuHost
    from safetensors import safe_open
    import json
    from sglang_exl3.offload import cpu_moe
    dev = torch.device("cuda")
    E, topk, H = 512, 10, 2560
    if "--ngram" in sys.argv:
        from sglang_exl3.offload.ngram_host import Exl3NgramHostTable
        tab = Exl3NgramHostTable(model)
        y = tab(torch.randint(0, tab.num_rows, (4, 16), device=dev))
        torch.cuda.synchronize()
        print("ngram table loaded + used", y.shape, flush=True)
    host = MoeCpuHost(types.SimpleNamespace(directory=model, infer_params=types.SimpleNamespace(
        moe_cpu_component="text", moe_cpu_threads=None)))
    idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    pre = "model.language_model.layers.0.mlp.experts"
    keys = {r: [f"{pre}.{e}.{r}_proj" for e in range(E)] for r in ("gate", "up", "down")}
    pd = dict(g=(2560, 640, 3), u=(2560, 640, 3), d=(640, 2560, 3))
    L = host.register_layer(pre, keys["gate"], keys["up"], keys["down"], 0, 0.0, H, H, topk, proj_dims=pd, aux=None)
    host.ensure_started()
    d = cpu_moe._DevPath(host, dev)
    cpu_moe._STATE.update(host=host, dev=d)
    u32 = np.frombuffer(host.shm.buf, dtype=np.uint32)
    F = mch.MOE_SLOT_FLAGS_OFFSET // 4

    def show(tag):
        print(f"{tag}: devseq {u32[1]} abort {u32[32]} tail {u32[64]} head {u32[80]} | job0 {u32[96:103].tolist()} "
              f"job1 {u32[96+264:103+264].tolist()} | data_ready0 {u32[F]} done0 {u32[F + 16*8]} consumed0 {u32[F + 32*8]} "
              f"| host.seq {host.seq}", flush=True)

    def wait(tag, limit=20):
        ev = torch.cuda.Event(); ev.record()
        t0 = time.time()
        while not ev.query():
            if time.time() - t0 > limit:
                show(tag + " TIMEOUT")
                return False
            time.sleep(0.01)
        show(tag + f" done in {time.time() - t0:.3f}s")
        return True

    show("start")
    x = torch.randn((1, H), device=dev).to(torch.bfloat16)
    ids = torch.randperm(E, device=dev)[:topk].view(1, topk).to(torch.int32)
    w = torch.full((1, topk), 0.1, device=dev)

    class _L:
        exl3_cpu_idx = L
    meth = cpu_moe.Exl3CpuMoEMethod.__new__(cpu_moe.Exl3CpuMoEMethod)
    meth.prefix = "dbg"
    a = meth._device_path(_L, x, ids, w)
    if not wait("device path 1"):
        os._exit(1)
    print("dev out", a.float().abs().mean().item(), flush=True)
    a2 = meth._device_path(_L, x, ids, w)
    if not wait("device path 2"):
        os._exit(1)
    print("dev out2 eq", torch.equal(a, a2), flush=True)
    host.seq = int(d.devseq[0]); host.slot_last_seq = [0] * len(host.slot_last_seq)
    host.begin_pass()
    show("before host")
    b = host.submit(L, x.half(), ids.long(), w.half())
    show("host enqueued")
    if not wait("host path"):
        os._exit(1)
    d.devseq[0] = host.seq
    print("host out eq dev", torch.equal(b.to(torch.bfloat16), a), (b.float() - a.float()).abs().max().item(), flush=True)
    a3 = meth._device_path(_L, x, ids, w)
    if not wait("device path 3 after host"):
        os._exit(1)
    print("dev3 eq", torch.equal(a3, a), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
