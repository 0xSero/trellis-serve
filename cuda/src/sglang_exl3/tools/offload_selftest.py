"""Self-test of the offload tiers on real Qwen3.8-Flash-Next tensors (no SGLang model):

  ngram: Exl3NgramHostTable (pinned host rows, zero-copy gather + decode kernel) == exllamav3 ext.ngram_dequant on the
         same packed rows (bit-exact after the bf16 cast), random + boundary row ids, CUDA graph replay == eager.
  moe:   exllamav3 MoeCpuHost with N real layers; the device-driven handoff (cpu_moe_issue/collect) == the stock host
         path (`submit`) bit for bit, both == a float64 torch reference within fp16 tolerance, CUDA-graph replay of the
         device path with fresh inputs == eager, host/device paths interleaved, streamed prefill (submit_prefill).

    python -m sglang_exl3.tools.offload_selftest /models/<ckpt> [--layers 2] [--skip-ngram] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def ngram_test(model, res):
    from exllamav3.ext import exllamav3_ext as ext
    from sglang_exl3.offload.ngram_host import Exl3NgramHostTable
    t0 = time.time()
    tab = Exl3NgramHostTable(model)
    res["ngram_load_s"] = round(time.time() - t0, 1)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    T = 4096
    ids = torch.randint(0, tab.num_rows, (T, 16), generator=g)
    ids[0] = torch.tensor([0, 1, tab.num_rows - 1, tab.num_rows - 2] * 4)
    ids_d = ids.to(dev)
    out = tab(ids_d)
    packed = tab.host[ids.reshape(-1)].to(dev)
    heads = torch.arange(16, dtype=torch.int32).repeat(T).to(dev)
    ref = torch.empty((T * 16, 160), dtype=torch.float16, device=dev)
    ext.ngram_dequant(packed, tab.K, heads, tab.head_bias, ref)
    same = torch.equal(out.reshape(-1, 160), ref.to(torch.bfloat16))
    res["ngram_bitexact_vs_ext"] = bool(same)
    res["ngram_max_abs_vs_fp16"] = float((out.reshape(-1, 160).float() - ref.float()).abs().max())
    # graph replay
    st = torch.randint(0, tab.num_rows, (8, 16), device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        tab(st)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        go = tab(st)
    ok = True
    for _ in range(3):
        st.copy_(torch.randint(0, tab.num_rows, (8, 16), device=dev))
        gr.replay()
        ok &= torch.equal(go, tab(st))
    res["ngram_graph_eq_eager"] = bool(ok)
    # decode-size latency
    one = torch.randint(0, tab.num_rows, (1, 16), device=dev)
    for _ in range(10):
        tab(one)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        tab(one)
    torch.cuda.synchronize()
    res["ngram_us_per_token_lookup"] = round((time.perf_counter() - t0) / 200 * 1e6, 1)
    x = torch.randint(0, tab.num_rows, (16384, 16), device=dev)
    tab(x); torch.cuda.synchronize()
    t0 = time.perf_counter()
    tab(x); torch.cuda.synchronize()
    res["ngram_ms_16k_tokens"] = round((time.perf_counter() - t0) * 1e3, 2)
    print("ngram:", {k: v for k, v in res.items() if k.startswith("ngram")}, flush=True)
    return tab      # kept alive like in the server (see Exl3NgramHostTable.release)


def moe_test(model, layers, res):
    import types
    import numpy as np
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.model.moe_cpu_host import MoeCpuHost
    from trellis_core.format import load_manifest
    from sglang_exl3.offload import cpu_moe
    dev = torch.device("cuda")
    man = load_manifest(model)
    E, topk, H = 512, 10, 2560
    host = MoeCpuHost(types.SimpleNamespace(directory=model, infer_params=types.SimpleNamespace(
        moe_cpu_component="text", moe_cpu_threads=int(os.environ.get("EXL3_MOE_CPU_THREADS", "0")) or None)))
    from safetensors import safe_open
    idx_file = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(key):
        fn = idx_file[key]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(model, fn), "pt", device="cpu")
        return handles[fn].get_tensor(key)
    t0 = time.time()
    lids = []
    for L in layers:
        pre = f"model.language_model.layers.{L}.mlp.experts"
        keys = {r: [f"{pre}.{e}.{r}_proj" for e in range(E)] for r in ("gate", "up", "down")}
        aux = {}
        for r, p in (("gate", "g"), ("up", "u"), ("down", "d")):
            for s in ("suh", "svh"):
                aux[f"{s}_{p}"] = [get(k + "." + s).to(dev, torch.float16) for k in keys[r]]
            aux[f"bias_{p}"] = None
        sh = {r: get(keys[r][0] + ".trellis").shape for r in keys}
        pd = {p: (sh[r][0] * 16, sh[r][1] * 16, sh[r][2] // 16) for r, p in (("gate", "g"), ("up", "u"), ("down", "d"))}
        lids.append(host.register_layer(f"{pre}", keys["gate"], keys["up"], keys["down"], 0, 0.0, H, H, topk,
                                        proj_dims=pd, aux=aux))
    host.ensure_started()
    def sync(tag, limit=90):
        ev = torch.cuda.Event(); ev.record()
        t0 = time.time()
        while not ev.query():
            if time.time() - t0 > limit:
                import numpy as _np
                from exllamav3.model import moe_cpu_host as _m
                u = _np.frombuffer(host.shm.buf, dtype=_np.uint32); F = _m.MOE_SLOT_FLAGS_OFFSET // 4
                raise RuntimeError(f"{tag}: GPU stream stuck > {limit}s: devseq {u[1]} abort {u[32]} tail {u[64]} head {u[80]} "
                                   f"data_ready {[int(u[F + 16 * s]) for s in range(4)]} done {[int(u[F + 128 + 16 * s]) for s in range(4)]} "
                                   f"consumed {[int(u[F + 256 + 16 * s]) for s in range(4)]} host.seq {host.seq}")
            time.sleep(0.005)

    res["moe_load_s"] = round(time.time() - t0, 1)
    cpu_moe._STATE["host"] = host
    d = cpu_moe._DevPath(host, dev)
    cpu_moe._STATE["dev"] = d
    meth = cpu_moe.Exl3CpuMoEMethod.__new__(cpu_moe.Exl3CpuMoEMethod)
    meth.prefix = "selftest"

    def rnd(t, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = (torch.randn((t, H), generator=g, device=dev) * 0.5).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, generator=g, device=dev)[:topk] for _ in range(t)]).to(torch.int32)
        w = torch.softmax(torch.randn((t, topk), generator=g, device=dev), -1).float()
        return x, ids, w

    def stock(L, x, ids, w):
        sync('stock-entry')      # same hand-over as Exl3CpuMoEMethod._host_path
        host.seq = int(d.devseq[0]); host.slot_last_seq = [0] * len(host.slot_last_seq)
        host.begin_pass()
        o = host.submit(L, x.to(torch.float16).contiguous(), ids.long(), w.to(torch.float16).contiguous())
        d.devseq[0] = host.seq
        return o

    def ref64(L, x, ids, w):
        # reference from the decoded weights (exllamav3 reconstruct + had_r_128), fp32 matmuls, fp16 activations
        pre = f"model.language_model.layers.{layers[L]}.mlp.experts"
        out = torch.zeros((x.shape[0], H), dtype=torch.float64, device=dev)
        cache = {}

        def W(key):
            if key not in cache:
                tr = get(key + ".trellis").to(dev)
                k, n = tr.shape[0] * 16, tr.shape[1] * 16
                wt = torch.empty((k, n), dtype=torch.float16, device=dev)
                ext.reconstruct(wt, tr, tr.shape[2] // 16, False, True)
                cache[key] = (wt.float(), get(key + ".suh").to(dev).half(), get(key + ".svh").to(dev).half())
            return cache[key]

        def lin(xv, key):
            wt, suh, svh = W(key)
            xh = torch.empty_like(xv)
            ext.had_r_128(xv, xh, suh, None, 1.0)
            y = (xh.float() @ wt).half()
            ext.had_r_128(y, y, None, svh, 1.0)
            return y
        xs = x.to(torch.float16)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                xv = xs[t:t + 1].contiguous()
                g_ = lin(xv, f"{pre}.{e}.gate_proj").float(); u_ = lin(xv, f"{pre}.{e}.up_proj").float()
                a = (torch.nn.functional.silu(g_) * u_).half()
                out[t] += float(w[t, j]) * lin(a, f"{pre}.{e}.down_proj")[0].double()
        return out

    class _L:
        pass
    lay = [_L() for _ in lids]
    for l, i in zip(lay, lids):
        l.exl3_cpu_idx = i
    ok_bit, rel = True, []
    for t in (1, 2, 7, 32):
        for li, l in enumerate(lay):
            x, ids, w = rnd(t, 100 * t + li)
            print('cmp', t, li, flush=True)
            a = meth._device_path(l, x, ids, w)
            b = stock(l.exl3_cpu_idx, x, ids, w).to(torch.bfloat16)
            sync('s1')
            ok_bit &= torch.equal(a, b)
            if t <= 2:
                r = ref64(li, x, ids, w)
                rel.append(float((a.double() - r).abs().mean() / r.abs().mean()))
    res["moe_dev_eq_stock_bitexact"] = bool(ok_bit)
    res["moe_rel_err_vs_fp64"] = [round(v, 5) for v in rel]
    # CUDA graph: capture the device path over all layers (a decode step's handoffs), replay with new inputs
    t = 4
    xs = [rnd(t, 7 + i) for i in range(len(lay))]
    stat = [(x.clone(), ids.clone(), w.clone()) for x, ids, w in xs]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (x, ids, w), l in zip(stat, lay):
            meth._device_path(l, x, ids, w)
    torch.cuda.current_stream().wait_stream(s)
    sync('s2')
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        gout = [meth._device_path(l, x, ids, w) for (x, ids, w), l in zip(stat, lay)]
    ok = True
    for rep in range(5):
        for i, (x, ids, w) in enumerate(stat):
            nx, ni, nw = rnd(t, 1000 + 10 * rep + i)
            x.copy_(nx); ids.copy_(ni); w.copy_(nw)
        gr.replay()
        sync('s3')
        for i, ((x, ids, w), l) in enumerate(zip(stat, lay)):
            ok &= torch.equal(gout[i], meth._device_path(l, x, ids, w))
        # interleave a stock host-path call between replays
        stock(lay[0].exl3_cpu_idx, *rnd(3, 5000 + rep))
    sync('s4')
    res["moe_graph_eq_eager_interleaved"] = bool(ok)
    # decode handoff latency per layer (graph replay of all layers / n layers)
    sync('s5')
    t0 = time.perf_counter()
    for _ in range(50):
        gr.replay()
    sync('s6')
    res["moe_dev_us_per_layer_4tok"] = round((time.perf_counter() - t0) / 50 / len(lay) * 1e6, 1)
    t = 1
    stat1 = [rnd(1, 77 + i) for i in range(len(lay))]
    gr1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr1):
        for (x, ids, w), l in zip(stat1, lay):
            meth._device_path(l, x, ids, w)
    sync('s7')
    t0 = time.perf_counter()
    for _ in range(100):
        gr1.replay()
    sync('s8')
    res["moe_dev_us_per_layer_1tok"] = round((time.perf_counter() - t0) / 100 / len(lay) * 1e6, 1)
    # streamed prefill through the host path (hot experts to the GPU, tail on the CPU)
    for T in (512, 4096):
        x, ids, w = rnd(T, 9)
        l = lay[0]
        y = meth._host_path(l, x, ids, w)
        sync('s9')
        t0 = time.perf_counter()
        y = meth._host_path(l, x, ids, w)
        sync('s10')
        res[f"moe_prefill_ms_{T}"] = round((time.perf_counter() - t0) * 1e3, 1)
        # compare with the plain CPU path on a subset of rows
        sub = stock(l.exl3_cpu_idx, x[:64], ids[:64], w[:64]).to(torch.bfloat16)
        res[f"moe_prefill_{T}_vs_cpu_rel"] = round(float((y[:64].float() - sub.float()).abs().mean() / sub.float().abs().mean()), 5)
    res["moe_abort_flag"] = int(host.v_abort[0])
    print("moe:", {k: v for k, v in res.items() if k.startswith("moe")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--skip-ngram", action="store_true")
    ap.add_argument("--skip-moe", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    torch.cuda.init()
    if not a.skip_ngram:
        keep = ngram_test(a.model, res)
    if not a.skip_moe:
        moe_test(a.model, list(range(a.layers)), res)
    ok = all(res.get(k, True) for k in ("ngram_bitexact_vs_ext", "ngram_graph_eq_eager", "moe_dev_eq_stock_bitexact",
                                          "moe_graph_eq_eager_interleaved")) and not res.get("moe_abort_flag")
    res["PASS"] = bool(ok)
    print("OFFLOAD_SELFTEST", "PASS" if ok else "FAIL", json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
