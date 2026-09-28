"""Parity of the Marlin kernel (`kernels/marlin`, csrc/) against ExLlamaV3 on real checkpoint tensors.

    PYTHONPATH=src:csrc/build/lib \
        python -m sglang_exl3.parity.marlin_parity /models/<checkpoint> [--json out.json]

One tensor per (k, n, K, codebook) class the kernel supports. Three checks each:

 (a) LOSSLESS DECODE: the weight matrix the kernel multiplies with, read out with identity inputs through the
     rotated-basis GEMM (1.0 * w + exact zeros, so the fp32 accumulator holds w exactly), must equal
     `exllamav3_ext.reconstruct` BIT FOR BIT. Done for every compiled row-block family (8 = transposed MMA,
     16, and 32/48/64 rows per launch when built), so every decode instance is covered. The load-time repack is
     checked to be an invertible permutation.
 (b) OUTPUTS, rows in {1,2,3,4,8,16}: error vs a float64 reference for our kernel and for ExLlamaV3's own paths
     (default dispatch = int8 GEMV at rows <= 2 / fp16 GEMV / GEMM; pinned fp16 GEMM; reconstruct + cuBLAS), and
     our distance to each. Gate: our error vs float64 <= 1.25 x the worst ExLlamaV3 fp16 path on the same input,
     i.e. within ExLlamaV3's own kernel-vs-kernel spread.
 (c) CUDA graph: capture + replay on fresh input == eager, bit for bit; 20 repeated calls identical.

 (d) ADVERSARIAL inputs (4 rows each): one huge activation per 128-block (|x * suh| = 3e4), all zeros, fp16
     denormals, mixed huge/denormal. Gate: no more non-finite outputs than ExLlamaV3's fp16 paths on the same input
     (zeros must give exact zeros), error vs float64 on the finite entries <= 1.25 x the worst ExLlamaV3 fp16 path.
 (f) the output Hadamard inside the GEMM launch must equal the separate output launch BIT FOR BIT (fp16 and bf16);
     so must the experimental cooperative launch with the input Hadamard as a prologue (knob, default off).
 (g) the many-row (prefill) path `runtime.ops.dense_group` must equal the explicit-cast reference dense path BIT FOR BIT.
 (e) bf16 boundary: `linear(x_bf16) -> bf16` must equal `linear(x_bf16.half()).bfloat16()` BIT FOR BIT (NaN-aware),
     on normal inputs and on bf16 inputs beyond the fp16 range (they become inf exactly as torch's cast makes them).

K = 3 (K3), K = 4, K = 5 (K5) and K = 6 (lm_head) classes. Fused groups (q/k/v, gate/up, GDN qkv/z: one launch for several matrices that share the input) get the same three
checks (`--no-groups` skips them): the decoded weights of every shard through the fused launch, outputs against
float64 and against ExLlamaV3's per-matrix kernels, graph replay.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

# With more than one block per SM (SGLANG_EXL3_BLOCKS_PER_SM, the 3090 default) the separate output-transform launch and the
# in-launch transform read the split-K partials in a different order at the rows>=16 families: both stay within the same
# error of float64 (checked below), but they are not bit-equal, so that comparison is informational at bps != 1.
_BPS = int(os.environ.get("SGLANG_EXL3_BLOCKS_PER_SM", "1"))
from safetensors import safe_open

from trellis_core.format import load_manifest
from ..kernels import marlin, reference
from ..runtime import ops
from .l1_linear import _block_had, _sha

ROWS = [1, 2, 3, 4, 8, 16, 32, 64]
_CB = {"3inst": 0, "mcg": 1, "mul1": 2}


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality of two 16-bit float tensors, any NaN == any NaN (payloads are not part of the contract)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    same = (a.view(torch.int16) == b.view(torch.int16)) | (nan_a & nan_b)
    return bool(same.all())


def adversarial_inputs(k: int, suh: torch.Tensor, gen) -> dict[str, torch.Tensor]:
    rows = 4
    base = (torch.randn((rows, k), generator=gen, dtype=torch.float32) * 0.5).cuda()
    s = suh.float().abs().clamp_min(1e-4)
    huge = base.clone()
    idx = torch.arange(0, k, 128, device="cuda") + torch.randint(0, 128, (k // 128,), generator=gen).cuda()
    huge[:, idx] = (3e4 / s[idx]).clamp_max(6e4) * torch.sign(base[:, idx])
    den = (torch.rand((rows, k), generator=gen, dtype=torch.float32).cuda() * 5e-5 + 1e-7) * torch.sign(base)
    mixed = torch.where(torch.rand((rows, k), generator=gen).cuda() < 0.5, den, base)
    mixed[:, idx[::2]] = huge[:, idx[::2]]
    return {"huge": huge.half(), "zeros": torch.zeros((rows, k), dtype=torch.float16, device="cuda"),
            "denormal": den.half(), "mixed": mixed.half()}


def check_adversarial(x, exact, y, refs: dict) -> dict:
    """y = ours, refs = ExLlamaV3 fp16 paths. Errors are measured where the float64 result fits fp16 and all paths are finite."""
    fits = exact.abs() < 65000
    finite_all = torch.isfinite(y) & fits
    for r in refs.values():
        finite_all &= torch.isfinite(r)
    denom = max(exact[finite_all].abs().mean().item(), 1e-30) if finite_all.any() else 1.0
    err = lambda t: ((t.double() - exact)[finite_all].abs().mean().item() / denom) if finite_all.any() else 0.0
    out = {"nonfinite_marlin": int((~torch.isfinite(y) & fits).sum()),
           "nonfinite_exl3_worst": max(int((~torch.isfinite(r) & fits).sum()) for r in refs.values()),
           "err64_marlin": err(y), "err64_exl3_worst": max(err(r) for r in refs.values())}
    out["pass"] = (out["nonfinite_marlin"] <= out["nonfinite_exl3_worst"]
                   and out["err64_marlin"] <= 1.25 * out["err64_exl3_worst"] + 1e-12)
    if not x.any():
        out["pass"] = out["pass"] and not bool(y.any())
    return out


def check_bf16(fn, x16: torch.Tensor, gen) -> bool:
    """fn(x, out_dtype) -> y. bf16 in/out must equal explicit torch casts around the fp16 path, bit for bit."""
    xb = x16.to(torch.bfloat16)
    wild = xb.clone()                                   # bf16 values outside the fp16 range, inf and tiny ones
    flat = wild.view(-1)
    pick = torch.randperm(flat.numel(), generator=gen)[:64].cuda()
    vals = torch.tensor([7e4, -7e4, 65519.0, 65520.0, -65520.0, 3e38, 1e-30, -1e-42], dtype=torch.float32).cuda()
    flat[pick] = vals.repeat(8).to(torch.bfloat16)
    ok = True
    for x in (xb, wild):
        want = fn(x.to(torch.float16), torch.float16).to(torch.bfloat16)
        ok = ok and bits_equal(fn(x, torch.bfloat16), want)
        ok = ok and bits_equal(fn(x, torch.float16).to(torch.bfloat16), want)      # bf16 in, fp16 out
    return ok


def check_inlaunch(fn, x16: torch.Tensor) -> bool:
    """fn(x, out_dtype) -> y. Output transform inside the GEMM launch == separate output launch, bit for bit."""
    ok = True
    for dt in (torch.float16, torch.bfloat16):
        marlin.set_out_had_inlaunch(False)
        want = fn(x16.to(dt), dt)
        marlin.set_out_had_inlaunch(True)
        ok = ok and bits_equal(fn(x16.to(dt), dt), want)
        marlin.set_in_had_inlaunch(True)       # experimental knob (default off): cooperative launch with the input
        try:                                   # transform as a grid-barrier prologue; must give the same bits too
            ok = ok and bits_equal(fn(x16.to(dt), dt), want)
        finally:
            marlin.set_in_had_inlaunch(False)
    return ok


def check_dense(trellis: list, suhs: list, svhs: list, cb: int, gen) -> bool:
    """Many-row path (runtime.ops.dense_group: our batched bf16 transforms around reconstruct + cuBLAS, written straight
    into the fused output) == the explicit-cast reference path it replaces, bit for bit, fp16 and bf16, 96 rows."""
    k = trellis[0].shape[0] * 16
    ns = [t.shape[1] * 16 for t in trellis]
    if max(ns) > 65536:
        return True                                   # served by the column-sliced reference path
    x16 = (torch.randn((96, k), generator=gen, dtype=torch.float32) * 0.5).to(torch.float16).cuda()
    suh_cat = torch.stack(suhs).contiguous()
    ok = True
    for dt in (torch.float16, torch.bfloat16):
        x = x16.to(dt)
        want = torch.cat([reference.dense_forward(x.to(torch.float16), t, su, sv, cb, max_weight_bytes=1 << 31).to(dt)   # unsliced: the bit-for-bit reference
                          for t, su, sv in zip(trellis, suhs, svhs)], dim=1)
        out = torch.empty((96, sum(ns)), dtype=dt, device="cuda")
        ok = ok and bits_equal(ops.dense_group(x, trellis, suh_cat, svhs, cb, out), want)
    return ok


def decoded_weight(packed: torch.Tensor, k: int, n: int, cb: int, chunk: int) -> torch.Tensor:
    w = torch.empty((k, n), dtype=torch.float16, device=packed.device)
    a = torch.zeros((chunk, k), dtype=torch.float16, device=packed.device)
    c = torch.empty((chunk, n), dtype=torch.float16, device=packed.device)
    for r0 in range(0, k, chunk):
        ar = torch.arange(min(chunk, k - r0), device=packed.device)   # the launch always has `chunk` rows
        a[ar, r0 + ar] = 1.0
        marlin.gemm_rotated(a, packed, cb, c)
        w[r0:r0 + len(ar)] = c[:len(ar)]
        a[ar, r0 + ar] = 0.0
    return w


def check(model_dir: str, key: str, spec, chunks) -> dict:
    t = {}
    for suffix, info in spec.tensors.items():
        with safe_open(os.path.join(model_dir, info.file), "pt") as f:
            t[suffix] = f.get_tensor(info.name).cuda()
    cb = _CB[spec.codebook.value]
    trellis, suh, svh = t["trellis"].contiguous(), t["suh"], t["svh"]
    k, n = spec.k, spec.n
    w_ref = reference.reconstruct(trellis, cb)
    packed = marlin.prepare_matrix(trellis)
    out = {"key": key, "k": k, "n": n, "K": spec.bits.value, "codebook": spec.codebook.value, "sha_w_hat": _sha(w_ref),
           "repack_invertible": bool(torch.equal(marlin.unprepare_matrix(packed), trellis)), "decode": {}, "rows": {}}
    for chunk in chunks:
        try:
            w = decoded_weight(packed, k, n, cb, chunk)
        except RuntimeError as e:
            if "not compiled" in str(e):
                out["decode"][chunk] = "kernel family not built"
                continue
            raise
        diff = int((w.view(torch.int16) != w_ref.view(torch.int16)).sum().item())
        out["decode"][chunk] = {"bit_identical": diff == 0, "mismatching_weights": diff, "of": w.numel()}
    w64 = w_ref.double()
    gen = torch.Generator().manual_seed(1234)
    for rows in ROWS:
        x = (torch.randn((rows, k), generator=gen, dtype=torch.float32) * 0.5).to(torch.float16).cuda()
        exact = _block_had(_block_had(x.double() * suh.double()) @ w64) * svh.double()
        ys = {"marlin": marlin.linear(x, packed, suh, svh, cb),
              "exl3_default": reference.gemm(x, trellis, suh, svh, cb),
              "exl3_gemm_fp16": reference.gemm(x, trellis, suh, svh, cb, force_shape_idx=1),
              "exl3_recon_cublas": reference.dense_forward(x, trellis, suh, svh, cb)}
        scale, rms = exact.abs().mean().item(), exact.pow(2).mean().sqrt().item()
        mean_rel = lambda a, b: (a.double() - b.double()).abs().mean().item() / scale
        max_rel = lambda a, b: (a.double() - b.double()).abs().max().item() / rms
        r = {"finite": bool(torch.isfinite(ys["marlin"]).all())}
        for name, y in ys.items():
            r[f"err64_mean_{name}"] = mean_rel(y, exact)
            r[f"err64_max_{name}"] = max_rel(y, exact)
        for name in ("exl3_default", "exl3_gemm_fp16", "exl3_recon_cublas"):
            r[f"marlin_vs_{name}_mean"] = mean_rel(ys["marlin"], ys[name])
        r["exl3_spread_mean"] = mean_rel(ys["exl3_gemm_fp16"], ys["exl3_recon_cublas"])
        worst_ref = max(r["err64_mean_exl3_gemm_fp16"], r["err64_mean_exl3_recon_cublas"])
        r["pass"] = r["finite"] and r["err64_mean_marlin"] <= 1.25 * worst_ref
        # (c) graph replay == eager, and determinism
        xs, xh, yg = x.clone(), torch.empty_like(x), torch.empty((rows, n), dtype=torch.float16, device="cuda")
        marlin.linear(xs, packed, suh, svh, cb, xh, yg)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            marlin.linear(xs, packed, suh, svh, cb, xh, yg)
        x2 = (torch.randn((rows, k), generator=gen, dtype=torch.float32) * 0.5).to(torch.float16).cuda()
        xs.copy_(x2)
        g.replay()
        torch.cuda.synchronize()
        eager = marlin.linear(x2, packed, suh, svh, cb)
        r["graph_equals_eager"] = bool(torch.equal(yg, eager))
        r["deterministic"] = all(torch.equal(marlin.linear(x2, packed, suh, svh, cb), eager) for _ in range(20))
        def fn(xx, dt):
            yy = torch.empty((xx.shape[0], n), dtype=dt, device="cuda")
            return marlin.linear(xx, packed, suh, svh, cb, torch.empty(xx.shape, dtype=torch.float16, device="cuda"), yy)
        r["bf16_equals_casts"] = check_bf16(fn, x2, gen)
        r["inlaunch_equals_separate"] = check_inlaunch(fn, x2)
        r["pass"] = (r["pass"] and r["graph_equals_eager"] and r["deterministic"] and r["bf16_equals_casts"]
                     and (r["inlaunch_equals_separate"] or _BPS != 1))
        out["rows"][rows] = r
        del g
    out["adversarial"] = {}
    for name, xa in adversarial_inputs(k, suh, gen).items():
        exact = _block_had(_block_had(xa.double() * suh.double()) @ w64) * svh.double()
        refs = {"gemm_fp16": reference.gemm(xa, trellis, suh, svh, cb, force_shape_idx=1),
                "recon_cublas": reference.dense_forward(xa, trellis, suh, svh, cb)}
        out["adversarial"][name] = check_adversarial(xa, exact, marlin.linear(xa, packed, suh, svh, cb), refs)
    out["dense_equals_reference"] = check_dense([trellis], [suh], [svh], cb, gen)
    out["pass"] = (out["repack_invertible"] and out["dense_equals_reference"] and all(a["pass"] for a in out["adversarial"].values()) and all(v["bit_identical"] for v in out["decode"].values() if isinstance(v, dict))
                   and all(r["pass"] for r in out["rows"].values()))
    return out


GROUPS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj"),
          ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
          ("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"))   # K5: MoE models' shared expert


def _load(model_dir, spec):
    t = {}
    for suffix, info in spec.tensors.items():
        with safe_open(os.path.join(model_dir, info.file), "pt") as f:
            t[suffix] = f.get_tensor(info.name).cuda()
    t["trellis"] = t["trellis"].contiguous()
    return t


def check_group(model_dir: str, keys, man, chunks) -> dict:
    specs = [man.matrices[k] for k in keys]
    ts = [_load(model_dir, s) for s in specs]
    cb, k = _CB[specs[0].codebook.value], specs[0].k
    packed, suh_cat, svh_cat, ends = marlin.prepare([t["trellis"] for t in ts], [t["suh"] for t in ts],
                                                          [t["svh"] for t in ts])
    w_ref = torch.cat([reference.reconstruct(t["trellis"], cb) for t in ts], dim=1)
    n, shards = w_ref.shape[1], len(ts)
    out = {"keys": list(keys), "k": k, "n": [s.n for s in specs], "codebook": specs[0].codebook.value, "decode": {}, "rows": {}}
    for chunk in chunks:
        w = torch.empty_like(w_ref)
        a = torch.zeros((shards * chunk, k), dtype=torch.float16, device="cuda")
        for r0 in range(0, k, chunk):
            ar = torch.arange(min(chunk, k - r0), device="cuda")
            for s_ in range(shards):
                a[s_ * chunk + ar, r0 + ar] = 1.0
            w[r0:r0 + len(ar)] = marlin.gemm_rotated_group(a, packed, ends, cb)[:len(ar)]
            a.zero_()
        diff = int((w.view(torch.int16) != w_ref.view(torch.int16)).sum().item())
        out["decode"][chunk] = {"bit_identical": diff == 0, "mismatching_weights": diff, "of": w.numel()}
    gen = torch.Generator().manual_seed(4321)
    w64 = w_ref.double()
    for rows in ROWS:
        x = (torch.randn((rows, k), generator=gen, dtype=torch.float32) * 0.5).to(torch.float16).cuda()
        exact = torch.cat([_block_had(_block_had(x.double() * t["suh"].double()) @ w64[:, a0:a1]) * t["svh"].double()
                           for t, a0, a1 in zip(ts, [0] + ends, ends + [n])], dim=1)
        y = marlin.linear_group(x, packed, suh_cat, svh_cat, ends, cb)
        y_gemm = torch.cat([reference.gemm(x, t["trellis"], t["suh"], t["svh"], cb, force_shape_idx=1) for t in ts], dim=1)
        y_rec = torch.cat([reference.dense_forward(x, t["trellis"], t["suh"], t["svh"], cb) for t in ts], dim=1)
        scale = exact.abs().mean().item()
        mean_rel = lambda a_, b_: (a_.double() - b_.double()).abs().mean().item() / scale
        r = {"finite": bool(torch.isfinite(y).all()), "err64_mean_marlin": mean_rel(y, exact),
             "err64_mean_exl3_gemm_fp16": mean_rel(y_gemm, exact), "err64_mean_exl3_recon_cublas": mean_rel(y_rec, exact),
             "marlin_vs_exl3_gemm_fp16_mean": mean_rel(y, y_gemm), "marlin_vs_exl3_recon_cublas_mean": mean_rel(y, y_rec)}
        xs, xh = x.clone(), torch.empty((shards * rows, k), dtype=torch.float16, device="cuda")
        yg = torch.empty_like(y)
        marlin.linear_group(xs, packed, suh_cat, svh_cat, ends, cb, xh, yg)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            marlin.linear_group(xs, packed, suh_cat, svh_cat, ends, cb, xh, yg)
        x2 = (torch.randn((rows, k), generator=gen, dtype=torch.float32) * 0.5).to(torch.float16).cuda()
        xs.copy_(x2)
        g.replay()
        torch.cuda.synchronize()
        eager = marlin.linear_group(x2, packed, suh_cat, svh_cat, ends, cb)
        r["graph_equals_eager"] = bool(torch.equal(yg, eager))
        r["deterministic"] = all(torch.equal(marlin.linear_group(x2, packed, suh_cat, svh_cat, ends, cb), eager) for _ in range(20))
        worst_ref = max(r["err64_mean_exl3_gemm_fp16"], r["err64_mean_exl3_recon_cublas"])
        gfn = lambda xx, dt: marlin.linear_group(xx, packed, suh_cat, svh_cat, ends, cb,
                                                 out=torch.empty((xx.shape[0], n), dtype=dt, device="cuda"))
        r["bf16_equals_casts"] = check_bf16(gfn, x2, gen)
        r["inlaunch_equals_separate"] = check_inlaunch(gfn, x2)
        r["pass"] = (r["finite"] and r["err64_mean_marlin"] <= 1.25 * worst_ref and r["graph_equals_eager"]
                     and r["deterministic"] and r["bf16_equals_casts"] and (r["inlaunch_equals_separate"] or _BPS != 1))
        out["rows"][rows] = r
        del g
    out["adversarial"] = {}
    for name, xa in adversarial_inputs(k, ts[0]["suh"], gen).items():
        exact = torch.cat([_block_had(_block_had(xa.double() * t["suh"].double()) @ w64[:, a0:a1]) * t["svh"].double()
                           for t, a0, a1 in zip(ts, [0] + ends, ends + [n])], dim=1)
        refs = {"gemm_fp16": torch.cat([reference.gemm(xa, t["trellis"], t["suh"], t["svh"], cb, force_shape_idx=1) for t in ts], dim=1),
                "recon_cublas": torch.cat([reference.dense_forward(xa, t["trellis"], t["suh"], t["svh"], cb) for t in ts], dim=1)}
        out["adversarial"][name] = check_adversarial(xa, exact, marlin.linear_group(xa, packed, suh_cat, svh_cat, ends, cb), refs)
    out["dense_equals_reference"] = check_dense([t["trellis"] for t in ts], [t["suh"] for t in ts], [t["svh"] for t in ts], cb, gen)
    out["pass"] = (all(v["bit_identical"] for v in out["decode"].values()) and all(r["pass"] for r in out["rows"].values())
                   and all(a["pass"] for a in out["adversarial"].values()) and out["dense_equals_reference"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--keys", nargs="*", help="explicit module keys (default: one per supported shape class)")
    ap.add_argument("--skip", nargs="*", default=["visual", "mtp."])
    ap.add_argument("--chunks", type=int, nargs="+", default=[8, 16, 32, 48, 64])
    ap.add_argument("--no-groups", action="store_true")
    ap.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    man = load_manifest(args.model_dir)
    ROWS[:] = args.rows
    if args.keys:
        keys = args.keys
    else:
        classes = defaultdict(list)
        for key, m in man.matrices.items():
            if not any(s in key for s in args.skip) and m.bits.value in (3, 4, 5, 6) and m.k % 128 == 0 and m.n % 128 == 0:   # K3, K5
                classes[(m.k, m.n, m.bits.value, m.codebook.value)].append(key)
        keys = [v[len(v) // 2] for _, v in sorted(classes.items())]
    doc = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, "linears": []}
    ok = True
    for key in keys:
        res = check(args.model_dir, key, man.matrices[key], args.chunks)
        doc["linears"].append(res)
        ok = ok and res["pass"]
        dec = " ".join(f"{c}:{'OK' if isinstance(v, dict) and v['bit_identical'] else ('-' if not isinstance(v, dict) else 'FAIL')}"
                       for c, v in res["decode"].items())
        print(f"\n{res['k']}->{res['n']} K={res['K']:g} {res['codebook']} {key}\n  decoded W_hat bit-identical to reconstruct, rows/launch {dec}"
              f"   repack invertible: {res['repack_invertible']}")
        print("  rows | err vs fp64 (mean rel): marlin  exl3-default exl3-gemm  recon+cublas | marlin vs gemm / vs recon | exl3 gemm-vs-recon | graph det pass")
        for rows, r in res["rows"].items():
            print(f"  {rows:>4} | {r['err64_mean_marlin']:.2e}  {r['err64_mean_exl3_default']:.2e}  {r['err64_mean_exl3_gemm_fp16']:.2e}  "
                  f"{r['err64_mean_exl3_recon_cublas']:.2e} | {r['marlin_vs_exl3_gemm_fp16_mean']:.2e} / {r['marlin_vs_exl3_recon_cublas_mean']:.2e} | "
                  f"{r['exl3_spread_mean']:.2e} | {r['graph_equals_eager']} {r['deterministic']} bf16:{r['bf16_equals_casts']} inl:{r['inlaunch_equals_separate']} {r['pass']}", flush=True)
        print(f"  many-row path == reference dense path bit for bit: {res['dense_equals_reference']}")
        print("  adversarial (4 rows): " + "  ".join(
            f"{nm}: nonfinite {a['nonfinite_marlin']}/{a['nonfinite_exl3_worst']} err {a['err64_marlin']:.1e}/{a['err64_exl3_worst']:.1e} {'ok' if a['pass'] else 'FAIL'}"
            for nm, a in res["adversarial"].items()), flush=True)
    doc["groups"] = []
    if not args.no_groups and not args.keys:
        seen = set()
        for key in sorted(man.matrices):
            for grp in GROUPS:
                if grp in seen or not key.endswith(grp[0]) or any(s in key for s in args.skip):
                    continue
                prefix = key[: -len(grp[0])]
                gkeys = [prefix + g for g in grp]
                specs = [man.matrices.get(g) for g in gkeys]
                if any(s is None for s in specs) or any(s.bits.value not in (3, 4, 5) or s.n % 128 or s.k % 128 for s in specs):
                    continue   # K3: fused groups at K = 3 or 4; K5: or 5
                if len({s.bits.value for s in specs}) != 1:
                    continue   # K3: one bitrate per fused group
                seen.add(grp)
                res = check_group(args.model_dir, gkeys, man, [c for c in args.chunks if c in (8, 16, 64)])
                doc["groups"].append(res)
                ok = ok and res["pass"]
                dec = " ".join(f"{c}:{'OK' if v['bit_identical'] else 'FAIL'}" for c, v in res["decode"].items())
                print(f"\nFUSED GROUP {res['k']}->{res['n']} {prefix}{{{', '.join(g.split('.')[-1] for g in grp)}}}"
                      f"\n  decoded W_hat of all shards bit-identical through the fused launch, rows/launch {dec}")
                print("  rows | err vs fp64 (mean rel): marlin  exl3-gemm  recon+cublas | marlin vs gemm / vs recon | graph det pass")
                for rows, r in res["rows"].items():
                    print(f"  {rows:>4} | {r['err64_mean_marlin']:.2e}  {r['err64_mean_exl3_gemm_fp16']:.2e}  "
                          f"{r['err64_mean_exl3_recon_cublas']:.2e} | {r['marlin_vs_exl3_gemm_fp16_mean']:.2e} / "
                          f"{r['marlin_vs_exl3_recon_cublas_mean']:.2e} | {r['graph_equals_eager']} {r['deterministic']} bf16:{r['bf16_equals_casts']} inl:{r['inlaunch_equals_separate']} {r['pass']}", flush=True)
                print(f"  many-row path == reference dense path bit for bit: {res['dense_equals_reference']}")
                print("  adversarial (4 rows): " + "  ".join(
                    f"{nm}: nonfinite {a['nonfinite_marlin']}/{a['nonfinite_exl3_worst']} err {a['err64_marlin']:.1e}/{a['err64_exl3_worst']:.1e} {'ok' if a['pass'] else 'FAIL'}"
                    for nm, a in res["adversarial"].items()), flush=True)
    doc["pass"] = ok
    print(f"\nMARLIN_PARITY {'PASS' if ok else 'FAIL'}")
    if args.json:
        json.dump(doc, open(args.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
