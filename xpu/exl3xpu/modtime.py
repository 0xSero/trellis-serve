"""Diagnostics (EXL3_MODTIME=1): synchronised wall time per component class during model forwards, printed to stderr
after every forward with >= EXL3_MODTIME_MIN_TOKENS rows. Adds syncs: never for measurements of totals."""
from __future__ import annotations

import importlib
import os
import sys
import time
from collections import defaultdict

import torch

_ACC = defaultdict(float)
_CNT = defaultdict(int)
_DEPTH = [0]
_MIN = int(os.environ.get("EXL3_MODTIME_MIN_TOKENS", "256"))


def _wrap(cls, meth, tag):
    orig = getattr(cls, meth)
    if getattr(orig, "_exl3_mt", False):
        return

    def w(self, *a, **k):
        if _DEPTH[0] > 0:                        # nested timed calls count in the outer one only
            return orig(self, *a, **k)
        torch.xpu.synchronize()
        t = time.perf_counter()
        _DEPTH[0] += 1
        try:
            return orig(self, *a, **k)
        finally:
            _DEPTH[0] -= 1
            torch.xpu.synchronize()
            _ACC[tag] += time.perf_counter() - t
            _CNT[tag] += 1
    w._exl3_mt = True
    setattr(cls, meth, w)


def install() -> None:
    targets = [
        ("sglang.srt.layers.moe.fused_moe_triton.layer", "FusedMoE", "forward", "moe_routed"),
        ("sglang.srt.layers.attention.qsa.qsa_indexer", "QSAIndexer", "forward", "qsa_indexer"),
        ("sglang.srt.layers.attention.qwen_sparse_attn_backend", "QwenSparseAttnBackend", "forward_extend", "qsa_attn_extend"),
        ("sglang.srt.layers.attention.qwen_sparse_attn_backend", "QwenSparseAttnBackend", "forward_decode", "qsa_attn_decode"),
        ("sglang.srt.layers.hyperconnection", "GatedResidual", "forward", "hyperconnection"),
    ]
    for mod, cls, meth, tag in targets:
        try:
            c = getattr(importlib.import_module(mod), cls)
            _wrap(c, meth, tag)
        except Exception as e:
            print(f"EXL3_MODTIME: {mod}.{cls}.{meth} not wrapped ({e})", file=sys.stderr, flush=True)
    # every nn.Module class in qwen4_exp / qwen3_5 whose name mentions the GDN / PLE / linear-attention parts
    for mod in ("sglang.srt.models.qwen4_exp", "sglang.srt.models.qwen3_5"):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for name in dir(m):
            c = getattr(m, name)
            if isinstance(c, type) and issubclass(c, torch.nn.Module) and any(
                    s in name for s in ("GatedDeltaNet", "PLE", "NGram", "SharedExpert", "LinearDecoderLayer")):
                if "forward" in c.__dict__ and "DecoderLayer" not in name:
                    _wrap(c, "forward", name)
    from . import sglang_plugin as sp
    Lin = sp.classes()[1]
    _wrap(Lin, "apply", "exl3_dense_linear")
    try:
        from sglang.srt.models import qwen4_exp as q
        base = getattr(q, "Qwen4ExpForConditionalGeneration")
        orig = base.forward

        def fwd(self, *a, **k):
            _ACC.clear(); _CNT.clear()
            torch.xpu.synchronize()
            t = time.perf_counter()
            out = orig(self, *a, **k)
            torch.xpu.synchronize()
            total = time.perf_counter() - t
            fb = k.get("forward_batch") or (a[2] if len(a) > 2 else None)
            n = int(getattr(fb, "input_ids", torch.empty(0)).numel()) if fb is not None else -1
            if n >= _MIN:
                parts = " ".join(f"{k_}={v * 1e3:.1f}ms/{_CNT[k_]}" for k_, v in sorted(_ACC.items(), key=lambda x: -x[1]))
                print(f"EXL3_MODTIME tokens={n} total={total * 1e3:.1f}ms {parts} other={1e3 * (total - sum(_ACC.values())):.1f}ms",
                      file=sys.stderr, flush=True)
            return out
        base.forward = fwd
    except Exception as e:
        print(f"EXL3_MODTIME: model forward not wrapped ({e})", file=sys.stderr, flush=True)
