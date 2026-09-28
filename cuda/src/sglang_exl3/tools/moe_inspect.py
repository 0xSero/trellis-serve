"""Facts about the routed experts of an EXL3 MoE checkpoint (CPU only; safetensors headers + a few tensors).

    python -m sglang_exl3.tools.moe_inspect /models/Qwen3.6-35B-A3B-EXL3-3.00bpw-H5 [--layer 3] [--experts 16]

Prints: config, tensor names of one MoE block, per projection the trellis shape / K, whether suh / svh are shared
between experts, whether gate and up share suh inside one expert; then a whole-checkpoint manifest summary: K and
codebook per (projection class) over all layers, the shared expert, router, MTP and vision tensors.
(After trellis_exl3.tools.moe_inspect by the same author.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import torch
from safetensors import safe_open

from trellis_core.format import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, default=16)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    t = cfg.get("text_config", cfg)
    print("config:", {k: t.get(k) for k in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok",
                                            "moe_intermediate_size", "shared_expert_intermediate_size", "norm_topk_prob",
                                            "hidden_act", "mtp_num_hidden_layers")})
    print("quantization_config:", cfg.get("quantization_config") or t.get("quantization_config"))
    idx = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if f".layers.{a.layer}.mlp." in k]
    print(len(keys), f"tensors in the mlp of layer {a.layer}; non-expert + expert 0:")
    for k in sorted(k for k in keys if ".experts.0." in k or ".experts." not in k):
        print("   ", k)

    def get(name):
        with safe_open(os.path.join(a.model_dir, idx[name]), "pt") as f:
            return f.get_tensor(name)

    pre = [k for k in keys if k.endswith("experts.0.gate_proj.suh")][0].rsplit("experts.0.", 1)[0]
    for proj in ("gate_proj", "up_proj", "down_proj"):
        suh = torch.stack([get(f"{pre}experts.{e}.{proj}.suh") for e in range(a.experts)])
        svh = torch.stack([get(f"{pre}experts.{e}.{proj}.svh") for e in range(a.experts)])
        tr = get(f"{pre}experts.0.{proj}.trellis")
        print(proj, "trellis", tuple(tr.shape), tr.dtype, f"K={tr.shape[2] / 16:g}", "| suh", tuple(suh.shape), suh.dtype,
              "| suh shared across experts:", bool((suh == suh[0]).all()),
              "| svh shared:", bool((svh == svh[0]).all()),
              "| |suh| min/max", float(suh.abs().min()), float(suh.abs().max()),
              "| |svh| min/max", float(svh.abs().min()), float(svh.abs().max()))
    g, u = get(f"{pre}experts.0.gate_proj.suh"), get(f"{pre}experts.0.up_proj.suh")
    print("expert 0: gate suh == up suh:", bool((g == u).all()),
          "| sign agreement", float((g.sign() == u.sign()).float().mean()))

    print("\n--- manifest (safetensors headers of the whole checkpoint)")
    man = load_manifest(a.model_dir)
    print("summary:", man.summary())
    cls = Counter()
    for key, m in man.matrices.items():
        em = re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$", key)
        group = ("expert." + em.group(2)) if em else re.sub(r"\.\d+\.", ".N.", key)
        if key.startswith("mtp."):
            group = "mtp:" + group
        cls[(group, f"K={m.bits.value:g}", m.codebook.value, (m.k, m.n), m.has_bias)] += 1
    for (group, bits, cb, kn, bias), c in sorted(cls.items()):
        print(f"  {c:6d}  {group:60s} {bits} {cb} k,n={kn} bias={bias}")
    other = Counter()
    for name, info in man.unquantized.items():
        head = name.split(".")[0] if not name.startswith("model.") else ".".join(name.split(".")[:3])
        other[(re.sub(r"\.\d+\.", ".N.", head), str(info.dtype))] += 1
    print("unquantized tensor groups:")
    for (h, dt), c in sorted(other.items()):
        print(f"  {c:6d}  {h} {dt}")
    exp = sorted(n for n in man.unquantized if ".experts." in n)
    print("unquantized tensors under experts:", len(exp), exp[:5])
    mtp = sorted(n for n in list(man.unquantized) + list(man.matrices) if n.startswith("mtp."))
    print("mtp tensors:", len(mtp))
    for n in mtp[:40]:
        print("   ", n)


if __name__ == "__main__":
    main()
