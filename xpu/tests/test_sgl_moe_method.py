"""
X003: Exl3XpuMoEMethod (SGLang FusedMoE quant method, XPU) driven the way SGLang drives it, in the
lmsysorg/sglang:v0.5.20-xpu image:
  - Exl3XpuConfig.from_config on the real checkpoint (header scan)
  - create_weights on a module, checkpoint tensors of one MoE layer delivered through the registered weight_loaders
    using SGLang's own FusedMoE.make_expert_params_mapping (names, shard ids w1/w3/w2, expert ids)
  - create_moe_runner (SiLU gated), process_weights_after_loading (pack to host USM, N experts into device slots)
  - apply(layer, dispatch_output) vs the fp32 reference, for decode and prefill sizes; placement changes through
    ExpertStore.make_resident / evict between calls (pointer-table updates, no reload).

  EXL3_MOE_LIB=exl3xpu/_moe_sgl.so EXL3_MOE_SLOTS=600 EXL3_MOE_RESIDENT_PER_LAYER=256 \
    python3 tests/test_sgl_moe_method.py --layer 5
"""
import os, sys, json, time, argparse, types
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core", "src"))
from trellis_core import reference as ref

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "/model"))
ap.add_argument("--layer", type=int, default=5)
ap.add_argument("--out", default="")
args = ap.parse_args()

from exl3xpu import sglang_plugin as sp
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

Cfg, _, _ = sp.classes()
cfg = Cfg.from_config({"quant_method": "exl3", "hf_config": types.SimpleNamespace(_name_or_path=args.model)})
L = args.layer
prefix = f"model.language_model.layers.{L}.mlp.experts"
experts = {k: v for k, v in cfg.modules.items() if k.startswith(sp.norm_key(prefix) + ".")}
meth = sp.Exl3XpuMoEMethod(prefix, experts)
print(f"{prefix}: {len(meth.infos)} expert matrices from the header scan", flush=True)

layer = torch.nn.Module()
E, H, I = 512, 2560, 640
meth.create_weights(layer, E, H, I, torch.bfloat16)
mapping = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name="gate_proj", ckpt_down_proj_name="down_proj",
                                              ckpt_up_proj_name="up_proj", num_experts=E)
params = dict(layer.named_parameters())
idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
handles = {}
t0 = time.time()
n = 0
ckpt_prefix = f"model.language_model.layers.{L}.mlp."
for name, fn in idx.items():
    if not name.startswith(ckpt_prefix + "experts."):
        continue
    if fn not in handles:
        handles[fn] = safe_open(f"{args.model}/{fn}", "pt", device="cpu")
    tensor = handles[fn].get_tensor(name)
    local = name[len(ckpt_prefix):]                         # experts.E.gate_proj.trellis
    for param_name, weight_name, expert_id, shard_id in mapping:
        if weight_name not in local:
            continue
        pname = local.replace(weight_name, param_name)      # experts.w13_trellis
        pname = pname[len("experts."):]
        p = params[pname]
        p.weight_loader(p, tensor, name, shard_id=shard_id, expert_id=expert_id)
        n += 1
        break
    else:
        raise KeyError(name)
print(f"delivered {n} tensors through the loaders in {time.time() - t0:.1f}s", flush=True)
meth.create_moe_runner(layer, types.SimpleNamespace(activation="silu", is_gated=True, apply_router_weight_on_input=False,
                                                    routed_scaling_factor=None))
t0 = time.time()
meth.process_weights_after_loading(layer)
torch.xpu.synchronize()
store = layer.exl3_moe_store
print(f"process_weights_after_loading {time.time() - t0:.1f}s; resident {store.resident_count(meth.key)}", flush=True)

dev = torch.device("xpu", 0)
cache = {}


def ref_moe(x, ids, w):
    out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=dev)
    flat = ids.flatten().long().cpu()
    for e in torch.unique(flat).tolist():
        if e not in cache:
            W = {}
            for pr in ("gate_proj", "up_proj", "down_proj"):
                base = f"{ckpt_prefix}experts.{e}.{pr}."
                g = lambda s: handles[idx[base + s]].get_tensor(base + s).to(dev)   # noqa: E731
                W[pr] = ref.weight_orig(g("trellis"), g("suh"), g("svh"), 3, 2)
            cache[e] = W
        W = cache[e]
        pos = (flat == e).nonzero().flatten()
        rows, ks = pos // ids.shape[1], pos % ids.shape[1]
        xs = x.float()[rows.to(dev)]
        d = (F.silu(xs @ W["gate_proj"]) * (xs @ W["up_proj"])) @ W["down_proj"]
        out.index_add_(0, rows.to(dev), d * w.cpu()[rows, ks].to(dev).unsqueeze(1))
    return out


res = []
for phase in ("as_loaded", "after_evict_128_make_resident_64"):
    if phase != "as_loaded":
        res_now = (store.slot_of[meth.key] >= 0).nonzero().flatten().tolist()
        store.evict(meth.key, res_now[:128])
        store.make_resident(meth.key, list(range(300, 364)))
        torch.xpu.synchronize()
    for M in (1, 4, 16, 200, 2048):
        g = torch.Generator().manual_seed(M)
        x = (torch.randn((M, H), generator=g) * 0.5).to(torch.bfloat16).to(dev)
        wts, ids = torch.topk(torch.softmax(torch.randn((M, E), generator=g), -1), 10, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float()
        do = types.SimpleNamespace(hidden_states=x, topk_output=types.SimpleNamespace(topk_ids=ids.to(dev), topk_weights=wts.to(dev)))
        y = meth.apply(layer, do).hidden_states
        torch.xpu.synchronize()
        r = ref_moe(x, ids, wts)
        rms = ((y.float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
        floor = ((r.bfloat16().float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
        rec = {"phase": phase, "M": M, "resident": store.resident_count(meth.key), "rms_rel": round(rms, 5),
               "floor_bf16": round(floor, 5), "ok": rms < 2 * floor + 1e-3}
        res.append(rec)
        print(rec, flush=True)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
sys.exit(0 if all(r["ok"] for r in res) else 1)
