"""K06a: the SGLANG_EXL3_HC_INT8 patch on a real SGLang GatedResidual (layer 10 mlp_hyper_connection weights):
patched mix vs the original bf16 mix (rows 1 / 16 / 64 = fallback path), weights freed, CUDA graph replay == eager.
python -m sglang_exl3.tools.hc_int8_patch_test <model> [row|g128]"""
import sys, os, json
import torch


def main():
    model, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "row")
    os.environ["SGLANG_EXL3_HC_INT8"] = mode
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
    from ..sglang_glue import hc_int8_patch
    from .hc_int8_bench import load_sites, normed_input
    cfg = HyperConnectionConfig(hc_count=4, hidden_size=2560, params_dtype=torch.bfloat16, hc_lowrank=320,
                                rms_norm_eps=1e-6, hc_per_branch_norm=True)
    site = [s for s in load_sites(model) if s["name"].endswith("layers.10.mlp_hyper_connection")][0]
    m = GatedResidual(cfg, use_mix=True, use_combine=True).cuda()
    m.hc_norm.to(torch.bfloat16)                      # SGLang builds the model under a bf16 default dtype
    with torch.no_grad():
        m.input_mix_weight_down.weight.copy_(site["down"].bfloat16())
        m.input_mix_weight_up.weight.copy_(site["up"].bfloat16())
        m.hc_norm.weight.copy_(site["norm"])
    gen = torch.Generator().manual_seed(0)
    xs = {r: (torch.randn((r, 10240), generator=gen) * torch.exp(torch.randn((r, 1), generator=gen))).bfloat16().cuda()
          for r in (1, 16, 64)}
    ref = {r: m.mix(x)[0].float() for r, x in xs.items()}
    wrap = torch.nn.Sequential(m)
    hc_int8_patch.install()
    st = hc_int8_patch.quantize_model_hc(wrap, mode)
    res = {"stats": st, "weights_freed": m.input_mix_weight_down.weight.numel() == 0, "rel_err_vs_bf16": {}}
    for r, x in xs.items():
        y = m.mix(x)[0].float()
        res["rel_err_vs_bf16"][r] = float((y - ref[r]).norm() / ref[r].norm())
    x = xs[1].clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        m.mix(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        yg = m.mix(x)[0]
    ok = True
    for i in range(10):
        x.copy_((torch.randn((1, 10240), generator=gen)).bfloat16())
        g.replay(); torch.cuda.synchronize()
        ok &= torch.allclose(yg.float(), m.mix(x)[0].float(), rtol=0, atol=1e-2 * float(yg.float().abs().max()))
    res["graph_replay_close_to_eager"] = bool(ok)   # atomics: split-K order varies -> tolerance, not bit-exact
    print("HC_PATCH", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
