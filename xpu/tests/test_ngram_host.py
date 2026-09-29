"""X004: XPU n-gram host table (exl3xpu/ngram_host.py) vs exllamav3's ngram_codec.dequant_rows on rows read from the
file. python3 tests/test_ngram_host.py [--model /model]"""
import os, sys, time, json, argparse, struct, random
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from exl3xpu.ngram_host import Exl3NgramHostTable, _header
import ngram_codec_ref as ref

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="/model"); ap.add_argument("--out", default="")
a = ap.parse_args()
dev = torch.device("xpu", 0)
t0 = time.time()
tab = Exl3NgramHostTable(a.model)
load_s = time.time() - t0
path = os.path.join(a.model, "ngram_embedding.safetensors")
hdr, base = _header(path)
shards = sorted([(int(k.split(".shard_")[1].split(".")[0]), v) for k, v in hdr.items() if ".shard_" in k])
rows_per = [v["shape"][0] for _, v in shards]
starts = [sum(rows_per[:i]) for i in range(len(rows_per))]


def file_row(r):
    for i, s0 in enumerate(starts):
        if r < s0 + rows_per[i]:
            a_, _ = shards[i][1]["data_offsets"]
            with open(path, "rb") as f:
                f.seek(base + a_ + (r - s0) * tab.words * 2)
                return torch.frombuffer(bytearray(f.read(tab.words * 2)), dtype=torch.int16)
    raise IndexError(r)


rng = random.Random(0)
T = 256
ids = torch.tensor([[rng.randrange(tab.num_rows) for _ in range(16)] for _ in range(T)], dtype=torch.int64)
ids[0, 0] = 0; ids[0, 1] = tab.num_rows - 1; ids[1, 2] = tab.rows_per_chunk; ids[1, 3] = tab.rows_per_chunk - 1
out = tab.gather(ids.to(dev)).float().cpu().view(T * 16, 160)
packed = torch.stack([file_row(int(r)) for r in ids.flatten()])
cbk = ref.mul1_codebook("cpu")
bias = tab.head_bias.cpu().float()[torch.arange(T * 16) % 16]
exp = ref.dequant_rows(packed, tab.K, cbk, bias)
err = (out - exp).abs().max().item()
rel = ((out - exp).abs() / exp.abs().clamp_min(1e-3)).max().item()
bf16_floor = (exp.bfloat16().float() - exp).abs().max().item()
# timings
torch.xpu.synchronize()
one = ids[:1].to(dev)
for _ in range(10): tab.gather(one)
torch.xpu.synchronize(); t = time.perf_counter()
for _ in range(200): tab.gather(one)
torch.xpu.synchronize(); t_dec = (time.perf_counter() - t) / 200 * 1e6
big = torch.randint(0, tab.num_rows, (16384, 16), dtype=torch.int64, device=dev)
tab.gather(big); torch.xpu.synchronize(); t = time.perf_counter()
for _ in range(5): tab.gather(big)
torch.xpu.synchronize(); t_pf = (time.perf_counter() - t) / 5 * 1e3
res = {"load_s": round(load_s, 1), "rows_checked": T * 16, "max_abs_err": err, "bf16_rounding_floor": bf16_floor,
       "max_rel_err": rel, "decode_token_16_rows_us": round(t_dec, 1), "prefill_16k_tokens_ms": round(t_pf, 2),
       "chunks": len(tab._chunks)}
print(res)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
sys.exit(0 if err <= 2 * bf16_floor + 1e-3 else 1)
