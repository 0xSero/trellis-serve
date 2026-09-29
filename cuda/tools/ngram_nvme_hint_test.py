"""Mock test of the SGLang scheduler prefill hint (install_prefill_hints): no GPU, no server.
A fake Scheduler with a chunked request -> the wrapped get_new_batch_prefill must warm exactly the next chunk's rows.
  python3 tools/ngram_nvme_hint_test.py --model /models/...
"""
import argparse
import types

import torch

EOS = 248044


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    from sglang.srt.managers import scheduler as sch
    from sglang_exl3.offload import ngram_host
    from sglang_exl3.offload.ngram_nvme import Exl3NgramNvmeTable, install_prefill_hints
    sch.Scheduler.get_new_batch_prefill = lambda self, *a, **k: "batch"   # stand-in for the real scheduling
    install_prefill_hints()
    tab = Exl3NgramNvmeTable(a.model, ram_gb=0.1, io="aio", device=None, max_tokens=8192, start_service=False)
    tab.eos_token_id = EOS
    ngram_host._TABLES["/m"] = tab
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 150000, (20000,), generator=g).tolist()
    fake = types.SimpleNamespace(chunked_req=types.SimpleNamespace(origin_input_ids=prompt, fill_ids=prompt[:8192]),
                                 chunked_prefill_size=8192)
    assert sch.Scheduler.get_new_batch_prefill(fake) == "batch"
    tab._hint_pool.shutdown(wait=True)
    tab._hint_pool = None
    st = tab.stats(gpu=False)
    # every row of chunk 2 must now hit; chunk 3 must miss
    exp = tab.hash_tokens(prompt[8192:16384], history=prompt[8190:8192], eos=EOS).reshape(-1).contiguous()
    slots = torch.empty(tab.cap, dtype=torch.long)
    tab.reset_stats()
    tab.store.resolve(exp, slots)
    hit2 = tab.stats(gpu=False)["lookup_hit_rate"]
    tab.reset_stats()
    nxt = tab.hash_tokens(prompt[16384:20000], history=prompt[16382:16384], eos=EOS).reshape(-1).contiguous()
    tab.store.resolve(nxt, slots)
    hit3 = tab.stats(gpu=False)["lookup_hit_rate"]
    print(dict(warm_lookups=st["warm_lookups"], warm_misses=st["warm_misses"], warm_ms=st["warm_us"] / 1e3,
               next_chunk_hit=hit2, chunk_after_hit=hit3))
    assert st["warm_lookups"] == 8192 * 16 and hit2 == 1.0 and hit3 < 0.1
    tab.release()
    print("PASS")


if __name__ == "__main__":
    main()
