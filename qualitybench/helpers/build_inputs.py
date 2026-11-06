#!/usr/bin/env python3
"""Build the fixed token inputs for the logprobs and needle tasks (runs in the lmeval environment).

  build_inputs.py --tokenizer <HF id or dir> --out <dir> [--what evalset,needle] [--lengths 32768,131072]

eval-set.json: {"ppl-wiki": [[ids]*40], "ppl-code": [[ids]*40], "kld": [[ids]*512], "meta": {...}}, 2048 tokens
per window. Every run with the same tokenizer scores exactly the same ids, so tokenizer drift cannot enter a
comparison. Sources (public, fetched once into the QualityBench HF cache):
  ppl-wiki  wikitext-2-raw-v1 test, "\n\n"-joined (the usual perplexity convention), first 40 windows
  ppl-code  codeparrot-clean-valid file 54, last 10 % of docs, shuffled with seed 608
  kld       256 windows of C4 en validation shard 1 (seed 608) + 256 more code windows
needle.json: the 20 needle prompts (32k and 128k x 10 depths), with C4 shard 1 as the haystack. The construction
is the Qwopus study's needle.py, unchanged, so the same tokenizer gives the same prompts and numbers.
"""
import argparse, gzip, hashlib, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--what", default="evalset,needle"); ap.add_argument("--lengths", default="32768,131072")
ap.add_argument("--depths", type=int, default=10); ap.add_argument("--seqlen", type=int, default=2048)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.tokenizer)
C4 = lambda: hf_hub_download("allenai/c4", "en/c4-validation.00001-of-00008.json.gz", repo_type="dataset")
L = a.seqlen

if "evalset" in a.what:
    from datasets import load_dataset
    random.seed(608)

    def windows(texts, n):
        out, buf = [], []
        for t in texts:
            buf += tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            while len(buf) >= L and len(out) < n:
                out.append(buf[:L]); buf = buf[L:]
            if len(out) >= n:
                return out
        raise SystemExit(f"only {len(out)} windows, wanted {n}")

    wiki = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    wiki_ids = tok(wiki, add_special_tokens=False)["input_ids"]
    ppl_wiki = [wiki_ids[i * L:(i + 1) * L] for i in range(40)]
    code_path = hf_hub_download("codeparrot/codeparrot-clean-valid", "file-000000000054.json.gz", repo_type="dataset")
    code_docs = [json.loads(l)["content"] for l in gzip.open(code_path, "rt")]
    code_tail = code_docs[int(len(code_docs) * 0.9):]
    random.shuffle(code_tail)
    code_win = windows(code_tail, 256 + 40)
    ppl_code, kld_code = code_win[:40], code_win[40:]
    c4 = [json.loads(l)["text"] for l in gzip.open(C4(), "rt")]
    random.shuffle(c4)
    kld = windows(c4, 256) + kld_code
    out = {"ppl-wiki": ppl_wiki, "ppl-code": ppl_code, "kld": kld}
    blob = json.dumps(out, separators=(",", ":")).encode()
    out["meta"] = {"seqlen": L, "seed": 608, "tokenizer": a.tokenizer, "counts": {k: len(v) for k, v in out.items()},
                   "sha256_ids": hashlib.sha256(blob).hexdigest(), "vocab": len(tok)}
    json.dump(out, open(os.path.join(a.out, "eval-set.json"), "w"), separators=(",", ":"))
    print("eval-set", json.dumps(out["meta"]))

if "needle" in a.what:
    rng = random.Random(608)
    docs = [json.loads(l)["text"] for l in gzip.open(C4(), "rt")]
    rng.shuffle(docs)
    lengths = list(map(int, a.lengths.split(",")))
    hay_ids, i = [], 0
    need = max(lengths) + 1024
    while len(hay_ids) < need:
        hay_ids += tok(docs[i] + "\n\n", add_special_tokens=False)["input_ids"]; i += 1
    prompts = []
    for Ln in lengths:
        for d in range(a.depths):
            depth = d / (a.depths - 1)
            num = str(rng.randrange(1_000_000, 9_999_999))
            needle = f"\n\nThe magic number for the blue lighthouse is {num}. Remember it.\n\n"
            body_len = Ln - 200  # room for the question and the chat template
            cut = int(body_len * depth)
            pre, suf = tok.decode(hay_ids[:cut]), tok.decode(hay_ids[cut:body_len])
            q = pre + needle + suf + "\n\nWhat is the magic number for the blue lighthouse? Answer with the number only."
            prompts.append({"length": Ln, "depth": round(depth, 3), "number": num, "prompt": q})
    json.dump({"tokenizer": a.tokenizer, "prompts": prompts}, open(os.path.join(a.out, "needle.json"), "w"))
    print("needle", len(prompts), "prompts")
