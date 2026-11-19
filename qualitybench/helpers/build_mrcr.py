#!/usr/bin/env python3
"""Fetch openai/mrcr and extract the pinned 8-needle rows (runs in the lmeval environment).

  build_mrcr.py --ids qualitybench/data/mrcr-ids.json --out <dir>

Writes <dir>/mrcr-8needle.json:
  {"meta": {"dataset", "n_needles", "count", "sha256_ids"},
   "items": [{"id", "length", "tokens", "messages", "answer", "prefix"}, ...]}

Every pinned row is checked against the sha256 of its prompt and of its answer, so a changed dataset fails
loudly instead of scoring other items. The text is fixed, so unlike the needle prompts this file does not
depend on the model's tokenizer. The last line printed on failure starts with "mrcr: " and becomes the task's
failure message.
"""
import argparse
import hashlib
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

pins = json.load(open(a.ids))
want = {p["id"]: p for p in pins["items"]}
os.makedirs(a.out, exist_ok=True)

from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

items = {}
for rel in pins["files"]:
    base = os.path.basename(rel)
    rows = sorted((p for p in want.values() if p["file"] == base), key=lambda p: p["row"])
    if not rows:
        continue
    try:
        path = hf_hub_download(pins["dataset"], rel, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"mrcr: could not download {pins['dataset']}/{rel} from Hugging Face: "
                         f"{type(e).__name__}: {str(e)[:200]}")
    t = pq.read_table(path, columns=["prompt", "answer", "random_string_to_prepend"])
    for p in rows:
        if p["row"] >= t.num_rows:
            raise SystemExit(f"mrcr: {base} has {t.num_rows} rows but pinned row {p['row']} is missing: the dataset changed")
        prompt = t.column("prompt")[p["row"]].as_py()
        answer = t.column("answer")[p["row"]].as_py()
        prefix = t.column("random_string_to_prepend")[p["row"]].as_py()
        if (hashlib.sha256(prompt.encode()).hexdigest()[:16] != p["id"]
                or hashlib.sha256(answer.encode()).hexdigest()[:16] != p["answer_sha256_16"]):
            raise SystemExit(f"mrcr: row {p['row']} of {base} is not the pinned item {p['id']}: the dataset changed")
        items[p["id"]] = {"id": p["id"], "length": p["length"], "tokens": p["tokens"], "messages": json.loads(prompt),
                          "answer": answer, "prefix": prefix}
    del t

missing = sorted(set(want) - set(items))
if missing:
    raise SystemExit(f"mrcr: {len(missing)} pinned items were not found in the dataset files (first: {missing[0]})")

order = sorted(items, key=lambda i: (items[i]["length"], i))
out = {"meta": {"dataset": pins["dataset"], "n_needles": pins["n_needles"], "count": len(items),
                "sha256_ids": hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()},
       "items": [items[i] for i in order]}
final = os.path.join(a.out, "mrcr-8needle.json")
tmp = final + ".tmp"
with open(tmp, "w") as f:
    json.dump(out, f, separators=(",", ":"))
os.replace(tmp, final)
print("mrcr", len(items), "items")
