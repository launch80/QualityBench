#!/usr/bin/env python3
"""Pad an evalplus samples file to the whole dataset (runs in the lmeval env).

  evalplus_pad.py humaneval samples.jsonl ids.txt

evalplus.evaluate refuses a samples file that lacks any problem, but a quick run answers only a fixed tenth.
Problems the run did not answer get an empty solution (which the scorer fails at once). The answered ids go
to ids.txt, so only they are kept in the results.
"""
import json, sys
from evalplus.data import get_human_eval_plus, get_mbpp_plus

ds, samples, ids_out = sys.argv[1:4]
probs = get_human_eval_plus() if ds == "humaneval" else get_mbpp_plus()
rows = [json.loads(l) for l in open(samples) if l.strip()]
have = {r["task_id"] for r in rows}
with open(ids_out, "w") as f:
    f.write("\n".join(sorted(have)) + "\n")
missing = [t for t in probs if t not in have]
with open(samples, "a") as f:
    for t in missing:
        f.write(json.dumps({"task_id": t, "solution": ""}) + "\n")
print(f"evalplus {ds}: {len(have)} answered, {len(missing)} padded for the scorer")
