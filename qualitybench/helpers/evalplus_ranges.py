#!/usr/bin/env python3
"""Split an evalplus dataset into --id_range shards (runs in the lmeval env).

  evalplus_ranges.py humaneval 16          -> 16 contiguous near-equal shards, "lo hi" per line
  evalplus_ranges.py humaneval 16 --quick  -> one single-id range per quick item (every tenth id)

evalplus.codegen's --id_range filters on the numeric part of task_id (lo <= n < hi), and MBPP+ ids are not
contiguous, so ranges are cut on the real id list.
"""
import sys
from evalplus.data import get_human_eval_plus, get_mbpp_plus

ds, n = sys.argv[1], int(sys.argv[2])
quick = "--quick" in sys.argv[3:]
probs = get_human_eval_plus() if ds == "humaneval" else get_mbpp_plus()
ids = sorted(int(t.split("/")[1]) for t in probs)
if quick:
    for i in ids[::10]:
        print(i, i + 1)
    raise SystemExit
n = max(1, min(n, len(ids)))
for k in range(n):
    part = ids[k * len(ids) // n:(k + 1) * len(ids) // n]
    if part:
        print(part[0], part[-1] + 1)
