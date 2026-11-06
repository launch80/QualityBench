"""Perplexity change, KL divergence and top-1 agreement of a test run vs a reference (the study's kld.py).

KLD(P_ref || Q_test) per position over the reference's top-k ids plus one tail bucket:
  P: ref top-k probs, tail = 1 - sum.
  Q: test's prob for each of those ids if it is in test's top-k, else test's tail mass spread evenly over the
     vocab ids outside its top-k; Q tail = 1 - sum(Q over ref top-k).
Exact for the top-k part; the tail bucket coarse-grains the rest.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_VOCAB = 248320


def compare(ref_npz: Path, test_npz: Path, vocab: int | None = None) -> dict:
    R, T = np.load(ref_npz), np.load(test_npz)
    rm, tm = json.loads(str(R["meta"])), json.loads(str(T["meta"]))
    if rm["eval_set_sha256"] != tm["eval_set_sha256"]:
        raise ValueError("the two runs were scored on different eval sets (different tokenizer or --eval-set)")
    if rm.get("windows") and tm.get("windows") and rm["windows"] != tm["windows"]:
        raise ValueError("the two runs scored different windows (a quick run against a full one?)")
    vocab = vocab or rm.get("vocab") or tm.get("vocab") or DEFAULT_VOCAB
    res: dict = {"ref": rm["model"], "test": tm["model"], "eval_set_sha256": rm["eval_set_sha256"]}
    for s in ("ppl-wiki", "ppl-code", "kld"):
        if f"{s}.target" not in R or f"{s}.target" not in T:
            continue
        pr, pt = math.exp(-R[f"{s}.target"].mean()), math.exp(-T[f"{s}.target"].mean())
        res[s] = {"ppl_ref": pr, "ppl_test": pt, "ppl_delta_pct": 100 * (pt / pr - 1)}
    if "kld.ids" in R and "kld.ids" in T:
        rid, rlp = R["kld.ids"].astype(np.int64), R["kld.lp"].astype(np.float64)
        tid, tlp = T["kld.ids"].astype(np.int64), T["kld.lp"].astype(np.float64)
        W, P, K = rid.shape
        rid, rlp, tid, tlp = (x.reshape(W * P, K) for x in (rid, rlp, tid, tlp))
        p = np.exp(rlp)
        p_tail = np.clip(1 - p.sum(1), 1e-12, None)
        tq = np.exp(tlp)
        t_tail_each = np.clip(1 - tq.sum(1), 1e-12, None) / (vocab - K)
        q = np.empty_like(p)
        for j in range(K):
            hit = tid == rid[:, j:j + 1]
            q[:, j] = np.where(hit.any(1), (tq * hit).sum(1), t_tail_each)
        q = np.clip(q, 1e-12, None)
        q_tail = np.clip(1 - q.sum(1), 1e-12, None)
        kl = (p * (np.log(np.clip(p, 1e-12, None)) - np.log(q))).sum(1) + p_tail * (np.log(p_tail) - np.log(q_tail))
        kl = np.clip(kl, 0, None)
        edges = np.logspace(-5, 2, 57)
        counts, _ = np.histogram(np.clip(kl, edges[0], edges[-1] * (1 - 1e-12)), bins=edges)
        klw, pe = kl.reshape(W, P), np.linspace(0, P, 33).astype(int)
        res.setdefault("kld", {}).update({
            "kld_mean": float(kl.mean()), "kld_median": float(np.median(kl)), "kld_p99": float(np.quantile(kl, 0.99)),
            "kld_max": float(kl.max()), "top1_agree_pct": float(100 * (rid[:, 0] == tid[:, 0]).mean()),
            "ref_topk_mass_mean": float(p.sum(1).mean()), "positions": int(W * P),
            "hist": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
            "by_pos": [[int(pe[i]), int(pe[i + 1]), float(klw[:, pe[i]:pe[i + 1]].mean())] for i in range(32)]})
    return res
