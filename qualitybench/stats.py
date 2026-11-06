"""Accuracy intervals and paired comparisons.

These match the Qwopus study's stats.py / qb_report.py exactly (seed 608, 10,000 resamples), so
re-scoring that study's per-item files reproduces its published numbers to the last digit.
"""
from __future__ import annotations

import math
import random

SEED = 608
RESAMPLES = 10000


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for k successes in n, in percent."""
    if not n:
        return [0, 0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [100 * (c - h) / d, 100 * (c + h) / d]


def score(items: dict[str, bool], src: str | None = None) -> dict:
    k = sum(items.values())
    n = len(items)
    out = {"k": k, "n": n, "acc": 100 * k / n, "ci": wilson(k, n)}
    if src:
        out["src"] = src
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p: a binomial test on the discordant pairs."""
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, k) for k in range(0, min(b, c) + 1)) / 2 ** m)


def paired(ref: dict[str, bool], test: dict[str, bool], resamples: int = RESAMPLES) -> dict:
    """Test minus reference over the same items, in points, with a paired bootstrap 95% CI."""
    if ref.keys() != test.keys():
        raise ValueError(f"item sets differ: {len(ref.keys() ^ test.keys())} ids not shared")
    ids = sorted(ref)
    n = len(ids)
    b = sum(1 for i in ids if ref[i] and not test[i])
    c = sum(1 for i in ids if test[i] and not ref[i])
    rng = random.Random(SEED)
    diffs = [int(test[i]) - int(ref[i]) for i in ids]
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo, hi = boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]
    return {"n": n, "delta": 100 * (sum(test.values()) - sum(ref.values())) / n, "ci": [100 * lo, 100 * hi],
            "ref_only": b, "test_only": c, "p": mcnemar_p(b, c)}
