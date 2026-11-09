"""Normalise each harness's per-item output to {"id", "correct"} records (the study's to_items.py)."""
from __future__ import annotations

import glob
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

LMEVAL_PICK = {"gsm8k": ("flexible-extract", "exact_match"), "ifeval": ("none", "prompt_level_strict_acc"),
               "mmlu_pro": ("custom-extract", "exact_match")}


def number(s: str) -> Decimal | None:
    """A gsm8k answer as a number: "$26.00", "26." and "1,234" are 26, 26 and 1234."""
    try:
        return Decimal(re.sub(r"[$,]", "", str(s)).strip().rstrip("."))
    except InvalidOperation:
        return None


def gsm8k_correct(d: dict) -> bool:
    """lm-eval compares the last number as text after stripping "$" and ",", so "$26.00" fails a target of 26.
    A reply that lm-eval accepts stays accepted; one it rejects is right if the number is the same."""
    if d["exact_match"]:
        return True
    got, want = number(d["filtered_resps"][0]), number(d["target"].split("####")[-1])
    return got is not None and got == want


def from_lmeval(samples: Path, task: str) -> list[dict]:
    flt, metric = LMEVAL_PICK["mmlu_pro" if task.startswith("mmlu_pro") else task]
    out = []
    for line in open(samples):
        d = json.loads(line)
        if d["filter"] == flt:
            ok = gsm8k_correct(d) if task == "gsm8k" else bool(d[metric])
            out.append({"id": f"{task}/{d['doc_id']}", "correct": ok})
    return out


def from_evalplus(eval_results: Path) -> list[dict]:
    out = []
    for tid, runs in json.load(open(eval_results))["eval"].items():
        r = runs[0]  # greedy: one sample per task
        out.append({"id": tid, "correct": r["base_status"] == "pass" and r["plus_status"] == "pass"})
    return out


def from_bfcl(score_dir: Path) -> list[dict]:
    # BFCL writes one score file per category: line 1 is the summary, then one line per FAILED item.
    # Passing items are not listed, so the generated ids come from the result files next to it.
    out = []
    for f in sorted(glob.glob(os.path.join(score_dir, "**", "*_score.json"), recursive=True)):
        lines = [json.loads(l) for l in open(f)]
        failed = {d["id"] for d in lines[1:] if "id" in d}
        res = f.replace("/score/", "/result/").replace("_score.json", "_result.json")
        for d in map(json.loads, open(res)):
            out.append({"id": d["id"], "correct": d["id"] not in failed})
    return out


def from_lcb(eval_all: Path) -> list[dict]:
    return [{"id": str(d["question_id"]), "correct": bool(d["graded_list"][0])} for d in json.load(open(eval_all))]


def write(path: Path, items: list[dict]) -> tuple[int, int]:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    tmp.replace(path)  # atomic: a half-written items file never counts as finished
    return sum(i["correct"] for i in items), len(items)


def load(path: Path) -> dict[str, bool] | None:
    if not path.is_file() or not path.stat().st_size:
        return None
    return {r["id"]: bool(r["correct"]) for r in map(json.loads, path.open()) if r}
