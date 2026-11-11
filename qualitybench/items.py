"""Normalise each harness's per-item output to {"id", "correct"} records (the study's to_items.py)."""
from __future__ import annotations

import glob
import re
import json
import os
import re
from pathlib import Path

LMEVAL_PICK = {"gsm8k": ("flexible-extract", "exact_match"), "ifeval": ("none", "prompt_level_strict_acc"),
               "mmlu_pro": ("custom-extract", "exact_match")}


GSM8K_GRADER = "qb-gsm8k-1"
_NUM = r"-?\$*\s?\d[\d,]*(?:\.\d+)?"


def _num(s: str) -> float | None:
    s = s.strip().replace("$", "").replace(",", "").replace("%", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_answer(reply: str) -> float | None:
    """The reply's final answer: the last \\boxed{} or '#### N'; else whichever comes last of a **bold** quantity
    and an 'Answer: N'; else the final sentence's last money amount, or its last number."""
    reply = re.sub(r"(?s)<think>.*?</think>", "", reply or "")
    reply = reply.replace("\\$", "$").replace("{,}", ",")  # LaTeX money and thousands
    for pat in (r"\\boxed\{([^{}]*)\}", r"####\s*(" + _NUM + ")"):
        for m in reversed(re.findall(pat, reply)):
            nums = re.findall(_NUM, m)
            if nums and (v := _num(nums[-1])) is not None:
                return v
    found = []  # (position, value)
    # **bold** spans, paired left to right; only quantities ("**45 miles**", "**$120**", "**80 + 150 = 230 miles**"),
    # not headings that hold a number ("**Step 4: ...**", "**4. Total:**").
    for m in re.finditer(r"\*\*(.+?)\*\*", reply):
        span = m.group(1).split("=")[-1]
        q = re.match(r"\s*(" + _NUM + r")(?!\d|\s*[.:)])", span)
        if q and (v := _num(q.group(1))) is not None:
            found.append((m.start(), v))
        elif m.end() >= len(reply.rstrip()) - 2 and (v := _sentence_answer(m.group(1))) is not None:
            found.append((m.start(), v))  # the reply ends on a bolded sentence: "**Cameron would save 250 pesos.**"
    for m in re.finditer(r"(?i)\banswer\s*(?:is)?\s*\**\s*[:=]?\s*\**\s*(" + _NUM + ")", reply):
        if (v := _num(m.group(1))) is not None:
            found.append((m.start(), v))
    if found:
        return max(found)[1]
    # Fallback: the final sentence's last money amount ("Josh makes $120 in 2 weeks."), else its last number.
    sentences = [x for x in re.split(r"(?<=[.!?])\s+|\n+", reply.strip()) if re.search(r"\d", x)]
    return _sentence_answer(sentences[-1]) if sentences else None


def _sentence_answer(sentence: str) -> float | None:
    nums = re.findall(r"\$+\s?\d[\d,]*(?:\.\d+)?", sentence) or re.findall(_NUM, sentence)
    return _num(nums[-1]) if nums else None


def grade_gsm8k(reply: str, target: str) -> bool:
    gold = _num(target.split("####")[-1])
    got = gsm8k_answer(reply)
    return gold is not None and got is not None and abs(got - gold) < 1e-6


def _reply(d: dict) -> str:
    r = d.get("resps")
    while isinstance(r, list):
        r = r[0] if r else ""
    return r or ""

# lm-eval's MMLU-Pro pattern is 'answer is \(?([A-J])\)?', so "the answer is **(D)**" has no answer.
_MMLU_MARKDOWN = re.compile(r"answer is[\s:*_]*\(?[\s*_]*([A-J])\b")


def mmlu_pro_correct(d: dict) -> bool:
    """lm-eval's letter when it found one; when it found none, the first letter after "answer is" once the
    markdown around it (**, _, a colon) is skipped."""
    if d["filtered_resps"][0] != "[invalid]":
        return bool(d["exact_match"])
    m = _MMLU_MARKDOWN.search(_reply(d))
    return bool(m) and m.group(1) == str(d["target"])





def from_lmeval(samples: Path, task: str) -> list[dict]:
    flt, metric = LMEVAL_PICK["mmlu_pro" if task.startswith("mmlu_pro") else task]
    out = []
    for line in open(samples):
        d = json.loads(line)
        if d["filter"] != flt:
            continue
        rec = {"id": f"{task}/{d['doc_id']}", "correct": bool(float(d[metric]))}
        if task == "gsm8k":
            rec = {"id": rec["id"], "correct": grade_gsm8k(_reply(d), str(d["target"])), "stock": rec["correct"],
                   "grader": GSM8K_GRADER}
        elif task.startswith("mmlu_pro"):
            rec["correct"] = mmlu_pro_correct(d)
        out.append(rec)
    return out


def regrade_gsm8k(side: Path) -> bool | None:
    """Re-grade an older run's gsm8k items with GSM8K_GRADER from its lm-eval samples.

    True if regraded (or already graded), False if it was graded the stock way and can't be (no samples),
    None if the run has no gsm8k.
    """
    p = side / "items" / "gsm8k.jsonl"
    if not p.is_file() or not p.stat().st_size:
        return None
    first = json.loads(p.open().readline())
    if first.get("grader") == GSM8K_GRADER:
        return True
    found = sorted((side / "lmeval").rglob("samples_gsm8k_*.jsonl"), key=lambda q: q.stat().st_mtime)
    if not found:
        return False
    keep = set(load(p))
    write(p, [r for r in from_lmeval(found[-1], "gsm8k") if r["id"] in keep])
    return True


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
