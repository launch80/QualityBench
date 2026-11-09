"""Re-scoring the Qwopus study's per-item files reproduces its published task numbers exactly."""
import json
import shutil
from pathlib import Path

from qualitybench import results, stats
from qualitybench.suite import BY_ID, SCORED

FIX = Path(__file__).parent / "fixtures"


def study_tasks():
    """The study's merge rule: later thinking-on dirs win per task, if they answered the same items."""
    from qualitybench import items
    rows = []
    for t in SCORED:
        R = items.load(FIX / "bf16" / f"{t}.jsonl")
        row = {"id": t, "label": BY_ID[t].label, "how": BY_ID[t].how, "off": stats.score(R)}
        T, src = None, None
        for d in ("bf16-think", "bf16-think2"):
            x = items.load(FIX / d / f"{t}.jsonl")
            if x is not None and x.keys() == R.keys():
                T, src = x, d
        if T:
            row["on"] = stats.score(T, src)
            row["paired"] = stats.paired(R, T)
        rows.append(row)
    return rows


def test_tasks_match_published_numbers_exactly():
    # The study's final BF16 report (2026-11-02 16:30), all seven tasks with thinking off and on.
    want = json.loads((FIX / "study-report-tasks.json").read_text())["tasks"]
    got = study_tasks()
    assert [t["id"] for t in got] == [t["id"] for t in want]
    for g, w in zip(got, want):
        for k in ("off", "on", "paired"):
            assert g.get(k) == w.get(k), (g["id"], k)


def test_task_rows_from_a_sidecar(tmp_path):
    side, on = tmp_path / "ref.qb", tmp_path / "think.qb"
    for src, dst in ((FIX / "bf16", side), (FIX / "bf16-think", on)):
        (dst / "items").mkdir(parents=True)
        for f in src.glob("*.jsonl"):
            shutil.copy(f, dst / "items" / f.name)
    rows = {r["id"]: r for r in results.task_rows(side, on, "bf16-think")}
    want = {t["id"]: t for t in json.loads((FIX / "study-report-tasks.json").read_text())["tasks"]}
    # The study's report described gsm8k as "last number"; the description changed in 0.3.0, the numbers did not.
    assert {k: v for k, v in rows["gsm8k"].items() if k != "how"} == {k: v for k, v in want["gsm8k"].items() if k != "how"}
    assert rows["mmlu_pro"]["off"] == want["mmlu_pro"]["off"] and "on" not in rows["mmlu_pro"]  # its thinking-on run is bf16-think2
