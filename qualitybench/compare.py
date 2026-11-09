"""`qualitybench compare`: a test run against a reference run, as an arm on the reference's page.

The output is the reference's results document plus `arms[<key>]` (KLD, top-1 agreement, perplexity change,
paired task deltas, spec parity) and a verdict per threshold. Thresholds come from a markdown table
(`| T1 | metric | rule |`); the default file is the Qwopus study's bars, minus its GGUF peer row.

Threshold ids have fixed meanings (the study's): T1 perplexity wikitext-2, T2 perplexity code, T3 mean KLD,
T4 top-1 agreement, T5 paired task deltas, T7 spec parity. The first number in each rule is the bar.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from . import envs, items, kld, stats
from .context import sidecar_for
from .results import file_hash, parse_thresholds
from .suite import SCORED

DEFAULT_GATES = {"task_delta_pts": -2.0, "top1_pct": 93.0, "ppl_delta_pct": 3.0, "kld_mean": 0.05}


def default_thresholds() -> Path:
    return envs.pkg_file("data", "thresholds-default.md")


def gates_from(rows: list[dict]) -> dict:
    g = dict(DEFAULT_GATES)
    num = lambda s: (m := re.search(r"[-+]?\d+(?:\.\d+)?", s)) and float(m.group())
    by = {r["id"]: r["rule"] for r in rows}
    for tid, key in (("T1", "ppl_delta_pct"), ("T3", "kld_mean"), ("T4", "top1_pct"), ("T5", "task_delta_pts")):
        if tid in by and (v := num(by[tid])) is not None:
            g[key] = -abs(v) if key == "task_delta_pts" else v
    return g


def arm_key(model: str) -> str:
    k = re.sub(r"[^A-Za-z0-9_.-]+", "-", model.rsplit("/", 1)[-1]).strip("-.") or "test"
    return k[:32]


def arm_block(ref_side: Path, test_side: Path, test_doc: dict, label: str | None, notes: list[str]) -> tuple[dict, list[str]]:
    warn = []
    env = test_doc.get("env", {})
    desc = " · ".join(x for x in (env.get("precision"), env.get("engine"), env.get("gpu")) if x)
    out = {"label": (label or test_doc["model"])[:60]}
    if desc:
        out["description"] = desc[:300]
    rz, tz = ref_side / "logprobs.npz", test_side / "logprobs.npz"
    if rz.is_file() and tz.is_file():
        try:
            k = kld.compare(rz, tz)
            if "kld" in k and "kld_mean" in k["kld"]:
                kd = k["kld"]
                out["kld"] = {"mean": kd["kld_mean"], "p50": kd["kld_median"], "p99": kd["kld_p99"], "max": kd["kld_max"],
                              "hist": kd["hist"], "by_pos": kd["by_pos"]}
                out["top1"] = {"agree": kd["top1_agree_pct"], "n": kd["positions"]}
            ppl = {s: {"ref": k[s]["ppl_ref"], "arm": k[s]["ppl_test"], "delta_pct": k[s]["ppl_delta_pct"]}
                   for s in ("ppl-wiki", "ppl-code", "kld") if s in k}
            if ppl:
                out["ppl"] = ppl
        except ValueError as e:
            warn.append(f"logprobs not compared: {e}")
    elif rz.is_file() or tz.is_file():
        warn.append("logprobs not compared: only one run has them")
    graded = (items.regrade_gsm8k(ref_side), items.regrade_gsm8k(test_side))
    if None not in graded and graded[0] != graded[1]:
        warn.append("gsm8k: one run is graded the stock lm-eval way (no samples left to re-grade), the other by "
                    "QualityBench's final-answer grader; the paired delta mixes the two")
    tasks = {}
    for t in SCORED:
        R, T = items.load(ref_side / "items" / f"{t}.jsonl"), items.load(test_side / "items" / f"{t}.jsonl")
        if R is None or T is None:
            continue
        if R.keys() != T.keys():
            warn.append(f"{t} not compared: the runs answered different items (quick vs full, or a partial run)")
            continue
        tasks[t] = stats.paired(R, T)
    if tasks:
        out["tasks"] = tasks
    rp, tp = ref_side / "parity.json", test_side / "parity.json"
    if rp.is_file() and tp.is_file():
        from .tasks.native import parity_compare
        out["spec_parity"] = {"vs": "ref", **parity_compare(json.loads(tp.read_text()), json.loads(rp.read_text()))}
    return out, warn


def verdicts(arm: dict, gates: dict, ref_tasks: list[dict]) -> dict:
    v = {}
    pct = lambda x: f"{x:+.2f} %"
    pl = arm.get("ppl", {})
    if "ppl-wiki" in pl:
        v["T1"] = {"value": pct(pl["ppl-wiki"]["delta_pct"]), "pass": pl["ppl-wiki"]["delta_pct"] <= gates["ppl_delta_pct"]}
    if "ppl-code" in pl:
        v["T2"] = {"value": pct(pl["ppl-code"]["delta_pct"]), "pass": pl["ppl-code"]["delta_pct"] <= gates["ppl_delta_pct"]}
    if "kld" in arm:
        v["T3"] = {"value": f"{arm['kld']['mean']:.4f} nats", "pass": arm["kld"]["mean"] <= gates["kld_mean"]}
    if "top1" in arm:
        v["T4"] = {"value": f"{arm['top1']['agree']:.2f} %", "pass": arm["top1"]["agree"] >= gates["top1_pct"]}
    want = [t["id"] for t in ref_tasks]
    if want and "tasks" in arm and all(t in arm["tasks"] for t in want):
        fails = [t for t in want if not (arm["tasks"][t]["delta"] >= gates["task_delta_pts"]
                                         or arm["tasks"][t]["ci"][0] <= 0 <= arm["tasks"][t]["ci"][1])]
        lab = {t["id"]: t["label"] for t in ref_tasks}
        v["T5"] = {"value": (("fails: " + ", ".join(lab[t] for t in fails)) if fails else f"{len(want)}/{len(want)} tasks")[:40],
                   "pass": not fails}
    if "spec_parity" in arm:
        s = arm["spec_parity"]
        v["T7"] = {"value": f"{s['n'] - len(s['mismatches'])}/{s['n']} identical", "pass": s["identical"]}
    return v


def build(ref_json: Path, test_json: Path, thresholds: str | None = None, key: str | None = None,
          label: str | None = None, ref_side: Path | None = None, test_side: Path | None = None) -> tuple[dict, list[str]]:
    ref = json.loads(Path(ref_json).read_text())
    test = json.loads(Path(test_json).read_text())
    rs, ts = ref_side or sidecar_for(Path(ref_json)), test_side or sidecar_for(Path(test_json))
    warn = []
    for name, s in (("reference", rs), ("test", ts)):
        if not s.is_dir():
            raise SystemExit(f"compare: the {name} run's per-item data ({s}) is missing; it is written next to the "
                             "results file by `qualitybench run` and must be kept with it")
    arm, w = arm_block(rs, ts, test, label, [])
    warn += w
    doc = copy.deepcopy(ref)
    key = key or arm_key(test["model"])
    if key == "ref":
        key = "test"
    arms = dict(doc.get("arms") or {})
    arms[key] = arm
    while len(arms) > 3:  # the page draws at most three arms
        arms.pop(next(iter(arms)))
    doc["arms"] = arms
    doc["gate_arm"] = key
    tpath = thresholds or str(default_thresholds())
    rows = parse_thresholds(tpath)
    gates = gates_from(rows)
    doc["gates"] = gates
    v = verdicts(arm, gates, ref.get("tasks", []))
    for r in rows:
        prev = next((x for x in doc.get("thresholds") or [] if x["id"] == r["id"]), {})
        res = {k: x for k, x in (prev.get("results") or {}).items() if k in arms}
        if r["id"] in v:
            res[key] = v[r["id"]]
        if res:
            r["results"] = res
    doc["thresholds"] = rows
    doc.setdefault("hashes", {})["THRESHOLDS.md"] = file_hash(tpath)
    doc["title"] = f"{ref['model']} vs {test['model']}"[:160] if not ref.get("arms") else doc["title"]
    doc["arms_note"] = (f"Arm {key}: {test['model']} scored against the reference {ref['model']} on the same items and "
                        "the same token windows.")[:600]
    return doc, warn


def summary_lines(doc: dict, key: str) -> list[str]:
    arm = doc["arms"][key]
    out = [f"arm {key}: {arm['label']}"]
    if "kld" in arm:
        out.append(f"  KLD mean {arm['kld']['mean']:.4f} nats, top-1 agreement {arm['top1']['agree']:.2f} %")
    for s, p in (arm.get("ppl") or {}).items():
        out.append(f"  perplexity {s}: {p['ref']:.4f} -> {p['arm']:.4f} ({p['delta_pct']:+.2f} %)")
    for t, q in (arm.get("tasks") or {}).items():
        out.append(f"  {t}: {q['delta']:+.2f} pts (95% CI {q['ci'][0]:+.2f} to {q['ci'][1]:+.2f}, McNemar p {q['p']:.3g})")
    for th in doc.get("thresholds", []):
        r = (th.get("results") or {}).get(key)
        out.append(f"  {th['id']} {'PASS' if r and r['pass'] else 'FAIL' if r else 'n/a '} {r['value'] if r else ''}  ({th['rule']})")
    return out
