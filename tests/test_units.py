import json
import time

import numpy as np
import pytest

from qualitybench import kld, stats, suite
from qualitybench.cli import main
from qualitybench.context import parse_deadline, sidecar_for
from qualitybench.results import parse_thresholds, scrub
from qualitybench.tasks import harness
from qualitybench.tasks.native import parity_compare


def test_suite_matches_l80_table():
    # L80's internal/qualitybench/suite.go holds the same ids, item counts and default flags. mrcr and arc_agi_2 are
    # opt-in until the release that adds them there (docs/contract.md).
    want = [("gsm8k", 1319, True), ("ifeval", 541, True), ("mmlu_pro", 1000, True), ("humaneval", 164, True),
            ("mbpp", 378, True), ("bfcl", 400, True), ("lcb", 175, True), ("arc_agi_2", 167, False),
            ("logprobs", 592, True), ("needle", 20, True), ("spec_parity", 32, False), ("mrcr", 200, False)]
    assert [(t.id, t.items, t.default) for t in suite.SUITE] == want
    assert suite.run_order(["mrcr", "gsm8k", "bfcl"], deadline=True) == ["bfcl", "gsm8k", "mrcr"]  # the dearest task last


def test_parse_tasks():
    assert suite.parse_tasks(None) == suite.default_ids()
    assert suite.parse_tasks("ifeval, GSM8K,ifeval") == ["gsm8k", "ifeval"]
    with pytest.raises(ValueError):
        suite.parse_tasks("gsm8k,nope")


# Each task's metadata lives in its own module: these are the values the old per-concern tables held.
TASK_FIELDS = {  # id: (scored, resumable, output, ref_minutes, overhead_s, floor_s, prefill, lmeval_pick)
    "gsm8k": (True, True, None, 19, 15, 0, False, ("flexible-extract", "exact_match")),
    "ifeval": (True, True, None, 9, 18, 0, False, ("none", "prompt_level_strict_acc")),
    "mmlu_pro": (True, True, None, 42, 25, 0, False, ("custom-extract", "exact_match")),
    "humaneval": (True, True, None, 22, 25, 0, False, None),
    "mbpp": (True, True, None, 32, 30, 0, False, None),
    "bfcl": (True, False, None, 4, 5, 0, False, None),
    "lcb": (True, True, None, 17, 35, 0, False, None),
    "logprobs": (False, False, "logprobs.npz", 16, 10, 1.5, True, None),
    "needle": (False, False, "needle.json", 6, 10, 0, True, None),
    "spec_parity": (False, False, "parity.json", 2, 5, 0, False, None),
    "mrcr": (False, False, "mrcr.json", 200, 15, 0, True, None),
}

PREFETCH_SCRIPT = r"""
import sys
want = set(sys.argv[1].split(","))
from datasets import load_dataset
if "gsm8k" in want: print("gsm8k", len(load_dataset("openai/gsm8k", "main", split="test")))
if "ifeval" in want: print("ifeval", len(load_dataset("google/IFEval", split="train")))
if "mmlu_pro" in want: print("mmlu_pro", len(load_dataset("TIGER-Lab/MMLU-Pro", split="test")))
if want & {"humaneval", "mbpp"}:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    print("humaneval+", len(get_human_eval_plus()), "mbpp+", len(get_mbpp_plus()))
"""


def test_task_modules_carry_the_old_tables(tmp_path):
    from types import SimpleNamespace
    from qualitybench import runner
    # a new task is one more module: pin these eleven and their relative order, not the whole set
    assert [t.id for t in suite.SUITE if t.id in TASK_FIELDS] == list(TASK_FIELDS)
    assert len({t.order for t in suite.SUITE}) == len(suite.SUITE) == len(suite.BY_ID)
    assert suite.SUITE == sorted(suite.SUITE, key=lambda t: t.order)
    assert [i for i in suite.SCORED if i in TASK_FIELDS] == ["gsm8k", "ifeval", "mmlu_pro", "humaneval", "mbpp", "bfcl", "lcb"]
    got = {i: (t.scored, t.resumable, t.output, t.ref_minutes, t.overhead_s, t.floor_s, t.prefill, t.lmeval_pick)
           for i in TASK_FIELDS for t in [suite.BY_ID[i]]}
    assert got == TASK_FIELDS
    # a non-scored task is finished once its sidecar file is there
    ctx = SimpleNamespace(state={"partials": {}}, side=tmp_path, items_path=lambda t: tmp_path / f"{t}.items.jsonl")
    for t in suite.SUITE:
        assert not runner.finished(ctx, t.id)
        (tmp_path / (t.output or f"{t.id}.items.jsonl")).write_text("x")
        assert runner.finished(ctx, t.id)


def test_run_task_runs_the_tasks_own_run(monkeypatch, tmp_path):
    import dataclasses
    from types import SimpleNamespace
    from qualitybench import runner
    assert all(callable(t.run) for t in suite.SUITE)
    ctx, client, got, emitted = SimpleNamespace(items_path=lambda t: tmp_path / f"{t}.items.jsonl"), object(), [], []

    def fake(c, cl):
        got.append((c, cl))
        (tmp_path / "gsm8k.items.jsonl").write_text('{"id": "a", "correct": 1}\n{"id": "b", "correct": 0}\n')
    monkeypatch.setitem(suite.BY_ID, "gsm8k", dataclasses.replace(suite.BY_ID["gsm8k"], run=fake))
    monkeypatch.setattr(runner, "emit", lambda task, **f: emitted.append((task, f)))
    runner.run_task(ctx, "gsm8k", client)
    assert got == [(ctx, client)] and emitted == [("gsm8k", {"state": "scored", "correct": 1, "total": 2})]


def test_each_task_runs_its_engine(monkeypatch):
    from qualitybench.tasks import native
    calls = []
    for mod, names in ((harness, ("run_lmeval", "run_evalplus", "run_bfcl", "run_lcb")),
                       (native, ("run_logprobs", "run_needle", "run_mrcr", "run_parity"))):
        for n in names:
            monkeypatch.setattr(mod, n, lambda ctx, *a, n=n: calls.append((ctx, n, a)))
    ctx, client = object(), object()
    want = {"gsm8k": ("run_lmeval", ("gsm8k",)), "ifeval": ("run_lmeval", ("ifeval",)),
            "mmlu_pro": ("run_lmeval", ("mmlu_pro",)), "humaneval": ("run_evalplus", ("humaneval",)),
            "mbpp": ("run_evalplus", ("mbpp",)), "bfcl": ("run_bfcl", ()), "lcb": ("run_lcb", ()),
            "logprobs": ("run_logprobs", (client,)), "needle": ("run_needle", (client,)),
            "mrcr": ("run_mrcr", (client,)), "spec_parity": ("run_parity", (client,))}
    for t, (fn, args) in want.items():
        calls.clear()
        suite.BY_ID[t].run(ctx, client)
        assert calls == [(ctx, fn, args)], t


def test_each_task_prepares_its_inputs(monkeypatch):
    from qualitybench.tasks import native
    calls = []
    monkeypatch.setattr(native, "ensure_inputs", lambda ctx, what: calls.append((ctx, what)))
    ctx = object()
    want = {"logprobs": "evalset", "needle": "needle", "gsm8k": None, "ifeval": None, "mmlu_pro": None,
            "humaneval": None, "mbpp": None, "bfcl": None, "lcb": None, "spec_parity": None, "mrcr": None}
    for t, what in want.items():
        calls.clear()
        if what is None:
            assert suite.BY_ID[t].prepare is None, t
            continue
        suite.BY_ID[t].prepare(ctx)
        assert calls == [(ctx, what)], t


def test_no_scored_task_names_launch80s_tasks(monkeypatch):
    import dataclasses
    from qualitybench import results
    msg = "no scored task (launch80 needs at least one of {})"
    assert results.validate_shape({"tasks": {}}) == [msg.format("gsm8k, ifeval, mmlu_pro, humaneval, mbpp, bfcl, lcb")]
    assert results.validate_shape({"tasks": {}}) == [msg.format(", ".join(t.id for t in suite.SUITE if t.scored and t.default))]
    # a scored task that is opt-in is not one launch80 carries yet
    monkeypatch.setitem(suite.BY_ID, "toy", dataclasses.replace(suite.BY_ID["gsm8k"], id="toy", order=999, default=False))
    monkeypatch.setattr(results, "SCORED", (*suite.SCORED, "toy"))
    assert "toy" not in results.validate_shape({"tasks": {}})[0]


def test_suite_rejects_clashing_tasks():
    from qualitybench.taskdef import Task
    a, b = suite.BY_ID["gsm8k"], suite.BY_ID["ifeval"]
    assert suite.build({"gsm8k": a, "ifeval": b}) == [a, b]
    with pytest.raises(ValueError, match="gsm8k.*copy.*id 'gsm8k'"):
        suite.build({"gsm8k": a, "ifeval": b, "copy": Task(**{**vars(a), "order": 999})})
    with pytest.raises(ValueError, match="ifeval.*other.*order 20"):
        suite.build({"gsm8k": a, "ifeval": b, "other": Task(**{**vars(b), "id": "other"})})


def test_setup_needs_follow_the_tasks(monkeypatch, tmp_path):
    from qualitybench import envs, estimate as e
    monkeypatch.setenv("QUALITYBENCH_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(envs, "ensure_venv", lambda n: calls.append(("venv", n)))
    monkeypatch.setattr(envs, "ensure_mrcr_inputs", lambda: calls.append(("mrcr",)))
    monkeypatch.setattr(envs, "ensure_evalplus_image", lambda: calls.append(("evalplus",)))
    monkeypatch.setattr(envs, "ensure_lcb", lambda sb: calls.append(("lcb", sb)))
    monkeypatch.setattr(envs, "lcb_docker_base", lambda net, mounts: ["docker"])

    def run(cmd, **kw):
        cmd = list(map(str, cmd))
        if envs.LCB_PREFETCH in cmd:
            calls.append(("lcb_prefetch",))
        else:
            assert cmd[2] == PREFETCH_SCRIPT
            calls.append(("prefetch", cmd[3]))
    monkeypatch.setattr(envs, "_run", run)
    want = {
        (tuple(TASK_FIELDS), "docker"): [("venv", "lmeval"), ("prefetch", "gsm8k,humaneval,ifeval,mbpp,mmlu_pro"),
                                         ("mrcr",), ("venv", "bfcl"), ("evalplus",), ("lcb", "docker"), ("lcb_prefetch",)],
        (("logprobs", "needle"), "docker"): [("venv", "lmeval")],
        (("spec_parity",), "docker"): [],
        (("mbpp",), "native"): [("venv", "lmeval"), ("prefetch", "mbpp")],
        (("lcb",), "native"): [("lcb", "native"), ("lcb_prefetch",)],
        (("mrcr",), "docker"): [("venv", "lmeval"), ("mrcr",)],
        (("ifeval", "bfcl"), "docker"): [("venv", "lmeval"), ("prefetch", "ifeval"), ("venv", "bfcl")],
    }
    for (tasks, sb), w in want.items():
        calls.clear()
        envs.setup(list(tasks), sb)
        assert calls == w, (tasks, sb)
    # nothing installed yet: the one-time minutes each set still needs
    monkeypatch.setattr(e, "_stamped", lambda n: False)
    monkeypatch.setattr(e, "_image", lambda i: False)
    minutes = {(tuple(TASK_FIELDS), "docker"): 17, (tuple(TASK_FIELDS), "native"): 14, (("gsm8k",), "docker"): 4,
               (("logprobs", "needle"), "docker"): 4, (("bfcl",), "docker"): 2, (("humaneval",), "docker"): 6,
               (("humaneval", "mbpp"), "native"): 4, (("lcb",), "native"): 5, (("lcb",), "docker"): 6,
               (("mrcr",), "docker"): 7, (("spec_parity",), "docker"): 0}
    assert {k: e.setup_minutes(list(k[0]), k[1]) for k in minutes} == minutes
    monkeypatch.setattr(e, "_stamped", lambda n: True)
    monkeypatch.setattr(e, "_image", lambda i: True)
    (tmp_path / "inputs" / "mrcr").mkdir(parents=True)
    envs.mrcr_inputs().write_text("{}")
    assert e.setup_minutes(list(TASK_FIELDS), "docker") == 0 == e.setup_minutes(list(TASK_FIELDS), "native")


def test_quick_subsets_are_fixed_tenths():
    assert harness.lmeval_samples("gsm8k", True) == {"gsm8k": list(range(0, 1319, 10))}
    assert harness.lmeval_samples("gsm8k", False) is None
    m = harness.lmeval_samples("mmlu_pro", True)
    assert sum(map(len, m.values())) == 100
    assert sum(map(len, harness.lmeval_samples("mmlu_pro", False).values())) == 1000
    b = harness.bfcl_ids(True)
    assert {k: len(v) for k, v in b.items()} == {k: 10 for k in b}
    assert len(harness.lcb_ids(False)) == 175 and len(harness.lcb_ids(True)) == 18
    assert harness.lcb_ids(True) == harness.lcb_ids(True)


def test_harness_versions_from_locks():
    assert harness.harness_versions() == {"lm_eval": "0.4.13", "evalplus": "0.3.1", "bfcl_eval": "2026.3.23"}


def test_wilson_and_paired():
    lo, hi = stats.wilson(50, 100)
    assert lo < 50 < hi and abs((50 - lo) - (hi - 50)) < 1e-9
    R = {str(i): i % 2 == 0 for i in range(100)}
    assert stats.paired(R, R) == {"n": 100, "delta": 0.0, "ci": [0.0, 0.0], "ref_only": 0, "test_only": 0, "p": 1.0}
    with pytest.raises(ValueError):
        stats.paired(R, {"x": True})
    assert stats.mcnemar_p(0, 10) < 0.01


def test_deadline():
    now = 1_000_000.0
    assert parse_deadline("6h", now) == now + 6 * 3600
    assert parse_deadline("1h30m", now) == now + 5400
    assert parse_deadline("90m", now) == now + 5400
    assert parse_deadline("2026-11-06T12:00:00Z") == 1793966400.0
    assert parse_deadline(None) is None
    with pytest.raises(ValueError):
        parse_deadline("tomorrow")


def test_sidecar_path(tmp_path):
    assert sidecar_for(tmp_path / "bf16.json") == tmp_path / "bf16.qb"


def test_scrub_removes_machine_details_but_keeps_endpoint():
    doc = {"model": "/home/me/models/Qwen3-8B", "env": {"endpoint": "http://127.0.0.1:8000/v1", "host": "box"},
           "caveats": ["served from http://192.168.1.5:8000/v1"]}
    changed = scrub(doc)
    assert doc["model"] == "Qwen3-8B"
    assert doc["env"]["endpoint"] == "http://127.0.0.1:8000/v1"
    assert "192.168" not in doc["caveats"][0]
    assert set(changed) == {"/model", "/caveats/0"}


def test_parse_thresholds_default():
    from qualitybench.compare import default_thresholds, gates_from
    rows = parse_thresholds(default_thresholds())
    assert [r["id"] for r in rows] == ["T1", "T2", "T3", "T4", "T5", "T7"]
    assert gates_from(rows) == {"task_delta_pts": -2.0, "top1_pct": 93.0, "ppl_delta_pct": 3.0, "kld_mean": 0.05}


def test_t2_is_scored_on_its_own_bar():
    from qualitybench.compare import code_ppl_gate, gates_from, verdicts
    rows = [{"id": "T1", "metric": "ppl wiki", "rule": "delta <= +3.0 %"},
            {"id": "T2", "metric": "ppl code", "rule": "delta <= +1.0 %"}]
    gates = gates_from(rows)
    assert code_ppl_gate(rows, gates) == 1.0
    assert code_ppl_gate(rows[:1], gates) == 3.0  # no T2 row: T1's bar, as before
    arm = {"ppl": {"ppl-wiki": {"delta_pct": 2.0}, "ppl-code": {"delta_pct": 2.0}}}
    v = verdicts(arm, gates, [], code_ppl_gate(rows, gates))
    assert v["T1"]["pass"] and not v["T2"]["pass"]


def _npz(path, logits, model, sha="x"):
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    o = np.argsort(-lp, -1)[..., :64]
    np.savez(path, **{"kld.ids": o.astype(np.int32), "kld.lp": np.take_along_axis(lp, o, -1).astype(np.float16),
                      "kld.target": lp[..., 0].astype(np.float32)},
             meta=json.dumps({"model": model, "topk": 64, "eval_set_sha256": sha, "vocab": logits.shape[-1]}))
    return lp


def test_kld_tracks_true_divergence(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.standard_t(2, (2, 50, 1000)) * 2.5
    a = _npz(tmp_path / "ref.npz", base, "ref")
    b = _npz(tmp_path / "test.npz", base + rng.normal(0, 0.3, base.shape), "test")
    true = float((np.exp(a) * (a - b)).sum(-1).mean())
    r = kld.compare(tmp_path / "ref.npz", tmp_path / "test.npz")
    assert abs(r["kld"]["kld_mean"] - true) / true < 0.25
    same = kld.compare(tmp_path / "ref.npz", tmp_path / "ref.npz")
    assert same["kld"]["kld_mean"] < 1e-3 and same["kld"]["top1_agree_pct"] == 100.0
    assert sum(r["kld"]["hist"]["counts"]) == 100
    _npz(tmp_path / "other.npz", base, "o", sha="y")
    with pytest.raises(ValueError):
        kld.compare(tmp_path / "ref.npz", tmp_path / "other.npz")


def test_parity_compare():
    a = [{"text": "hi there", "tokens": ["hi", " there"], "top": [{"hi": 0}, {" there": -0.1, " you": -0.2}]}]
    b = [{"text": "hi you", "tokens": ["hi", " you"], "top": [{"hi": 0}, {" you": -0.1}]}]
    r = parity_compare(a, b)
    assert r["identical"] is False and r["mismatches"] == [{"i": 0, "token": 1, "a": " there", "b": " you", "a_top2_gap": 0.1}]
    assert parity_compare(a, a) == {"identical": True, "n": 1, "mismatches": []}


def test_cli_version_and_usage(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == "0.4.0"
    with pytest.raises(SystemExit) as e:
        main(["run", "--endpoint", "x"])
    assert e.value.code == 2
    assert main(["run", "--endpoint", "x", "--model", "m", "--out", "o.json", "--deadline", "soon"]) == 2
    assert main(["tasks", "--json"]) == 0


# --- estimate and progress ------------------------------------------------------------------------
def test_estimate_scales_with_server_speed_and_history():
    from qualitybench import estimate as e
    ref = {"decode_tps": e.REF_DECODE_TPS, "prefill_tps": e.REF_PREFILL_TPS}
    full = e.estimate(["gsm8k", "logprobs"], False, "off", "m", ref, history=[])
    assert [r["min"] for r in full["tasks"]] == [19, 16] and full["source"] == "probe"
    fast = e.estimate(["gsm8k"], False, "off", "m", {"decode_tps": e.REF_DECODE_TPS * 4, "prefill_tps": 1}, history=[])
    ov = suite.BY_ID["gsm8k"].overhead_s / 60
    assert abs(fast["tasks"][0]["min"] - (ov + (19 - ov) / 4)) < 0.1
    think = e.estimate(["gsm8k"], False, "on", "m", ref, history=[])
    assert think["tasks"][0]["min"] > 3 * full["tasks"][0]["min"]
    # mrcr is prefill-bound: thinking on changes nothing, a 10x faster prefill cuts it tenfold
    mr = e.estimate(["mrcr"], False, "on", "m", ref, history=[])
    assert mr["tasks"][0]["min"] == suite.BY_ID["mrcr"].ref_minutes and mr["tasks"][0]["items"] == 200
    quick_fast = e.estimate(["mrcr"], True, "off", "m", {"decode_tps": 1, "prefill_tps": e.REF_PREFILL_TPS * 10}, history=[])
    assert quick_fast["tasks"][0]["items"] == 20 and quick_fast["tasks"][0]["min"] < 3
    nothing = e.estimate(["gsm8k"], True, None, "m", None, history=[])
    assert nothing["source"] == "reference" and nothing["tasks"][0]["items"] == 132
    # an earlier quick run of this model: scale its wall time to the full item count
    o = suite.BY_ID["gsm8k"].overhead_s
    hist = [{"tasks": {"gsm8k": {"items": 132, "wall_s": o + 66}}, "probe": None}]
    h = e.estimate(["gsm8k", "ifeval"], False, "off", "m", None, history=hist)
    assert h["tasks"][0]["source"] == "history" and abs(h["tasks"][0]["min"] - (o + 66 * 1319 / 132) / 60) < 0.1
    assert h["source"] == "reference"  # ifeval had no history: the range says so


def test_confirm_modes(monkeypatch):
    from qualitybench import estimate as e
    d = {"total_min": 20}
    monkeypatch.delenv("QUALITYBENCH_CONFIRM", raising=False)
    assert e.confirm_mode() == "always" and e.needs_confirm(d, "always")
    assert not e.needs_confirm(d, "never")
    assert not e.needs_confirm(d, "over:30m") and e.needs_confirm({"total_min": 45}, "over:30m")
    assert e.needs_confirm({"total_min": 61}, "over:1h") and not e.needs_confirm(d, "over:25")
    monkeypatch.setenv("QUALITYBENCH_CONFIRM", "bogus")
    assert e.confirm_mode() == "always"


def test_tracker_eta_converges():
    from qualitybench.progress import Tracker
    now = [0.0]
    tr = Tracker({"a": 10, "b": 10}, ["a", "b"], clock=lambda: now[0])
    tr.start("a")
    assert tr.snapshot()["eta_s"] == 1200 and tr.snapshot()["fraction"] == 0
    # task a actually runs at twice its estimate: 20 min
    for minute in range(1, 21):
        now[0] = minute * 60
        tr.update("a", {"done": minute, "total": 20})
        if minute == 10:
            s = tr.snapshot()
            # a: half generated in 10 min, so 10 more plus its scoring share; b still at its estimate
            assert abs(s["eta_s"] - (733 + 600)) < 60
    tr.end("a")
    tr.start("b")
    s = tr.snapshot()
    assert abs(s["eta_s"] - 1200) < 60 and s["fraction"] == 0.5  # b corrected by a's 2x ratio
    now[0] += 600
    tr.update("b", {"done": 5, "total": 10})
    assert abs(tr.snapshot()["eta_s"] - 733) < 60
    tr.update("b", {"state": "scored"})
    tr.end("b")
    assert tr.snapshot()["eta_s"] == 0 and tr.snapshot()["fraction"] == 1.0


# ------------------------------------------------------------------ 0.3.0: GSM8K grader, caps, containers, no logprobs

@pytest.mark.parametrize("reply,gold,ok", [
    ("Sally has $3.00 left.", "3", True),                                   # gsm8k/410
    ("Peter paid $7.00 in total.", "7", True),                              # gsm8k/280
    ("Judy makes $7,425 in 1 week.", "7425", True),                         # gsm8k/70
    ("So **15** liters.\nThere are 15 liters of water in the remaining 24 liters.", "15", True),  # gsm8k/20, bolded
    ("Josh makes $120 in 2 weeks.", "120", True),                           # gsm8k/1300
    ("It will cost him $18.00 to purchase 6 pounds of butter.", "18", True),  # gsm8k/1070
    ("Answer: 42\nThat is 6 groups of 7.", "42", True),
    (r"The total is \boxed{1,200}.", "1200", True),
    ("x\n#### 5", "5", True),
    ("<think>maybe 9</think>She has 10 left.", "10", True),
    ("She has 9 left.", "10", False),
    ("I cannot tell.", "10", False),
])
def test_gsm8k_grader(reply, gold, ok):
    from qualitybench.items import grade_gsm8k
    assert grade_gsm8k(reply, f"some working\n#### {gold}") is ok


def _gsm8k_samples(path, replies):
    rows = []
    for i, (reply, gold) in enumerate(replies):
        for flt, em in (("strict-match", 0.0), ("flexible-extract", 0.0)):
            rows.append({"doc_id": i, "target": f"w\n#### {gold}", "resps": [[reply]], "filter": flt, "exact_match": em})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_gsm8k_items_keep_stock_and_regrade_old_runs(tmp_path):
    from qualitybench import items
    samples = tmp_path / "side" / "lmeval" / "gsm8k" / "m" / "samples_gsm8k_x.jsonl"
    _gsm8k_samples(samples, [("Sally has $3.00 left.", 3), ("It is 4.", 5)])
    recs = items.from_lmeval(samples, "gsm8k")
    assert [(r["id"], r["correct"], r["stock"]) for r in recs] == [("gsm8k/0", True, False), ("gsm8k/1", False, False)]
    # a 0.2.0 sidecar: stock-graded items, samples still there
    it = tmp_path / "side" / "items" / "gsm8k.jsonl"
    it.parent.mkdir(parents=True)
    items.write(it, [{"id": "gsm8k/0", "correct": False}, {"id": "gsm8k/1", "correct": False}])
    assert items.regrade_gsm8k(tmp_path / "side") is True
    assert items.load(it) == {"gsm8k/0": True, "gsm8k/1": False}
    # no samples left: can't regrade, said so
    bare = tmp_path / "bare" / "items" / "gsm8k.jsonl"
    bare.parent.mkdir(parents=True)
    items.write(bare, [{"id": "gsm8k/0", "correct": False}])
    assert items.regrade_gsm8k(tmp_path / "bare") is False
    assert items.regrade_gsm8k(tmp_path / "none") is None


def test_container_run_podman_flags(monkeypatch, tmp_path):
    from qualitybench import envs
    monkeypatch.setenv("QUALITYBENCH_CONTAINER", "podman")
    cmd = envs.container_run("none", [(tmp_path, "/work", "rw")])
    assert cmd[:2] == ["podman", "run"] and "--userns=keep-id" in cmd and "label=disable" in cmd
    assert f"{tmp_path}:/work:rw" in cmd
    monkeypatch.setenv("QUALITYBENCH_CONTAINER", "docker")
    cmd = envs.container_run("none", [])
    assert cmd[0] == "docker" and "--userns=keep-id" not in cmd and "label=disable" not in cmd
    assert envs.EVALPLUS_IMAGE.startswith("docker.io/")  # podman refuses short names without a TTY
    monkeypatch.setenv("QUALITYBENCH_CONTAINER", "lxc")
    with pytest.raises(SystemExit):
        envs.container()


def _side(tmp_path, state, thinking="server default"):
    from qualitybench import items
    side = tmp_path / "r.qb"
    (side / "items").mkdir(parents=True)
    (side / "ENV.json").write_text(json.dumps({"model": "m", "thinking": thinking, "date": "2026-11-09"}))
    (side / "state.json").write_text(json.dumps(state))
    items.write(side / "items" / "lcb.jsonl", [{"id": str(i), "correct": i < 5} for i in range(20)])
    return side


def test_filtered_reply_stands_in_only_for_the_content_filter():
    from qualitybench.proxy import filtered_reply
    azure = b'{"error":{"message":"litellm.BadRequestError: Azure_aiException - {\\"code\\":\\"content_filter\\"}"}}'
    req = json.dumps({"model": "m", "messages": [{"role": "user", "content": "q"}]}).encode()
    data, ctype = filtered_reply(req, 400, azure)
    d = json.loads(data)
    assert ctype == "application/json" and d["model"] == "m"
    assert d["choices"] == [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "content_filter"}]
    assert d["usage"]["completion_tokens"] == 0  # a counted reply, so the task's token counts stay whole
    assert filtered_reply(req, 400, b'{"error":{"innererror":{"code":"ResponsibleAIPolicyViolation"}}}') is not None
    assert filtered_reply(req, 400, b'{"error":{"message":"context too long"}}') is None
    assert filtered_reply(req, 500, azure) is None
    data, ctype = filtered_reply(json.dumps({"model": "m", "stream": True}).encode(), 400, azure)
    assert ctype == "text/event-stream" and data.endswith(b"\n\ndata: [DONE]\n\n")
    assert json.loads(data.split(b"\n")[0][5:])["choices"][0]["finish_reason"] == "content_filter"


def test_content_filtered_prompts_are_a_caveat(tmp_path):
    from qualitybench import results
    doc = results.build(_side(tmp_path, {"caps": {"lcb": {"replies": 20, "length": 0, "reasoning": 0, "filtered": 2}}}, "off"))
    assert ("LiveCodeBench v6: the server's content filter refused 2 of its prompts; they count as wrong, so this "
            "score understates the model.") in doc["caveats"]
    assert results.validate_shape(doc) == []
    quiet = results.build(_side(tmp_path / "q", {"caps": {"lcb": {"replies": 20, "length": 0, "reasoning": 0}}}, "off"))
    assert not any("content filter" in c for c in quiet["caveats"])  # caps from before the count


def test_run_lmeval_caches_replies_and_starts_caps_with_the_cache(tmp_path):
    from qualitybench.context import Ctx, TaskFailed
    ctx = Ctx(model="m", endpoint="x", side=tmp_path / "r.qb")
    ctx.load_state()
    cmds = []

    def failed(cmd, task, env=None, total=None):
        cmds.append(cmd)
        raise TaskFailed("lm-eval exited with status 1")

    ctx.sh = failed
    cache = tmp_path / "r.qb" / "lmeval" / "mmlu_pro-replies"
    ctx.state["caps"] = {"mmlu_pro": {"replies": 990, "length": 0}}  # counted before there was a cache
    with pytest.raises(TaskFailed):
        harness.run_lmeval(ctx, "mmlu_pro")
    assert cmds[0][cmds[0].index("--use_cache") + 1] == cache
    assert "mmlu_pro" not in json.loads((tmp_path / "r.qb" / "state.json").read_text())["caps"]
    # the cache holds the replies counted since, so a rerun's replies add up with those
    harness.lmeval_cache_db(cache).write_bytes(b"")
    ctx.state["caps"]["mmlu_pro"] = {"replies": 990, "length": 0}
    with pytest.raises(TaskFailed):
        harness.run_lmeval(ctx, "mmlu_pro")
    assert ctx.state["caps"]["mmlu_pro"]["replies"] == 990


def test_token_cap_caveats_and_no_logprobs_is_submittable(tmp_path):
    from qualitybench import results
    side = _side(tmp_path, {"caps": {"lcb": {"replies": 20, "length": 6, "reasoning": 20, "thinking": ""}}})
    doc = results.build(side)
    cav = " ".join(doc["caveats"])
    assert "6 of 20 replies (30%) stopped at the token cap" in cav
    assert "thinking was left at the server's default" in doc["caveats"][0]
    assert "No logprobs" in cav and doc["logprobs"] == {}
    assert results.validate_shape(doc) == []  # task scores alone may be submitted
    quiet = results.build(_side(tmp_path / "q", {"caps": {"lcb": {"replies": 200, "length": 1, "reasoning": 0}}}, "off"))
    assert not any("token cap" in c for c in quiet["caveats"])  # under 1%: no caveat
    assert not any("Output tokens" in c for c in quiet["caveats"])  # caps from 0.4.0 and earlier have no tokens


def test_tokens_summary_rounds_to_whole_tokens():
    assert stats.tokens_summary([100, 200, 600]) == {"n": 3, "mean": 300, "median": 200}
    assert stats.tokens_summary([1, 2, 4, 10]) == {"n": 4, "mean": 4, "median": 3}  # even length: mean of the middle two
    assert stats.tokens_summary([3, 4]) == {"n": 2, "mean": 4, "median": 4}  # 3.5 rounds to 4
    assert stats.tokens_summary([]) is None


def test_tokens_summary_thinking_only_when_every_reply_has_some():
    plain = {"n": 3, "mean": 300, "median": 200}
    assert stats.tokens_summary([100, 200, 600], [50, 150, 400]) == {**plain, "thinking": {"mean": 200, "median": 150}}
    assert stats.tokens_summary([100, 200, 600], [50, 150]) == plain  # not one count per reply
    assert stats.tokens_summary([100, 200, 600], [50, None, 400]) == plain  # a reply without the server's count
    assert stats.tokens_summary([100, 200, 600], [0, 0, 0]) == plain  # thinking off
    assert stats.tokens_summary([100, 200, 600], [0, 0, 9]) == {**plain, "thinking": {"mean": 3, "median": 0}}
    assert stats.tokens_summary([], []) is None


def test_output_tokens_caveat(tmp_path):
    from qualitybench import results
    lcb = [1000] * 5 + [2000] * 10 + [6000] * 5  # one count per item (_side writes 20 lcb items)
    caps = {"lcb": {"replies": 20, "length": 0, "reasoning": 0, "tokens": lcb},
            "gsm8k": {"replies": 1, "length": 0, "reasoning": 0, "tokens": [50]}}  # gsm8k has no items: not run
    doc = results.build(_side(tmp_path, {"caps": caps}, "off"))
    note = "Output tokens per reply as the server counted them, mean (median): LiveCodeBench v6 2,750 (2,000)"
    assert note in doc["caveats"]
    # 19 counts for 20 items: a reply went uncounted (e.g. a killed run's first attempt), so no stats at all
    short = {"lcb": {**caps["lcb"], "tokens": lcb[:-1]}}
    assert not any(c.startswith(results.TOKENS_NOTE) for c in results.build(_side(tmp_path / "s", {"caps": short}, "off"))["caveats"])
    full = results.build(_side(tmp_path / "f", {"caps": caps}, "off"), notes=[f"n{i}" for i in range(11)])
    assert note in full["caveats"]
    # before the --note caveats too: with 12 others ahead of it, a note gives way, not the tokens line
    on = results.build(_side(tmp_path / "o", {"caps": caps}, "on"), notes=[f"n{i}" for i in range(11)])
    assert on["caveats"][0].startswith("This run had thinking ON") and on["caveats"][1] == note
    assert on["caveats"][2] == "Note: n0" and len(on["caveats"]) == 12 and "Note: n10" not in on["caveats"]
    none = {"lcb": {"replies": 3, "length": 0, "reasoning": 0, "tokens": []}}
    assert not any("Output tokens" in c for c in results.build(_side(tmp_path / "n", {"caps": none}, "off"))["caveats"])


def test_output_tokens_caveat_pairs_thinking_off_and_on(tmp_path):
    from qualitybench import items, results
    counted = lambda toks: {"replies": len(toks), "length": 0, "reasoning": 0, "tokens": toks}
    off = _side(tmp_path / "off", {"caps": {"gsm8k": counted([10]), "ifeval": {"replies": 1, "length": 0, "reasoning": 0},
                                            "lcb": counted([300, 320] * 10)}}, "off")  # 20 lcb items, one count each
    on = _side(tmp_path / "on", {"caps": {"gsm8k": counted([999]), "ifeval": counted([50]),
                                          "lcb": counted([2500] * 15 + [11000] * 5)}}, "on")
    for side, gsm in ((off, "g1"), (on, "g2")):  # gsm8k: other items on the thinking-on side, so not paired
        items.write(side / "items" / "gsm8k.jsonl", [{"id": gsm, "correct": True}])
        items.write(side / "items" / "ifeval.jsonl", [{"id": "i1", "correct": True}])
    doc = results.build(off, on_side=on, on_src="on")
    assert ("Output tokens per reply as the server counted them, mean (median): GSM8K off 10 (10) · IFEval on 50 (50) · "
            "LiveCodeBench v6 off 310 (310), on 4,625 (2,500)") in doc["caveats"]


def test_output_tokens_caveat_says_thinking_tokens(tmp_path):
    from qualitybench import results
    lcb = {"replies": 20, "length": 0, "reasoning": 20, "tokens": [300, 320] * 10, "thinking_tokens": [200, 220] * 10}
    one = results.build(_side(tmp_path / "one", {"caps": {"lcb": lcb}}, "on"))
    assert ("Output tokens per reply as the server counted them, mean (median): LiveCodeBench v6 310 (310), "
            "thinking 210 (210)") in one["caveats"]
    # the thinking part follows the side it belongs to; the thinking-off run's zeros say nothing
    off = _side(tmp_path / "off", {"caps": {"lcb": {**lcb, "thinking_tokens": [0] * 20}}}, "off")
    on = _side(tmp_path / "on", {"caps": {"lcb": {**lcb, "tokens": [2500] * 15 + [11000] * 5,
                                                  "thinking_tokens": [2100] * 15 + [9300] * 5}}}, "on")
    pair = results.build(off, on_side=on, on_src="on")
    assert ("Output tokens per reply as the server counted them, mean (median): LiveCodeBench v6 off 310 (310), "
            "on 4,625 (2,500), thinking 3,900 (2,100)") in pair["caveats"]
    # caps kept before thinking counts were (tokens only): totals as before
    old = results.build(_side(tmp_path / "old", {"caps": {"lcb": {k: v for k, v in lcb.items() if k != "thinking_tokens"}}}, "on"))
    assert "Output tokens per reply as the server counted them, mean (median): LiveCodeBench v6 310 (310)" in old["caveats"]


def test_pct_change():
    from qualitybench.compare import _pct
    assert _pct(800, 1000) == "+25%" and _pct(1000, 800) == "-20%"
    assert _pct(0, 5) == "n/a"  # no division by a zero reference
    # halves round away from zero, as the report's table does (Math.floor(|p| + 0.5)), not to even
    assert _pct(200, 205) == "+3%" and _pct(200, 195) == "-3%" and _pct(200, 203) == "+2%" and _pct(200, 200) == "+0%"


def test_record_caps_adds_up_resumed_tasks(tmp_path):
    from qualitybench.context import Ctx
    from qualitybench.runner import _record_caps
    ctx = Ctx(model="m", endpoint="x", side=tmp_path / "r.qb", thinking="off")
    ctx.load_state()
    was = {"replies": 2, "length": 1, "reasoning": 0, "thinking": "off", "max_tokens": 0}
    ctx.state["caps"] = {"lcb": {**was, "tokens": [10, 20]}, "bfcl": {**was, "tokens": [10, 20]},
                         "mbpp": was,  # caps from 0.4.0: no tokens
                         "humaneval": {**was, "tokens": []}}  # a server that sent no usage
    new = lambda: {"replies": 1, "length": 0, "reasoning": 0, "tokens": [60], "thinking": [40]}
    for t in ("lcb", "bfcl", "mbpp", "humaneval"):
        _record_caps(ctx, t, new())
    caps = json.loads((tmp_path / "r.qb" / "state.json").read_text())["caps"]
    # "thinking" in caps is the run's --thinking, so the replies' thinking tokens are kept as thinking_tokens,
    # aligned with tokens: the earlier replies' counts are unknown
    assert caps["lcb"] == {"replies": 3, "length": 1, "reasoning": 0, "tokens": [10, 20, 60],
                           "thinking_tokens": [None, None, 40], "thinking": "off",
                           "max_tokens": 0}  # a rerun asks only for the missing items
    assert caps["bfcl"] == {"replies": 1, "length": 0, "reasoning": 0, "tokens": [60], "thinking_tokens": [40],
                             "thinking": "off", "max_tokens": 0}
    # kept as counted; results publish a task's counts only if every item's reply has one
    assert caps["mbpp"]["replies"] == 3 and caps["mbpp"]["tokens"] == [60] and caps["mbpp"]["thinking_tokens"] == [40]
    assert caps["humaneval"]["replies"] == 3 and caps["humaneval"]["tokens"] == [60]
    assert caps["humaneval"]["thinking_tokens"] == [40]
    log = (tmp_path / "r.qb" / "run.log").read_text()
    assert [l.split("] ", 1)[1] for l in log.splitlines() if "output tokens per reply" in l] == [
        "lcb: 3 replies, 30 output tokens per reply (mean), 20 (median)",
        "bfcl: 1 reply, 60 output tokens per reply (mean), 60 (median)",
        "mbpp: 1 reply, 60 output tokens per reply (mean), 60 (median)",
        "humaneval: 1 reply, 60 output tokens per reply (mean), 60 (median)"]
    _record_caps(ctx, "lcb", new())  # resumed again: both lists grow together
    caps = json.loads((tmp_path / "r.qb" / "state.json").read_text())["caps"]
    assert caps["lcb"]["tokens"] == [10, 20, 60, 60] and caps["lcb"]["thinking_tokens"] == [None, None, 40, 40]


def test_cap_caveat_follows_thinking(tmp_path):
    from qualitybench import results

    def cav(thinking, reasoning):
        caps = {"lcb": {"replies": 20, "length": 6, "reasoning": reasoning, "thinking": thinking}}
        doc = results.build(_side(tmp_path / f"{thinking}{reasoning}", {"caps": caps}, thinking))
        assert results.validate_shape(doc) == []
        return " ".join(doc["caveats"])
    on = cav("on", 20)
    assert "understates the model; raise --thinking-max-tokens." in on and "reasoned in" not in on
    off = cav("off", 3)
    assert ("understates the model; 3 of them reasoned although thinking was off, so the server may not honour "
            "enable_thinking: check its chat template, or run with --thinking on.") in off
    assert "The model reasoned in 3 replies although thinking was off (enable_thinking=false)" in off
    plain = cav("off", 0)
    assert ("understates the model; with thinking off the harness's own limit applies (2048 tokens, 4096 for "
            "LiveCodeBench) and the model writes longer replies than that.") in plain
    assert "reasoned" not in plain and "--thinking on" not in plain


def test_record_caps_warns_when_thinking_off_still_reasons(tmp_path):
    from qualitybench.context import Ctx
    from qualitybench.runner import _record_caps
    ctx = Ctx(model="m", endpoint="x", side=tmp_path / "r.qb", thinking="off")
    ctx.load_state()
    _record_caps(ctx, "mmlu_pro", {"replies": 1000, "length": 125, "reasoning": 40, "tokens": [8] * 1000})
    assert ("mmlu_pro: WARNING: 40 of 1000 replies had reasoning although --thinking off was set; the server may not "
            "honour enable_thinking") in (tmp_path / "r.qb" / "run.log").read_text()


def test_tokenizer_failure_names_the_reason(tmp_path, monkeypatch):
    # build_inputs.py prints "tokenizer: <why>" as its last line; ensure_inputs makes that the task's failure
    import sys
    from qualitybench.context import Ctx, TaskFailed
    from qualitybench.tasks import native
    script = tmp_path / "build_inputs.py"
    script.write_text('print("loading")\nraise SystemExit("tokenizer: org/m is a gated Hugging Face repo and no HF_TOKEN is set")\n')
    monkeypatch.setenv("QUALITYBENCH_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(native.envs, "ensure_venv", lambda name: None)
    monkeypatch.setattr(native.envs, "venv_python", lambda name: sys.executable)
    monkeypatch.setattr(native.envs, "pkg_file", lambda *p: script)
    for tok, hint in ((None, True), ("org/m", False)):
        ctx = Ctx(model="org/m", endpoint="x", side=tmp_path / "r.qb", tokenizer=tok)
        with pytest.raises(TaskFailed) as e:
            native.ensure_inputs(ctx, "evalset")
        msg = str(e.value)
        assert msg.startswith("org/m is a gated Hugging Face repo and no HF_TOKEN is set")
        assert ("pass --tokenizer" in msg) is hint and "exited with status" not in msg
    script.write_text("raise SystemExit(3)\n")  # any other failure keeps the generic message, without the hint
    with pytest.raises(TaskFailed, match="exited with status 3.*for --tokenizer org/m"):
        native.ensure_inputs(Ctx(model="vllm", endpoint="x", side=tmp_path / "r.qb", tokenizer="org/m"), "evalset")


def test_run_skips_tasks_whose_inputs_cannot_be_built(tmp_path, monkeypatch):
    import sys
    from qualitybench.context import Ctx
    from qualitybench.runner import precheck_inputs
    from qualitybench.tasks import native
    script = tmp_path / "build_inputs.py"
    script.write_text('raise SystemExit("tokenizer: gpt-6.1-sol is neither a local dir nor a Hugging Face repo you can read")\n')
    monkeypatch.setenv("QUALITYBENCH_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(native.envs, "ensure_venv", lambda name: None)
    monkeypatch.setattr(native.envs, "venv_python", lambda name: sys.executable)
    monkeypatch.setattr(native.envs, "pkg_file", lambda *p: script)
    ctx = Ctx(model="gpt-6.1-sol", endpoint="x", side=tmp_path / "r.qb")
    ctx.load_state()
    assert precheck_inputs(ctx, ["logprobs", "needle", "gsm8k"]) == ["gsm8k"]  # the non-tokenizing task is untouched
    state = json.loads((tmp_path / "r.qb" / "state.json").read_text())["partials"]
    for t in ("logprobs", "needle"):
        assert "skipped before running" in state[t] and "neither a local dir" in state[t]


def test_precheck_builds_inputs_and_keeps_runnable_tasks(tmp_path, monkeypatch):
    from qualitybench.context import Ctx
    from qualitybench.runner import precheck_inputs
    from qualitybench.tasks import native
    what = []
    monkeypatch.setattr(native, "ensure_inputs", lambda ctx, w: what.append(w))
    ctx = Ctx(model="m", endpoint="x", side=tmp_path / "r.qb")
    ctx.load_state()
    assert precheck_inputs(ctx, ["gsm8k", "needle", "logprobs"]) == ["gsm8k", "needle", "logprobs"]
    assert what == ["needle", "evalset"]  # only the tokenizing tasks are probed


def test_missing_eval_set_skips_logprobs(tmp_path):
    from qualitybench.context import Ctx
    from qualitybench.runner import precheck_inputs
    ctx = Ctx(model="m", endpoint="x", side=tmp_path / "r.qb", eval_set=str(tmp_path / "nope.json"))
    ctx.load_state()
    assert precheck_inputs(ctx, ["logprobs"]) == []
    assert "does not exist" in (tmp_path / "r.qb" / "state.json").read_text()


def test_request_params_parse_nest_and_refuse_pinned_fields():
    from qualitybench.proxy import describe, parse_params
    p = parse_params(["reasoning_effort=high", "chat_template_kwargs.reasoning_budget=8192", "x.flag=true", "name=a=b"])
    assert p == {"reasoning_effort": "high", "chat_template_kwargs": {"reasoning_budget": 8192}, "x": {"flag": True},
                 "name": "a=b"}
    assert describe(p) == "reasoning_effort=high, chat_template_kwargs.reasoning_budget=8192, x.flag=true, name=a=b"
    for bad in ("temperature=0.6", "top_p=0.95", "seed=1", "max_tokens=99", "stop=x", "messages=[]",
                "chat_template_kwargs.enable_thinking=false", "noequals", "a..b=1"):
        with pytest.raises(ValueError):
            parse_params([bad])
    with pytest.raises(ValueError, match="--thinking-max-tokens"):
        parse_params(["max_tokens=32768"])
    with pytest.raises(ValueError, match="already set"):
        parse_params(["a=1", "a.b=2"])
    assert main(["run", "--endpoint", "http://127.0.0.1:9", "--model", "m", "--out", "x.json",
                 "--request-param", "temperature=0.7"]) == 2

def test_mmlu_pro_reads_markdown_answers_only_when_lm_eval_found_none(tmp_path):
    from qualitybench.items import from_lmeval
    rows = [("Therefore, the answer is **(D)**.", "[invalid]", "D", 0, True),
            ("The answer is: **D**", "[invalid]", "D", 0, True),
            ("the answer is _(C)_", "[invalid]", "C", 0, True),
            ("Therefore, the answer is **(D)**.", "[invalid]", "E", 0, False),
            ("I am not sure.", "[invalid]", "A", 0, False),
            ("the answer is Freddie Mac, so the answer is (H)", "F", "H", 0, False),  # lm-eval's letter stands
            ("the answer is (B)", "B", "B", 1, True)]
    f = tmp_path / "samples_mmlu_pro_law.jsonl"
    f.write_text("".join(json.dumps({"doc_id": i, "filter": "custom-extract", "resps": [[reply]], "filtered_resps": [got],
                                     "target": want, "exact_match": em}) + "\n"
                         for i, (reply, got, want, em, _) in enumerate(rows)))
    assert [r["correct"] for r in from_lmeval(f, "mmlu_pro_law")] == [r[4] for r in rows]


def test_report_tokens_keeps_only_what_it_knows(tmp_path):
    from qualitybench import results
    counted = lambda toks: {"caps": {"lcb": {"replies": 20, "length": 0, "reasoning": 0, "tokens": toks}}}
    ref, test = _side(tmp_path / "ref", counted([300, 320] * 10), "off"), _side(tmp_path / "test", counted([100] * 20), "on")
    s = lambda m: {"lcb": {"n": 20, "mean": m, "median": m}}
    assert results.report_tokens({"tasks": [{"id": "lcb"}]}, ref) == {"ref": s(310)}
    # the thinking-on run's counts only where its scores pair with the reference's, as in the caveat
    assert results.report_tokens({"tasks": [{"id": "lcb"}]}, ref, on_side=test) == {"ref": s(310), "on": {}}
    assert results.report_tokens({"tasks": [{"id": "lcb", "on": {}}]}, ref, on_side=test) == {"ref": s(310), "on": s(100)}
    # the gate arm, on its compared tasks only
    doc = {"tasks": [{"id": "lcb"}], "gate_arm": "new", "arms": {"old": {"tasks": {"lcb": {}}}, "new": {"tasks": {"lcb": {}}}}}
    assert results.report_tokens(doc, ref, arm_side=test) == {"ref": s(310), "arms": {"new": s(100)}}
    doc["arms"]["new"] = {}  # no task compared (e.g. other items)
    assert results.report_tokens(doc, ref, arm_side=test) == {"ref": s(310), "arms": {"new": {}}}


def test_report_embeds_output_tokens_outside_the_document(tmp_path, capsys):
    import shutil
    from qualitybench import report, results
    side = _side(tmp_path, {"caps": {"lcb": {"replies": 20, "length": 0, "reasoning": 0, "tokens": [300, 320] * 10}}}, "off")
    res = tmp_path / "r.json"
    results.write(results.build(side), res)
    doc = json.loads(res.read_text())
    tokens = {"ref": {"lcb": {"n": 20, "mean": 310, "median": 310}}, "arms": {"q4</script>": {}}}
    html = report.render(doc, tokens)
    assert 'const T = {"ref":{"lcb":{"n":20,"mean":310,"median":310}},"arms":{"q4<\\/script>":{}}};' in html
    assert html.count("</script>") == 1  # the data never ends the script block
    assert '<section id="s-tokens">' in html and "<h2>How much the model writes</h2>" in html
    assert html.index('id="s-tasks"') < html.index('id="s-tokens"') < html.index('id="s-think"')
    # the card's columns and the run labels its cells pair ("reference" in a comparison, thinking off/on on a pair)
    for col in ('"Typical reply"', '"Whole benchmark"', '"reference"', '"thinking off"', '"thinking on"', "All tasks"):
        assert col in html
    assert "const T = null;" in report.render(doc)
    # write() keeps them next to the results file, so `qualitybench report` draws the same table later
    report.write(doc, res, tokens=tokens)
    assert json.loads((tmp_path / "r.qb" / "tokens.json").read_text()) == tokens
    # no task counted anywhere: no tokens.json (nor a sidecar made for it), and none left from an earlier write
    report.write(doc, tmp_path / "none.json", tokens={"ref": {}, "on": {}, "arms": {"q4": {}}})
    assert (tmp_path / "none.html").is_file() and not (tmp_path / "none.qb").exists()
    report.write(doc, res, tokens={"ref": {}})
    assert not (tmp_path / "r.qb" / "tokens.json").exists()
    report.write(doc, res, tokens=tokens)
    (tmp_path / "r.html").unlink()
    assert main(["report", str(res)]) == 0
    assert 'const T = {"ref":{"lcb":{"n":20,"mean":310,"median":310}},"arms":{"q4<\\/script>":{}}};' in (tmp_path / "r.html").read_text()
    (tmp_path / "r.qb" / "tokens.json").unlink()  # a run from before tokens.json: its own counts, from state.json
    assert main(["report", str(res)]) == 0
    assert 'const T = {"ref":{"lcb":{"n":20,"mean":310,"median":310}}};' in (tmp_path / "r.html").read_text()
    # a comparison or thinking pair: its sidecar's counts may be another run's than the reference's, so none
    shutil.copytree(side, tmp_path / "p.qb")
    (tmp_path / "p.qb" / "tokens.json").unlink()
    (tmp_path / "p.json").write_text(json.dumps({**doc, "think_dirs": ["on"]}))
    assert main(["report", str(tmp_path / "p.json")]) == 0
    assert "const T = null;" in (tmp_path / "p.html").read_text()
    (tmp_path / "p.json").write_text(json.dumps({**doc, "arms": {"q4": {"label": "q4"}}, "gate_arm": "q4"}))
    assert main(["report", str(tmp_path / "p.json")]) == 0
    assert "const T = null;" in (tmp_path / "p.html").read_text()
    # a results file without its sidecar renders without the table
    (tmp_path / "bare.json").write_text(res.read_text())
    assert main(["report", str(tmp_path / "bare.json")]) == 0
    assert "const T = null;" in (tmp_path / "bare.html").read_text() and not (tmp_path / "bare.qb").exists()


def test_proxy_task_cap_lowers_the_thinking_floor():
    import json
    from qualitybench.proxy import Proxy
    body = lambda **kw: json.dumps({"model": "m", "messages": [], **kw}).encode()
    on = Proxy("http://127.0.0.1:1", thinking="on", min_max_tokens=65536)
    out = json.loads(on.rewrite("/v1/chat/completions", body(max_tokens=3000, qb_max_tokens_cap=3000)))
    assert out["max_tokens"] == 3000 and "qb_max_tokens_cap" not in out
    assert json.loads(on.rewrite("/v1/chat/completions", body(max_tokens=3000)))["max_tokens"] == 65536  # no cap: the floor
    assert json.loads(on.rewrite("/v1/chat/completions", body(max_tokens=70000, qb_max_tokens_cap=3000)))["max_tokens"] == 70000  # never lowers the ask
    for p in (Proxy("http://127.0.0.1:1", thinking="off"), Proxy("http://127.0.0.1:1")):  # the field never reaches the server
        out = json.loads(p.rewrite("/v1/chat/completions", body(max_tokens=2048, qb_max_tokens_cap=2048)))
        assert out["max_tokens"] == 2048 and "qb_max_tokens_cap" not in out


def test_every_task_is_explained(capsys):
    import json
    from qualitybench import report
    for t in suite.SUITE:
        assert t.about and t.about.endswith(".") and len(t.about) <= 240 and t.about.isascii(), t.id
    html = report.render({"title": "x", "model": "m", "tasks": []})
    assert "/*__ABOUT__*/" not in html and all(t.about in html for t in suite.SUITE)  # every description ships in the page
    assert "What the benchmarks measure" in html
    assert main(["tasks", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == [t.id for t in suite.SUITE] and all(r["about"] for r in rows)
    assert main(["tasks"]) == 0
    assert suite.BY_ID["gsm8k"].about in capsys.readouterr().out
