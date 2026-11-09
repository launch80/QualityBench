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
    # L80's internal/qualitybench/suite.go holds the same ids, item counts and default flags.
    want = [("gsm8k", 1319, True), ("ifeval", 541, True), ("mmlu_pro", 1000, True), ("humaneval", 164, True),
            ("mbpp", 378, True), ("bfcl", 400, True), ("lcb", 175, True), ("logprobs", 592, True),
            ("needle", 20, True), ("spec_parity", 32, False)]
    assert [(t.id, t.items, t.default) for t in suite.SUITE] == want


def test_parse_tasks():
    assert suite.parse_tasks(None) == suite.default_ids()
    assert suite.parse_tasks("ifeval, GSM8K,ifeval") == ["gsm8k", "ifeval"]
    with pytest.raises(ValueError):
        suite.parse_tasks("gsm8k,nope")


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
    assert e.value.code == 0 and capsys.readouterr().out.strip() == "0.3.0"
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
    ov = e.OVERHEAD_S["gsm8k"] / 60
    assert abs(fast["tasks"][0]["min"] - (ov + (19 - ov) / 4)) < 0.1
    think = e.estimate(["gsm8k"], False, "on", "m", ref, history=[])
    assert think["tasks"][0]["min"] > 3 * full["tasks"][0]["min"]
    nothing = e.estimate(["gsm8k"], True, None, "m", None, history=[])
    assert nothing["source"] == "reference" and nothing["tasks"][0]["items"] == 132
    # an earlier quick run of this model: scale its wall time to the full item count
    o = e.OVERHEAD_S["gsm8k"]
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
