"""The run loop end to end against a fake server: native tasks, resume, deadline, compare, schema, proxy."""
import hashlib
import json
import time
from pathlib import Path

import jsonschema
import pytest

from qualitybench import http as qhttp
from qualitybench.cli import main
from qualitybench.proxy import Proxy
from qualitybench.tasks.native import inputs_dir

from fakeserver import Fake

SCHEMA = json.loads((Path(__file__).parent / "schema" / "bench-qualitybench-v1.json").read_text())


def publishable(doc):
    """What L80 sends: the results minus env.endpoint/env.host, plus $template."""
    d = json.loads(json.dumps(doc))
    d["env"].pop("endpoint", None)
    d["env"].pop("host", None)
    d["$template"] = "bench.qualitybench.v1"
    jsonschema.Draft202012Validator(SCHEMA).validate(d)
    return d


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "qbhome"
    monkeypatch.setenv("QUALITYBENCH_HOME", str(h))
    qhttp.CANCEL.clear()
    # fixed inputs for the fake tokenizer "tok": a small eval set and needle prompts
    sets = {"ppl-wiki": [[(i * 7 + j) % 1000 for j in range(64)] for i in range(4)],
            "ppl-code": [[(i * 11 + j) % 1000 for j in range(64)] for i in range(4)],
            "kld": [[(i * 13 + j) % 1000 for j in range(64)] for i in range(12)]}
    es = dict(sets, meta={"sha256_ids": hashlib.sha256(json.dumps(sets).encode()).hexdigest(), "vocab": 1000})
    d = inputs_dir("tok")
    d.mkdir(parents=True)
    (d / "eval-set.json").write_text(json.dumps(es))
    prompts = [{"length": L, "depth": round(k / 9, 3), "number": str(1000000 + 10 * k + L // 1024),
                "prompt": ("hay " * (L // 4)) + f"The magic number for the blue lighthouse is {1000000 + 10 * k + L // 1024}."}
               for L in (32768, 131072) for k in range(10)]
    (d / "needle.json").write_text(json.dumps({"prompts": prompts}))
    return h


def run(fake, out, *extra):
    return main(["run", "--endpoint", fake.url, "--model", "fake", "--out", str(out), "--tokenizer", "tok",
                 "--tasks", "logprobs,needle,spec_parity", "--skip-setup", "--concurrency", "4", *extra])


def test_run_writes_valid_results_and_resumes(home, tmp_path, capsys):
    fake = Fake()
    try:
        out = tmp_path / "ref.json"
        assert run(fake, out, "--note", "precision=BF16", "--note", "made in a test") == 0
        o = capsys.readouterr().out
        assert 'qb:progress {"task":"logprobs","done":0,"total":20,"elapsed_s":' in o
        # the estimate comes first, then run-level lines with the time left
        est = json.loads(next(l for l in o.splitlines() if l.startswith("qb:estimate "))[12:])
        assert est["source"] == "probe" and [t["id"] for t in est["tasks"]] == ["logprobs", "needle", "spec_parity"]
        runs = [json.loads(l[12:])["run"] for l in o.splitlines() if l.startswith('qb:progress {"run"')]
        assert runs and runs[-1]["task_index"] == 3 and runs[-1]["fraction"] == 1.0 and runs[-1]["eta_s"] == 0
        assert o.index("qb:estimate ") < o.index('qb:progress {"task"')
        env = json.loads((tmp_path / "ref.qb" / "ENV.json").read_text())
        assert env["estimate"]["source"] == "probe" and "wall_min" in env
        published = json.loads(out.read_text())["env"]
        assert "estimate" not in published and "wall_min" not in published  # the schema has no room for them
        hist = [json.loads(l) for l in (home / "timings.jsonl").read_text().splitlines()]
        assert hist[-1]["model"] == "fake" and set(hist[-1]["tasks"]) == {"logprobs", "needle", "spec_parity"}
        assert '"state":"scored"' in o
        doc = json.loads(out.read_text())
        assert doc["schema_version"] == 1 and doc["model"] == "fake"
        assert set(doc["logprobs"]) == {"ppl-wiki", "ppl-code", "kld"}
        assert doc["env"]["precision"] == "BF16" and "Note: made in a test" in doc["caveats"]
        # 128k needles exceed the fake's context: skipped and listed, never silently dropped
        assert {i["length"] for i in doc["needle"]["items"]} == {32768}
        assert any("128k" in p for p in doc["partials"])
        assert (tmp_path / "ref.html").is_file() and (tmp_path / "ref.qb" / "parity.json").is_file()
        # the first request is the warmup/preflight, never part of a task
        first = [r for r in fake.requests if r[0] == "POST"][0]
        assert "single word OK" in first[2]["messages"][0]["content"]
        assert doc["tasks"] == []  # no scored task in this run, so launch80 would refuse it: say so, don't fake it
        n = len(fake.requests)
        assert run(fake, out) == 0  # resume: logprobs and parity are not asked again
        again = [r for r in fake.requests[n:] if r[1].endswith("/v1/completions") and "prompt" in (r[2] or {})]
        assert again == []
    finally:
        fake.close()


def test_compare_and_schema(home, tmp_path, capsys):
    ref, test = Fake(), Fake(noise=0.05)
    try:
        assert run(ref, tmp_path / "ref.json") == 0
        assert run(test, tmp_path / "test.json") == 0
        # give both runs the same scored task so the doc is publishable, with one changed item
        for name, flip in (("ref", False), ("test", True)):
            (tmp_path / f"{name}.qb" / "items").mkdir(exist_ok=True)
            items = [{"id": f"gsm8k/{i}", "correct": (i % 3 != 0) ^ (flip and i == 1)} for i in range(30)]
            (tmp_path / f"{name}.qb" / "items" / "gsm8k.jsonl").write_text("\n".join(map(json.dumps, items)) + "\n")
        assert main(["compare", str(tmp_path / "ref.json"), str(tmp_path / "test.json"), "--out", str(tmp_path / "cmp.json")]) == 0
        cmp_doc = json.loads((tmp_path / "cmp.json").read_text())
        arm = cmp_doc["arms"]["fake"]
        assert arm["kld"]["mean"] > 0 and 0 <= arm["top1"]["agree"] <= 100
        assert arm["spec_parity"]["identical"] is True
        assert cmp_doc["gate_arm"] == "fake"
        t3 = next(t for t in cmp_doc["thresholds"] if t["id"] == "T3")
        assert "fake" in t3["results"]
        # a results file built from the sidecar is valid under the server schema once it has a task
        from qualitybench import results
        doc = results.build(tmp_path / "ref.qb")
        assert doc["tasks"][0]["id"] == "gsm8k"
        publishable(doc)
        cmp_doc["tasks"] = doc["tasks"]
        publishable(cmp_doc)
        assert (tmp_path / "cmp.html").is_file()
    finally:
        ref.close()
        test.close()


def test_reference_flag_writes_comparison_to_out(home, tmp_path):
    ref, test = Fake(), Fake(noise=0.05)
    try:
        assert run(ref, tmp_path / "ref.json") == 0
        assert run(test, tmp_path / "test.json", "--reference", str(tmp_path / "ref.json"), "--arm", "int4") == 0
        doc = json.loads((tmp_path / "test.json").read_text())
        assert "int4" in doc["arms"] and (tmp_path / "test.qb" / "run.json").is_file()
    finally:
        ref.close()
        test.close()


def test_deadline_marks_tasks_partial(home, tmp_path):
    fake = Fake()
    try:
        assert run(fake, tmp_path / "r.json", "--deadline", "1ms") == 0
        doc = json.loads((tmp_path / "r.json").read_text())
        assert len(doc["partials"]) == 3 and all("deadline" in p for p in doc["partials"])
    finally:
        fake.close()


def test_thinking_preflight(home, tmp_path):
    fake = Fake(reasoning=False)
    try:
        assert run(fake, tmp_path / "r.json", "--thinking", "on") == 1
        fake.reasoning = True
        assert run(fake, tmp_path / "r.json", "--thinking", "on") == 0
        chat = [r[2] for r in fake.requests if r[1].endswith("/chat/completions")]
        probe = [b for b in chat if "chat_template_kwargs" not in b]  # the speed probe goes to the server directly
        chat = [b for b in chat if "chat_template_kwargs" in b]
        assert chat and all(b["chat_template_kwargs"]["enable_thinking"] is True for b in chat)
        assert all(b["max_tokens"] >= 32768 for b in chat)
        assert probe and all(b["max_tokens"] <= 128 for b in probe)  # never raised to the thinking floor
    finally:
        fake.close()


def test_default_thinking_model_is_told_to_pick(home, tmp_path, capsys):
    fake = Fake(reasoning=True)
    try:
        assert run(fake, tmp_path / "d.json") == 1
        assert "--thinking off" in (tmp_path / "d.qb" / "run.log").read_text()
    finally:
        fake.close()


def test_inline_think_counts_as_thinking(home, tmp_path):
    fake = Fake(inline=True)  # no reasoning parser: the <think> block arrives in content
    try:
        assert run(fake, tmp_path / "d.json") == 1
        assert "--thinking off" in (tmp_path / "d.qb" / "run.log").read_text()
        assert run(fake, tmp_path / "e.json", "--thinking", "on") == 0
    finally:
        fake.close()


def test_proxy_counts_capped_and_reasoning_replies(home):
    fake = Fake(reasoning=True, finish="length")
    try:
        with Proxy(fake.url) as px:
            c = qhttp.Client(px.url)
            for _ in range(3):
                c.chat("fake", [{"role": "user", "content": "hi"}], max_tokens=8)
            c.post("/v1/completions", {"model": "fake", "prompt": [1, 2, 3], "max_tokens": 1})  # not a chat
            assert px.take_stats() == {"replies": 3, "length": 3, "reasoning": 3}
            assert px.take_stats() == {"replies": 0, "length": 0, "reasoning": 0}
            fake.finish, fake.reasoning = "stop", False
            c.chat("fake", [{"role": "user", "content": "hi"}], max_tokens=8)
            assert px.take_stats() == {"replies": 1, "length": 0, "reasoning": 0}
    finally:
        fake.close()


def test_rerun_with_other_tier_is_refused(home, tmp_path):
    fake = Fake()
    try:
        assert run(fake, tmp_path / "r.json") == 0
        with pytest.raises(SystemExit):
            run(fake, tmp_path / "r.json", "--quick")
    finally:
        fake.close()


def test_proxy_adds_key_and_cancels_abandoned_requests(home, monkeypatch):
    fake = Fake()
    try:
        with Proxy(fake.url, api_key="sekrit", thinking="off") as px:
            c = qhttp.Client(px.url)
            c.get("/v1/models")
            assert fake.requests[-1][3].get("Authorization") == "Bearer sekrit"
            r = c.chat("fake", [{"role": "user", "content": "hi"}], stop=["x"])
            assert fake.requests[-1][2]["chat_template_kwargs"] == {"enable_thinking": False}
            assert fake.requests[-1][2]["stop"] == ["x"]
            # a client that gives up: the proxy must drop the upstream generation too
            slow = qhttp.Client(px.url, timeout=0.5, attempts=1)
            t0 = time.time()
            with pytest.raises(Exception):
                slow.chat("slow", [{"role": "user", "content": "x"}])
            assert fake.aborted.wait(5), "the upstream request was not cancelled"
            assert time.time() - t0 < 6
            assert sum(1 for r in fake.requests if (r[2] or {}).get("model") == "slow") == 1  # never resent
    finally:
        fake.close()


def test_max_minutes_refuses_before_any_task(home, tmp_path, capsys):
    fake = Fake()
    try:
        out = tmp_path / "r.json"
        assert run(fake, out, "--max-minutes", "0.01") == 3
        assert "not started" in capsys.readouterr().err
        assert not any(r[1].endswith("/v1/completions") for r in fake.requests)  # no logprobs work began
    finally:
        fake.close()


def test_tty_prompt_declined_and_accepted(home, tmp_path, monkeypatch, capsys):
    import io

    class TTY(io.StringIO):
        def isatty(self):
            return True

    fake = Fake()
    try:
        monkeypatch.setattr("sys.stdin", TTY("n\n"))
        assert run(fake, tmp_path / "a.json") == 0
        assert "Start? [y/N]" in capsys.readouterr().err
        assert not (tmp_path / "a.json").exists()
        monkeypatch.setattr("sys.stdin", TTY("y\n"))
        assert run(fake, tmp_path / "b.json") == 0 and (tmp_path / "b.json").is_file()
        monkeypatch.setattr("sys.stdin", TTY(""))  # --yes never reads the terminal
        assert run(fake, tmp_path / "c.json", "--yes") == 0 and (tmp_path / "c.json").is_file()
        monkeypatch.setenv("QUALITYBENCH_CONFIRM", "never")
        assert run(fake, tmp_path / "d.json") == 0 and (tmp_path / "d.json").is_file()
    finally:
        fake.close()


def test_estimate_command(home, capsys):
    fake = Fake()
    try:
        assert main(["estimate", "--endpoint", fake.url, "--model", "fake", "--quick", "--json"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["source"] == "probe" and d["probe"]["decode_tps"] > 0 and d["range_min"][0] <= d["total_min"]
        n = len(fake.requests)
        assert main(["estimate", "--endpoint", fake.url, "--model", "fake", "--json"]) == 0
        assert len(fake.requests) == n  # the probe is reused for 10 minutes
        capsys.readouterr()
    finally:
        fake.close()
    assert main(["estimate", "--endpoint", "http://127.0.0.1:9/v1", "--model", "other", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["source"] == "reference" and "could not measure" in d["warning"]
