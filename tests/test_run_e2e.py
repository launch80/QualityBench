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
        for name, toks in (("ref", [1000, 1500, 2000] * 10), ("test", [600, 1200, 1200] * 10)):  # one per item
            st = tmp_path / f"{name}.qb" / "state.json"
            caps = {"gsm8k": {"replies": 30, "length": 0, "reasoning": 0, "tokens": toks}}
            st.write_text(json.dumps({**json.loads(st.read_text()), "caps": caps}))
        capsys.readouterr()
        assert main(["compare", str(tmp_path / "ref.json"), str(tmp_path / "test.json"), "--out", str(tmp_path / "cmp.json")]) == 0
        lines = capsys.readouterr().out.splitlines()
        i = next(i for i, l in enumerate(lines) if l.startswith("  gsm8k: "))
        assert lines[i + 1] == "    tokens/reply: mean 1,500 -> 1,000 (-33%), median 1,500 -> 1,200 (-20%)"
        cmp_doc = json.loads((tmp_path / "cmp.json").read_text())
        assert "fake vs reference, output tokens per reply (mean, median): GSM8K 1,500→1,000 (-33%, -20%)" in cmp_doc["caveats"]
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


def _scored_run(tmp_path, name, caps, notes=(), ifeval=30, thinking="off"):
    """A finished run's results file and sidecar, written by hand: gsm8k and ifeval items, state.json caps."""
    from qualitybench import items, results
    side = tmp_path / f"{name}.qb"
    (side / "items").mkdir(parents=True)
    (side / "ENV.json").write_text(json.dumps({"model": name, "thinking": thinking, "date": "2026-11-12"}))
    (side / "state.json").write_text(json.dumps({"caps": caps}))
    for t, n in (("gsm8k", 30), ("ifeval", ifeval)):
        items.write(side / "items" / f"{t}.jsonl", [{"id": f"{t}/{i}", "correct": i % 3 != 0} for i in range(n)])
    results.write(results.build(side, notes=list(notes)), tmp_path / f"{name}.json")
    return tmp_path / f"{name}.json"


def test_compare_tokens_caveat_survives_a_full_reference(home, tmp_path, capsys):
    caps = lambda n: {t: {"replies": 30, "length": 0, "reasoning": 0, "tokens": [n] * 30} for t in ("gsm8k", "ifeval")}
    # a reference results file with 12 caveats and no tokens caveat of its own (written before its counts were kept)
    ref = _scored_run(tmp_path, "ref", {}, notes=[f"n{i}" for i in range(12)])
    (tmp_path / "ref.qb" / "state.json").write_text(json.dumps({"caps": caps(800)}))
    assert len(json.loads(ref.read_text())["caveats"]) == 12
    test = _scored_run(tmp_path, "test", caps(1000), ifeval=20)  # ifeval: other items, not compared
    assert main(["compare", str(ref), str(test), "--arm-label", "Q4 build", "--out", str(tmp_path / "cmp.json")]) == 0
    o = capsys.readouterr()
    assert "ifeval not compared" in o.err and "output tokens not compared" not in o.err
    assert [l for l in o.out.splitlines() if "tokens/reply" in l] == [
        "    tokens/reply: mean 800 -> 1,000 (+25%), median 800 -> 1,000 (+25%)"]
    doc = publishable(json.loads((tmp_path / "cmp.json").read_text()))
    assert len(doc["caveats"]) == 12
    # before the --note caveats, which give way at 12; IFEval has counts on both sides but was not compared
    assert doc["caveats"][0] == "Q4 build vs reference, output tokens per reply (mean, median): GSM8K 800→1,000 (+25%, +25%)"
    assert doc["caveats"][1] == "Note: n0" and "Note: n11" not in doc["caveats"]


def test_compare_thinking_tokens_when_both_runs_counted_them(home, tmp_path, capsys):
    cap = lambda toks, th=None: {"replies": 30, "length": 0, "reasoning": 30, "tokens": toks * 10,
                                 **({"thinking_tokens": th * 10} if th else {})}
    # gsm8k: both runs have thinking counts; ifeval: only the reference's server sent them
    ref = _scored_run(tmp_path, "ref", {"gsm8k": cap([1000, 1500, 2000], [700, 800, 1200]),
                                        "ifeval": cap([1000, 1500, 2000], [700, 800, 1200])})
    test = _scored_run(tmp_path, "test", {"gsm8k": cap([600, 1200, 1200], [400, 450, 650]),
                                          "ifeval": cap([600, 1200, 1200])})
    assert main(["compare", str(ref), str(test), "--out", str(tmp_path / "cmp.json")]) == 0
    lines = capsys.readouterr().out.splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("  gsm8k: "))
    assert lines[i + 1:i + 3] == ["    tokens/reply: mean 1,500 -> 1,000 (-33%), median 1,500 -> 1,200 (-20%)",
                                  "    thinking/reply: mean 900 -> 500 (-44%), median 800 -> 450 (-44%)"]
    i = next(i for i, l in enumerate(lines) if l.startswith("  ifeval: "))
    assert lines[i + 1] == "    tokens/reply: mean 1,500 -> 1,000 (-33%), median 1,500 -> 1,200 (-20%)"
    assert sum("thinking/reply" in l for l in lines) == 1
    doc = publishable(json.loads((tmp_path / "cmp.json").read_text()))
    assert ("test vs reference, output tokens per reply (mean, median): GSM8K 1,500→1,000 (-33%, -20%) · "
            "IFEval 1,500→1,000 (-33%, -20%)") in doc["caveats"]  # totals only
    assert ("Output tokens per reply as the server counted them, mean (median): GSM8K 1,500 (1,500), thinking 900 (800) · "
            "IFEval 1,500 (1,500), thinking 900 (800)") in doc["caveats"]  # the reference's own
    # the HTML table's data, next to the comparison: the reference's counts and the arm's
    s = lambda m, md, th=None: {"n": 30, "mean": m, "median": md, **({"thinking": th} if th else {})}
    want = {"ref": {"gsm8k": s(1500, 1500, {"mean": 900, "median": 800}), "ifeval": s(1500, 1500, {"mean": 900, "median": 800})},
            "arms": {"test": {"gsm8k": s(1000, 1200, {"mean": 500, "median": 450}), "ifeval": s(1000, 1200)}}}
    assert json.loads((tmp_path / "cmp.qb" / "tokens.json").read_text()) == want
    assert f"const T = {json.dumps(want, separators=(',', ':'))};" in (tmp_path / "cmp.html").read_text()
    # cmp.json as the reference is refused as a comparison; as the test run, its cmp.qb/ (only that tokens.json)
    # is not a run's per-item data
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        main(["compare", str(tmp_path / "cmp.json"), str(test), "--out", str(tmp_path / "again.json")])
    assert e.value.code == 2
    assert f"compare: {tmp_path / 'cmp.json'} is a comparison; pass the reference run's own results file" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="per-item data"):
        main(["compare", str(ref), str(tmp_path / "cmp.json"), "--out", str(tmp_path / "again.json")])
    # a run's results file whose sidecar has no ENV.json (not a run's data) is not compared either
    (tmp_path / "bare.json").write_text(ref.read_text())
    (tmp_path / "bare.qb").mkdir()
    (tmp_path / "bare.qb" / "tokens.json").write_text("{}")
    with pytest.raises(SystemExit, match="per-item data"):
        main(["compare", str(tmp_path / "bare.json"), str(test), "--out", str(tmp_path / "again.json")])


def test_compare_names_each_runs_thinking_when_they_differ(home, tmp_path):
    # a run left at the server's default against one with thinking on
    ref = _scored_run(tmp_path, "ref", {}, thinking="server default")
    test = _scored_run(tmp_path, "test", {}, thinking="on")
    assert main(["compare", str(ref), str(test), "--out", str(tmp_path / "cmp.json")]) == 0
    doc = json.loads((tmp_path / "cmp.json").read_text())
    assert doc["title"] == "ref, default thinking vs test, thinking on"
    assert doc["arms"]["test"]["label"] == "test, thinking on"
    assert doc["arms_note"].startswith("Arm test: test, thinking on scored against the reference ref, default thinking ")
    # a long model id gives way in the label, the thinking setting does not
    long = "org/" + "m" * 66
    _scored_run(tmp_path, "long", {}, thinking="on")
    lj = tmp_path / "long.json"
    lj.write_text(json.dumps({**json.loads(lj.read_text()), "model": long}))
    off = _scored_run(tmp_path, "off", {})
    assert main(["compare", str(off), str(lj), "--out", str(tmp_path / "l.json")]) == 0
    label = json.loads((tmp_path / "l.json").read_text())["arms"]["m" * 32]["label"]
    assert len(long) == 70 and len(label) == 60 and label == long[:47] + ", thinking on"


def test_compare_refuses_an_old_thinking_pair_as_reference(home, tmp_path, capsys):
    # a document `run --reference` wrote for thinking off vs on before 0.5: its sidecar is the thinking-on run's
    test = _scored_run(tmp_path, "test", {})
    old = tmp_path / "pair.json"
    old.write_text(json.dumps({**json.loads(test.read_text()), "think_dirs": ["on"]}))
    import shutil
    shutil.copytree(tmp_path / "test.qb", tmp_path / "pair.qb")  # a run's data, but not the reference's
    with pytest.raises(SystemExit) as e:
        main(["compare", str(old), str(test), "--out", str(tmp_path / "c.json")])
    assert e.value.code == 2
    assert f"compare: {old} is a comparison; pass the reference run's own results file" in capsys.readouterr().err
    assert not (tmp_path / "c.json").exists()


def test_compare_says_once_when_one_run_has_no_token_counts(home, tmp_path, capsys):
    counted = {"replies": 30, "length": 0, "reasoning": 0, "tokens": [100, 200, 600] * 10}
    old = {"replies": 30, "length": 0, "reasoning": 0}  # caps as QualityBench 0.4.0 kept them
    ref = _scored_run(tmp_path, "ref", {"gsm8k": counted, "ifeval": counted})
    test = _scored_run(tmp_path, "test", {"gsm8k": old, "ifeval": old})
    assert main(["compare", str(ref), str(test), "--out", str(tmp_path / "cmp.json")]) == 0
    o = capsys.readouterr()
    assert o.err.count("output tokens not compared") == 1  # once, not per task
    assert ("warning: output tokens not compared for gsm8k, ifeval: the test run has no counts for them (made with "
            "QualityBench 0.4.0 or earlier, the server sent no usage, or not every reply was counted)") in o.err
    assert "  gsm8k: " in o.out and "  ifeval: " in o.out and "tokens/reply" not in o.out
    doc = publishable(json.loads((tmp_path / "cmp.json").read_text()))
    assert not any("vs reference, output tokens" in c for c in doc["caveats"])
    # neither run counted: no tokens.json, so no cmp.qb/ beside the comparison
    old_ref = _scored_run(tmp_path, "oldref", {"gsm8k": old, "ifeval": old})
    assert main(["compare", str(old_ref), str(test), "--out", str(tmp_path / "old.json")]) == 0
    assert (tmp_path / "old.html").is_file() and not (tmp_path / "old.qb").exists()


def test_compare_skips_tasks_whose_replies_were_not_all_counted(home, tmp_path, capsys):
    full = lambda toks: {"replies": 30, "length": 0, "reasoning": 0, "tokens": toks * 10}
    ref = _scored_run(tmp_path, "ref", {"gsm8k": full([100, 200, 600]), "ifeval": full([100, 200, 600])})
    # ifeval: 29 counts for 30 items (a killed run's first attempt was never recorded), so none are published
    test = _scored_run(tmp_path, "test", {"gsm8k": full([50, 100, 300]), "ifeval": {**full([50, 100, 300]),
                                                                                    "tokens": [50, 100, 300] * 9 + [50, 100]}})
    assert main(["compare", str(ref), str(test), "--out", str(tmp_path / "cmp.json")]) == 0
    o = capsys.readouterr()
    assert o.err.count("output tokens not compared") == 1
    assert ("warning: output tokens not compared for ifeval: the test run has no counts for it (made with QualityBench "
            "0.4.0 or earlier, the server sent no usage, or not every reply was counted)") in o.err
    assert [l for l in o.out.splitlines() if "tokens/reply" in l] == [
        "    tokens/reply: mean 300 -> 150 (-50%), median 200 -> 100 (-50%)"]
    doc = publishable(json.loads((tmp_path / "cmp.json").read_text()))
    assert "test vs reference, output tokens per reply (mean, median): GSM8K 300→150 (-50%, -50%)" in doc["caveats"]
    own = json.loads(test.read_text())["caveats"]
    assert "Output tokens per reply as the server counted them, mean (median): GSM8K 150 (100)" in own


def test_run_has_no_reference_or_arm_flags(home, tmp_path, capsys):
    # comparing is `qualitybench compare`; run only writes its own results
    fake = Fake()
    try:
        for flag in ("--reference", "--arm", "--arm-label"):
            with pytest.raises(SystemExit) as e:
                run(fake, tmp_path / "r.json", flag, "x")
            assert e.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err
        assert fake.requests == [] and not (tmp_path / "r.qb").exists()
    finally:
        fake.close()


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
            # the fake counts a reply's words as its output tokens: "answer to: hi" is 3, its reasoning "thinking..." 1
            assert px.take_stats() == {"replies": 3, "length": 3, "reasoning": 3, "filtered": 0, "tokens": [3, 3, 3], "thinking": [1, 1, 1]}
            assert px.take_stats() == {"replies": 0, "length": 0, "reasoning": 0, "filtered": 0, "tokens": [], "thinking": []}
            fake.finish, fake.reasoning = "stop", False
            c.chat("fake", [{"role": "user", "content": "hi"}], max_tokens=8)  # no completion_tokens_details
            assert px.take_stats() == {"replies": 1, "length": 0, "reasoning": 0, "filtered": 0, "tokens": [3], "thinking": [None]}
            # thinking tokens as radiance sends them; junk ones count as unknown, never shift the two lists apart
            for det in (b'{"reasoning_tokens":36}', b'{"reasoning_tokens":true}', b'{"reasoning_tokens":-1}',
                        b'{"reasoning_tokens":2.0}', b'"n/a"'):
                px.tally(b'{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],'
                         b'"usage":{"completion_tokens":41,"completion_tokens_details":' + det + b'}}')
            px.tally(b'{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],'
                     b'"usage":{"completion_tokens_details":{"reasoning_tokens":36}}}')  # no total: neither is kept
            assert px.take_stats() == {"replies": 6, "length": 0, "reasoning": 0, "filtered": 0, "tokens": [41] * 5,
                                       "thinking": [36, None, None, None, None]}
            # streamed: usage arrives last, in a chunk with no choices; a reply without (valid) usage still counts
            px.tally(b'data: {"choices":[{"delta":{"content":"a b"},"finish_reason":null}]}\n\n'
                     b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":null}\n\n'
                     b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n')
            px.tally(b'{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}]}')
            px.tally(b'{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],"usage":{"completion_tokens":-1}}')
            assert px.take_stats() == {"replies": 3, "length": 0, "reasoning": 0, "filtered": 0, "tokens": [2], "thinking": [None]}
            for junk in (b'"n/a"', b'{"completion_tokens":true}', b'{"completion_tokens":2.0}'):
                px.tally(b'{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],"usage":' + junk + b'}')
            assert px.take_stats() == {"replies": 3, "length": 0, "reasoning": 0, "filtered": 0, "tokens": [], "thinking": []}
            # junk usage from the server must not spoil the reply the client already got
            fake.usage = "n/a"
            r = qhttp.Client(px.url, attempts=1).chat("fake", [{"role": "user", "content": "hi"}], max_tokens=8)
            assert r["choices"][0]["message"]["content"] == "answer to: hi" and r["usage"] == "n/a"
            assert px.take_stats() == {"replies": 1, "length": 0, "reasoning": 0, "filtered": 0, "tokens": [], "thinking": []}
    finally:
        fake.close()


def test_run_reports_output_tokens_per_task(home, tmp_path, monkeypatch, capsys):
    from qualitybench import items
    from qualitybench.tasks import harness

    def lmeval(prompts):  # stands in for lm-eval: chat requests through the proxy, then the per-item results
        def go(ctx, task):
            c = qhttp.Client(ctx.api)
            for p in prompts[task]:
                c.chat("fake", [{"role": "user", "content": p}], max_tokens=8)
            items.write(ctx.items_path(task), [{"id": f"{task}/{i}", "correct": i != 1}
                                               for i in range(len(prompts[task]))])  # one reply per item
        return go

    fake = Fake()
    try:
        # the fake counts a reply's words as its output tokens: "answer to: x x x" is 5
        monkeypatch.setattr(harness, "run_lmeval", lmeval({"gsm8k": ["x", "x x x", "x x x x x x x x x x"],
                                                           "ifeval": ["x x x"]}))
        assert run(fake, tmp_path / "ref.json", "--tasks", "gsm8k,ifeval", "--thinking", "off") == 0
        doc = publishable(json.loads((tmp_path / "ref.json").read_text()))
        own = "Output tokens per reply as the server counted them, mean (median): GSM8K 7 (5) · IFEval 5 (5)"
        assert own in doc["caveats"]
        log = (tmp_path / "ref.qb" / "run.log").read_text()
        assert "gsm8k: 3 replies, 7 output tokens per reply (mean), 5 (median)" in log
        assert "ifeval: 1 reply, 5 output tokens per reply (mean), 5 (median)" in log
        # a test run, then compare: the comparison says how the replies' lengths changed
        monkeypatch.setattr(harness, "run_lmeval", lmeval({"gsm8k": ["x", "x", "x x x"], "ifeval": ["x"]}))
        assert run(fake, tmp_path / "test.json", "--tasks", "gsm8k,ifeval", "--thinking", "off") == 0
        test_tk = {"gsm8k": {"n": 3, "mean": 4, "median": 3}, "ifeval": {"n": 1, "mean": 3, "median": 3}}
        assert json.loads((tmp_path / "test.qb" / "tokens.json").read_text()) == {"ref": test_tk}  # its own only
        capsys.readouterr()
        assert main(["compare", str(tmp_path / "ref.json"), str(tmp_path / "test.json"), "--out", str(tmp_path / "cmp.json")]) == 0
        cmp = publishable(json.loads((tmp_path / "cmp.json").read_text()))
        note = "fake vs reference, output tokens per reply (mean, median): GSM8K 7→4 (-43%, -40%) · IFEval 5→3 (-40%, -40%)"
        assert note in cmp["caveats"]
        assert cmp["caveats"].index(note) == cmp["caveats"].index(own) + 1  # right after the reference's own
        lines = capsys.readouterr().out.splitlines()
        i = next(i for i, l in enumerate(lines) if l.startswith("  gsm8k: ") and "pts" in l)
        assert lines[i + 1] == "    tokens/reply: mean 7 -> 4 (-43%), median 5 -> 3 (-40%)"
        assert not any("thinking/reply" in l for l in lines)  # this server sends no thinking counts
        # the HTML table's data stays out of the results file: next to each, for `qualitybench report` too
        embedded = lambda p: json.loads(p.read_text().split("const T = ", 1)[1].split(";\n", 1)[0])
        ref_tk = {"gsm8k": {"n": 3, "mean": 7, "median": 5}, "ifeval": {"n": 1, "mean": 5, "median": 5}}
        assert json.loads((tmp_path / "ref.qb" / "tokens.json").read_text()) == {"ref": ref_tk}
        assert embedded(tmp_path / "ref.html") == {"ref": ref_tk}
        arm = {"ref": ref_tk, "arms": {"fake": test_tk}}
        assert json.loads((tmp_path / "cmp.qb" / "tokens.json").read_text()) == arm
        assert embedded(tmp_path / "cmp.html") == arm
        # thinking on: the run writes its own results, and the server counts the reasoning
        fake.reasoning = True
        monkeypatch.setattr(harness, "run_lmeval", lmeval({"gsm8k": ["x", "x", "x x x"], "ifeval": ["x"]}))
        assert run(fake, tmp_path / "on.json", "--tasks", "gsm8k,ifeval", "--thinking", "on") == 0
        on = publishable(json.loads((tmp_path / "on.json").read_text()))
        assert on["model"] == "fake" and "think_dirs" not in on and "arms" not in on
        assert not any("on" in t or "paired" in t for t in on["tasks"])
        assert not (tmp_path / "on.qb" / "run.json").exists()
        assert on["caveats"][0] == ("This run had thinking ON (max_tokens at least 32768): these are its scores with "
                                    "thinking on. `qualitybench compare` against a thinking-off run shows what "
                                    "thinking changes.")
        assert ("Output tokens per reply as the server counted them, mean (median): GSM8K 4 (3), thinking 1 (1) · "
                "IFEval 3 (3), thinking 1 (1)") in on["caveats"]
        on_tk = {"gsm8k": {"n": 3, "mean": 4, "median": 3, "thinking": {"mean": 1, "median": 1}},
                 "ifeval": {"n": 1, "mean": 3, "median": 3, "thinking": {"mean": 1, "median": 1}}}
        assert json.loads((tmp_path / "on.qb" / "tokens.json").read_text()) == {"ref": on_tk}
        assert embedded(tmp_path / "on.html") == {"ref": on_tk}
        # thinking off vs on of the same model is a compare: paired per task, the arm named by its thinking
        capsys.readouterr()
        assert main(["compare", str(tmp_path / "ref.json"), str(tmp_path / "on.json"), "--out", str(tmp_path / "think.json")]) == 0
        o = capsys.readouterr()
        th = publishable(json.loads((tmp_path / "think.json").read_text()))
        a = th["arms"][th["gate_arm"]]
        assert set(a["tasks"]) == {"gsm8k", "ifeval"}
        assert all(set(q) == {"n", "delta", "ci", "ref_only", "test_only", "p"} for q in a["tasks"].values())
        assert a["label"] == "fake, thinking on"
        assert th["title"] == "fake, thinking off vs fake, thinking on"
        assert th["arms_note"].startswith("Arm fake: fake, thinking on scored against the reference fake, thinking off ")
        assert ("fake, thinking on vs reference, output tokens per reply (mean, median): GSM8K 7→4 (-43%, -40%) · "
                "IFEval 5→3 (-40%, -40%)") in th["caveats"]
        assert "  gsm8k: " in o.out and "different request params" not in o.err
        assert json.loads((tmp_path / "think.qb" / "tokens.json").read_text()) == {"ref": ref_tk, "arms": {"fake": on_tk}}
    finally:
        fake.close()


def test_proxy_answers_a_content_filtered_prompt_with_an_empty_reply(home):
    fake = Fake(refuse="forbidden")
    try:
        with Proxy(fake.url) as px:
            c = qhttp.Client(px.url)
            r = c.chat("fake", [{"role": "user", "content": "a forbidden question"}], max_tokens=8)
            assert r["choices"][0]["message"]["content"] == "" and r["choices"][0]["finish_reason"] == "content_filter"
            assert r["usage"]["completion_tokens"] == 0
            c.chat("fake", [{"role": "user", "content": "hi"}], max_tokens=8)
            with pytest.raises(qhttp.HTTPError):  # any other 400 goes through as it is
                c.chat("fake", [{"role": "user", "content": "x" * 100_001}], max_tokens=8)
            assert px.take_stats() == {"replies": 2, "length": 0, "reasoning": 0, "filtered": 1,
                                       "tokens": [0, 3], "thinking": [None, None]}
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


def test_request_params_reach_chat_requests_only(home, tmp_path, capsys):
    fake = Fake(reasoning=True)
    try:
        params = ["--request-param", "reasoning_effort=high", "--request-param", "chat_template_kwargs.budget=4096"]
        assert run(fake, tmp_path / "r.json", "--thinking", "on", *params) == 0
        chat = [r[2] for r in fake.requests if r[1].endswith("/chat/completions") and "chat_template_kwargs" in r[2]]
        assert chat and all(b["reasoning_effort"] == "high" for b in chat)
        assert all(b["chat_template_kwargs"] == {"budget": 4096, "enable_thinking": True} for b in chat)
        scored = [r[2] for r in fake.requests if r[1] == "/v1/completions"]
        assert scored and not any("reasoning_effort" in b for b in scored)  # prompt logprobs stay as they are
        env = json.loads((tmp_path / "r.qb" / "ENV.json").read_text())
        assert env["request_params"] == {"reasoning_effort": "high", "chat_template_kwargs": {"budget": 4096}}
        doc = json.loads((tmp_path / "r.json").read_text())
        assert "request_params" not in doc["env"]  # the schema has no room: it is a caveat instead
        assert any("reasoning_effort=high, chat_template_kwargs.budget=4096" in c for c in doc["caveats"])
        hist = [json.loads(l) for l in (home / "timings.jsonl").read_text().splitlines()]
        assert hist[-1]["request_params"] == env["request_params"]
        # resuming with other params (or none) is refused; the same params resume
        with pytest.raises(SystemExit):
            run(fake, tmp_path / "r.json", "--thinking", "on", "--request-param", "reasoning_effort=low")
        with pytest.raises(SystemExit):
            run(fake, tmp_path / "r.json", "--thinking", "on")
        assert run(fake, tmp_path / "r.json", "--thinking", "on", *params) == 0
        # compare warns when the two runs sent different params
        assert run(fake, tmp_path / "plain.json", "--thinking", "on") == 0
        capsys.readouterr()
        assert main(["compare", str(tmp_path / "plain.json"), str(tmp_path / "r.json"), "--out", str(tmp_path / "c.json")]) == 0
        err = capsys.readouterr().err
        assert "different request params (reference: none; test: reasoning_effort=high" in err
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
