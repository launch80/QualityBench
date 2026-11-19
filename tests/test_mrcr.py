"""The MRCR 8-needle task: the pinned items, the grader, the results block, a fake-server run, and compare."""
import json

import pytest

from qualitybench import envs, results, stats
from qualitybench.cli import main
from qualitybench.tasks import native

from fakeserver import Fake
from test_run_e2e import home, publishable  # noqa: F401  (the home fixture)


def test_pinned_items_are_the_two_bins_whole():
    doc = json.loads(envs.pkg_file("data", "mrcr-ids.json").read_text())
    its = doc["items"]
    assert len(its) == 200 == doc["count"] and doc["n_needles"] == 8 and doc["dataset"] == "openai/mrcr"
    lengths = [i["length"] for i in its]
    assert lengths.count(65536) == 100 and lengths.count(131072) == 100
    ids = [i["id"] for i in its]
    assert len(set(ids)) == 200 and all(len(i) == 16 for i in ids)
    assert its == sorted(its, key=lambda i: (i["length"], i["id"]))
    for i in its:
        lo, hi = (32768, 65536) if i["length"] == 65536 else (65536, 131072)
        assert lo < i["tokens"] <= hi
        assert i["file"] in ("8needle_0.parquet", "8needle_1.parquet") and 0 <= i["row"] < 400
        assert len(i["answer_sha256_16"]) == 16
    assert doc["max_answer_tokens"] * 2 <= native.MRCR_MAX_TOKENS  # the reply cap leaves twice the longest answer
    full, quick = native.mrcr_pinned(False), native.mrcr_pinned(True)
    assert [i["id"] for i in full] == ids
    assert len(quick) == 20 and [i["length"] for i in quick].count(65536) == 10 and quick == full[::10]


def test_grade_is_the_dataset_cards():
    assert native.grade_mrcr("abc123 the poem", "abc123 the poem", "abc123") == 1.0
    assert native.grade_mrcr("the poem", "abc123 the poem", "abc123") == 0.0  # no prefix: 0, however close the text
    assert native.grade_mrcr(" abc123 the poem", "abc123 the poem", "abc123") == 0.0  # strict, as the card grades it
    part = native.grade_mrcr("abc123 the poem about rain", "abc123 the poem", "abc123")
    assert 0.5 < part < 1.0
    assert native.grade_mrcr("abc123", "abc123 the poem", "abc123") == 0.0


def test_paired_values():
    R = {str(i): i / 10 for i in range(10)}
    assert stats.paired_values(R, R) == {"n": 10, "ref": 45.0, "test": 45.0, "delta": 0.0, "ci": [0.0, 0.0]}
    T = {k: max(0.0, v - 0.1) for k, v in R.items()}
    d = stats.paired_values(R, T)
    assert d["delta"] == pytest.approx(-9.0) and d["ci"][0] <= d["delta"] <= d["ci"][1] and d["ci"][1] < 0
    with pytest.raises(ValueError):
        stats.paired_values(R, {"x": 1.0})


def test_summary_and_results_block(tmp_path):
    its = [{"id": f"i{k}", "length": 65536 if k < 4 else 131072, "ratio": r, "prefix_ok": r > 0, "prompt_tokens": 50000 + k,
            "completion_tokens": 300, "finish": "length" if k == 5 else "stop"}
           for k, r in enumerate([1.0, 0.8, 0.0, 1.0, 0.5, 0.2])]
    s = native.mrcr_summary(its)
    assert s == {"65536": {"n": 4, "mean": 0.7, "p50": 0.9, "prefix_fail": 1, "capped": 0, "prompt_tokens": 50002, "completion_tokens": 300},
                 "131072": {"n": 2, "mean": 0.35, "p50": 0.35, "prefix_fail": 0, "capped": 1, "prompt_tokens": 50005, "completion_tokens": 300}}
    its[1]["reasoning_tokens"] = 120  # a server that reports the thinking share: the mean over the items that have it
    assert native.mrcr_summary(its)["65536"]["reasoning_tokens"] == 120 and "reasoning_tokens" not in native.mrcr_summary(its)["131072"]
    del its[1]["reasoning_tokens"]
    side = tmp_path / "r.qb"
    (side / "items").mkdir(parents=True)
    (side / "ENV.json").write_text(json.dumps({"model": "m", "thinking": "off", "date": "2026-11-18"}))
    (side / "mrcr.json").write_text(json.dumps({"model": "m", "n_needles": 8, "summary": s, "items": its}))
    (side / "items" / "gsm8k.jsonl").write_text("".join(json.dumps({"id": f"gsm8k/{i}", "correct": i % 2 == 0}) + "\n" for i in range(20)))
    doc = results.build(side)
    assert doc["mrcr"]["summary"] == s and doc["mrcr"]["n_needles"] == 8
    assert doc["mrcr"]["items"][0] == {"id": "i0", "length": 65536, "ratio": 1.0, "prompt_tokens": 50000, "completion_tokens": 300}  # no per-item text
    assert "items:mrcr" in doc["hashes"]
    assert any("MRCR 8-needle: 1 of 6 replies (17%) stopped at the token cap" in c for c in doc["caveats"])
    publishable(doc)


# ------------------------------------------------------------------ a run against the fake server

def _item(id, length, tokens, k, prefix, answer_k=None, pad=0):
    poems = ["Rain falls softly on the roof tonight", "Sun burns bright above the hill"]
    msgs = []
    for p in poems:
        msgs += [{"role": "user", "content": "write a poem about the weather" + (" " + "x" * pad if pad else "")},
                 {"role": "assistant", "content": p}]
    msgs.append({"role": "user", "content": f"Prepend {prefix} to the {k}{'st' if k == 1 else 'nd'} (1 indexed) poem about "
                                            "the weather. Do not include any other text in your response."})
    return {"id": id, "length": length, "tokens": tokens, "messages": msgs, "answer": f"{prefix} {poems[(answer_k or k) - 1]}",
            "prefix": prefix}


@pytest.fixture
def mrcr_home(home, monkeypatch):  # noqa: F811
    items = [
        _item("aaaa000000000001", 65536, 1000, 1, "AbCdEfGh12"),                   # asks the 1st poem: exact
        _item("bbbb000000000002", 65536, 1200, 2, "AbCdEfGh13", answer_k=1),       # asks the 2nd, expects the 1st: partial
        _item("cccc000000000003", 131072, 125000, 1, "AbCdEfGh14"),                # over the fake's 131072 context: never sent
        _item("dddd000000000004", 131072, 100000, 1, "AbCdEfGh15", pad=120_000),   # fits by the count, the server says 400
        _item("eeee000000000005", 131072, 90000, 1, "AbCdEfGh16"),
    ]
    f = envs.mrcr_inputs()
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"meta": {"n_needles": 8}, "items": items}))
    pins = sorted(({"id": i["id"], "length": i["length"], "tokens": i["tokens"]} for i in items), key=lambda p: (p["length"], p["id"]))
    monkeypatch.setattr(native, "mrcr_pinned", lambda quick: pins)
    return home


def _args(url, out):
    return ["run", "--endpoint", url, "--model", "fake", "--out", str(out), "--tasks", "mrcr", "--skip-setup", "--concurrency", "4"]


def test_run_mrcr_against_fake_server(mrcr_home, tmp_path, capsys):
    fake = Fake()
    try:
        out = tmp_path / "r.json"
        assert main(_args(fake.url, out)) == 0  # items over the context are partials, not failures
        o = capsys.readouterr().out
        assert 'qb:progress {"task":"mrcr","done":0,"total":5' in o
        scored = json.loads(next(l for l in o.splitlines() if l.startswith('qb:progress {"task":"mrcr","state":"scored"'))[12:])
        assert scored["done"] == 3 and scored["total"] == 5 and 0 < scored["score"] < 100
        doc = json.loads(out.read_text())
        by = {i["id"]: i for i in doc["mrcr"]["items"]}
        assert set(by) == {"aaaa000000000001", "bbbb000000000002", "eeee000000000005"}
        assert by["aaaa000000000001"]["ratio"] == 1.0 and by["eeee000000000005"]["ratio"] == 1.0
        assert 0 < by["bbbb000000000002"]["ratio"] < 1
        assert doc["mrcr"]["summary"]["65536"]["n"] == 2
        s128 = doc["mrcr"]["summary"]["131072"]
        assert {k: s128[k] for k in ("n", "mean", "p50", "prefix_fail", "capped")} == {"n": 1, "mean": 1.0, "p50": 1.0, "prefix_fail": 0, "capped": 0}
        assert s128["prompt_tokens"] > 0 and s128["completion_tokens"] > 0 and "reasoning_tokens" not in s128  # the fake reports no thinking
        assert by["eeee000000000005"]["completion_tokens"] > 0 and "reasoning_tokens" not in by["eeee000000000005"]
        assert "Reply tokens" in (tmp_path / "r.html").read_text()
        assert doc["partials"] == ["mrcr: 2 of 3 at 128k skipped, over the server's context (131072 tokens)"]
        assert "items:mrcr" in doc["hashes"] and (tmp_path / "r.qb" / "mrcr-items.jsonl").is_file()
        assert "MRCR 8-needle" in (tmp_path / "r.html").read_text()
        asked = lambda: sum(1 for r in fake.requests if r[1].endswith("/chat/completions") and "Prepend" in r[2]["messages"][-1]["content"])
        assert asked() == 4  # cccc was never sent; dddd was sent once and refused
        n = asked()
        assert main(_args(fake.url, out)) == 0  # resume: scored items are not asked again; the refused one is tried once more
        assert asked() == n + 1
        assert json.loads(out.read_text())["mrcr"]["items"] == doc["mrcr"]["items"]
    finally:
        fake.close()


def test_compare_mrcr_arm(mrcr_home, tmp_path, capsys):
    ref, test = Fake(), Fake(mrcr_truncate=True)
    try:
        assert main(_args(ref.url, tmp_path / "ref.json")) == 0
        assert main(_args(test.url, tmp_path / "test.json")) == 0
        for name in ("ref", "test"):  # a scored task, so the document is publishable
            (tmp_path / f"{name}.qb" / "items").mkdir(exist_ok=True)
            (tmp_path / f"{name}.qb" / "items" / "gsm8k.jsonl").write_text(
                "".join(json.dumps({"id": f"gsm8k/{i}", "correct": i % 3 != 0}) + "\n" for i in range(30)))
        capsys.readouterr()
        assert main(["compare", str(tmp_path / "ref.json"), str(tmp_path / "test.json"), "--out", str(tmp_path / "cmp.json")]) == 0
        out, err = capsys.readouterr()
        assert "mrcr" not in err  # same items on both sides: nothing to warn about
        doc = json.loads((tmp_path / "cmp.json").read_text())
        m = doc["arms"]["fake"]["mrcr"]
        assert m["n"] == 3 and m["ref"] > m["arm"] and m["delta"] < 0 and m["ci"][0] <= m["delta"] <= m["ci"][1] <= 0
        assert set(m["by_length"]) == {"65536", "131072"} and m["by_length"]["131072"]["n"] == 1
        assert m["by_length"]["131072"]["ref"] == 100.0 and m["by_length"]["131072"]["arm"] < 100.0
        assert "mrcr: mean match" in out and "conversations)" in out
        assert "MRCR 8-needle match" in (tmp_path / "cmp.html").read_text()
        doc["tasks"] = results.build(tmp_path / "ref.qb")["tasks"]
        publishable(doc)
        # one side skipped an item: compare uses the shared ones and says so
        side = tmp_path / "test.qb" / "mrcr.json"
        md = json.loads(side.read_text())
        md["items"] = [i for i in md["items"] if i["id"] != "eeee000000000005"]
        side.write_text(json.dumps(md))
        assert main(["compare", str(tmp_path / "ref.json"), str(tmp_path / "test.json"), "--out", str(tmp_path / "cmp2.json")]) == 0
        err = capsys.readouterr().err
        assert "mrcr: compared on the 2 items both runs answered (reference 3, test 2" in err
        assert json.loads((tmp_path / "cmp2.json").read_text())["arms"]["fake"]["mrcr"]["n"] == 2
    finally:
        ref.close()
        test.close()
