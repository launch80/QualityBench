"""ARC-AGI-2: ARC Prize's prompt and grid parser, a run with a stub client, resume, quick, the up-front download of
the pinned evaluation set (served from memory, no network), and a run through the CLI against the fake server."""
import dataclasses
import io
import json
import os
import tarfile
import threading
import urllib.error

import pytest

from qualitybench import items, suite
from qualitybench.cli import main
from qualitybench.context import Ctx, TaskFailed
from qualitybench.suite import arc_agi_2 as arc
from qualitybench.taskdef import quick_pick

from fakeserver import Fake
from test_run_e2e import home, publishable  # noqa: F401  (the home fixture)

TOP = f"ARC-AGI-2-{arc.SHA}"


def _tid(n):
    return f"{0xa0000000 + n:08x}"


def _task(n, tests):
    """A task whose test inputs carry its number and the test's index, so a stub can tell them apart."""
    train = [{"input": [[n % 10, 1]], "output": [[1, n % 10]]}, {"input": [[2]], "output": [[3]]}]
    test = [{"input": [[int(c) for c in f"{n:03d}"] + [k]], "output": [[(n + k) % 10, 1], [2, 3]]} for k in range(tests)]
    return {"train": train, "test": test}


def _tasks():
    # the real set's shape: 120 tasks, 75 with one test input, 43 with two, 2 with three: 167 in all
    return {_tid(n): _task(n, 1 if n < 75 else 2 if n < 118 else 3) for n in range(120)}


def _seed(tasks=None):
    f = arc.inputs_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(tasks or _tasks()))
    return f


ALL = sorted(f"{t}/{k}" for t, task in _tasks().items() for k in range(len(task["test"])))


def _want(i):
    n, k = int(i.split("/")[0], 16) - 0xa0000000, int(i.split("/")[1])
    return (n + k) % 4 == 0


def test_task_is_opt_in_scored_and_dearest():
    t = suite.BY_ID["arc_agi_2"]
    assert (t.label, t.items, t.default, t.scored, t.resumable, t.prepare, t.run) == (
        "ARC-AGI-2", 167, False, True, True, arc.prepare, arc.run)
    assert t.needs == frozenset() and t.overhead_s == 10 and t.order == 78
    ids = [x.id for x in suite.SUITE]
    assert ids.index("lcb") < ids.index("arc_agi_2") < ids.index("logprobs")
    assert t.cost > max(x.cost for x in suite.SUITE if x.scored and x.id != "arc_agi_2")
    assert len(ALL) == 167 and len(quick_pick(ALL)) == 17


def test_prompt_is_arc_prizes_byte_for_byte():
    train = [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}, {"input": [[0]], "output": [[5]]}]
    assert arc.prompt(train, [[7, 8]]) == (
        "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n"
        "\n"
        "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or "
        "transformation in the training examples that maps the input to the output, then apply that pattern to the "
        "test input to give a final output.\n"
        "\n"
        "Respond in the format of the training output examples\n"
        "\n"
        "--Training Examples--\n"
        "--Example 0-- \n\n INPUT: \n\n[[1, 2], [3, 4]]\n\nOUTPUT: \n\n[[4, 3], [2, 1]]\n\n"
        "--Example 1-- \n\n INPUT: \n\n[[0]]\n\nOUTPUT: \n\n[[5]]\n\n"
        "\n"
        "--End of Training Examples--\n"
        "\n"
        "--Test Input--\n"
        "[[7, 8]]\n"
        "--End of Test Input--\n"
        "\n"
        "Your response:")


@pytest.mark.parametrize("reply,want", [
    ("\\boxed{[[1,2],[3,4]]}", [[1, 2], [3, 4]]),
    ("The answer is \\boxed{\n[[1, 2]]\n} done", [[1, 2]]),
    ("\\boxed{[[1]]} and later [[2]]", [[1]]),                    # boxed wins over a later grid
    ("\\boxed{[[1]]} or rather \\boxed{[[2]]}", [[1]]),            # the first boxed one, matched lazily
    ("\\boxed{[]}", []),                                          # their boxed check takes an empty list
    ("\\boxed{x} but [[1]]", [[1]]),                              # a boxed non-grid falls back to the backscan
    ("\\boxed{[1, 2]} but [[3]]", [[3]]),
    ("first [[1,2]] then [[3,4]]", [[3, 4]]),                     # the last grid
    ("Output:\n[[1, 0], [0, 1]]\nThat is the pattern.", [[1, 0], [0, 1]]),
    ("```json\n[[1, 2],\n [3, 4]]\n```", [[1, 2], [3, 4]]),
    ("<think>[[9]]</think>Answer: [[1]]", [[1]]),
    ("<think>[[9]]</think>no grid here", None),                   # a grid only inside the thinking is not one
    ("draft [[9]]</think>\n[[2]]", [[2]]),                        # thinking with no opening tag
    ("draft [[9]]</think>\nno grid after it", None),
    ("[[1]]\n<think>[[9]]</think> and that is all", [[1]]),       # outside a closed block, not inside it
    ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
    ("[[1]]\n<think>[[9]]", [[1]]),
    ("<think>so it is [[9]]", None),                              # thinking cut off before it closed
    ('{"a": 1}', None),
    ('[[1]] and {"x": 2}', None),                                 # the last bracket group is not a grid
    ("[[1]] then [1, 2]", None),
    ("[[1]] then []", None),
    ("[[1]] see [note]", None),
    ("[[1, 2]", None),
    ("[1, [2]]", None),
    ("no brackets at all", None),
    ("]", None),
    ("", None),
    (None, None),
])
def test_parse_grid(reply, want):
    assert arc.parse_grid(reply) == want


# A reply nested this deep crashes json.loads in a worker thread (a segfault on musl, a RecursionError on glibc).
# Each check runs in a child process, so a regression fails the test instead of killing the test run.
DEEP = ["\\boxed{" + "[" * 1500 + "}", "[" * 30000 + "]" * 30000]


def _child(code, tmp_path):
    import subprocess
    import sys
    env = dict(os.environ, QUALITYBENCH_HOME=str(tmp_path / "qbhome"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    return r.stdout


def test_a_deeply_nested_reply_is_no_grid_in_a_thread(tmp_path):
    out = _child(f"""
import threading
from qualitybench.suite import arc_agi_2 as arc
got = []
for reply in {DEEP!r}:
    t = threading.Thread(target=lambda: got.append(arc.parse_grid(reply)))
    t.start()
    t.join()
print(got)
""", tmp_path)
    assert out.strip() == "[None, None]"


def test_depth_guard_counts_only_brackets_outside_strings():
    deep_text = [["[" * 100 + "]" * 100]]  # a list of lists whose one string is full of brackets
    assert arc.parse_grid("[[1]] " + json.dumps(deep_text)) == deep_text
    assert arc.parse_grid("\\boxed{" + json.dumps(deep_text) + "}") == deep_text
    assert arc.parse_grid("[" * 64 + "]" * 64) is not None and arc.parse_grid("[" * 65 + "]" * 65) is None


def test_a_run_with_a_deeply_nested_reply_scores_it_wrong_and_finishes(tmp_path):
    out = _child(f"""
import json
from qualitybench import items
from qualitybench.context import Ctx
from qualitybench.suite import arc_agi_2 as arc
tasks = {{"00000000": {{"train": [{{"input": [[1]], "output": [[2]]}}],
                       "test": [{{"input": [[1]], "output": [[2]]}}, {{"input": [[3]], "output": [[2]]}}]}}}}
f = arc.inputs_file()
f.parent.mkdir(parents=True)
f.write_text(json.dumps(tasks))


class Deep:
    def chat(self, model, messages, **k):
        deep = "[[3]]" in messages[0]["content"].split("--Test Input--")[1]
        return {{"choices": [{{"message": {{"content": {DEEP[1]!r} if deep else "[[2]]"}}, "finish_reason": "stop"}}]}}


ctx = Ctx(model="m", endpoint="x", side=__import__("pathlib").Path({str(tmp_path / "r.qb")!r}), concurrency=2)
ctx.load_state()
arc.run(ctx, Deep())
print(json.dumps(items.load(ctx.items_path("arc_agi_2"))))
print(sorted((d["id"], d["parsed"]) for d in map(json.loads, ctx.path("arc_agi_2-replies.jsonl").open())))
""", tmp_path)
    scored, reps = out.strip().splitlines()[-2:]  # after the qb:progress lines
    assert json.loads(scored) == {"arc_agi_2/00000000/0": True, "arc_agi_2/00000000/1": False}
    assert reps == "[('00000000/0', True), ('00000000/1', False)]"


class Stub:
    """Replies by the test input's number n and index k: (n+k)%4 == 0 the right grid, 1 one of the same shape with
    one cell wrong, 2 prose without a grid, 3 an empty reply."""

    def __init__(self, max_tokens=32768, fail_on=()):
        self.asked, self.fail_on, self.lock, self.max_tokens = [], set(fail_on), threading.Lock(), max_tokens

    def chat(self, model, messages, max_tokens=None, temperature=None, **kw):
        assert model == "m" and max_tokens == self.max_tokens and temperature == 0 and not kw and len(messages) == 1
        assert messages[0]["role"] == "user"
        text = messages[0]["content"]
        grid = json.loads(text.split("--Test Input--\n")[1].split("\n--End of Test Input--")[0])
        n, k = int("".join(map(str, grid[0][:3]))), grid[0][3]
        i = f"{_tid(n)}/{k}"
        assert text == arc.prompt(_tasks()[_tid(n)]["train"], grid)
        with self.lock:
            self.asked.append(i)
        if i in self.fail_on:
            raise OSError("connection reset")
        right = [[(n + k) % 10, 1], [2, 3]]
        reply = [f"<think>maybe [[0]]</think>\\boxed{{{json.dumps(right)}}}", f"Here: {json.dumps([[(n + k) % 10, 1], [2, 4]])}",
                 "I cannot tell.", ""][(n + k) % 4]
        return {"choices": [{"message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 10 + n}}


@pytest.fixture
def arc_home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUALITYBENCH_HOME", str(tmp_path / "qbhome"))
    _seed()

    def boom(*a, **k):
        raise AssertionError("downloaded")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    ctx = Ctx(model="m", endpoint="x", side=tmp_path / "r.qb", concurrency=4)
    ctx.load_state()
    return ctx


def test_run_scores_and_keeps_no_grids(arc_home):
    ctx, client = arc_home, Stub()
    arc.run(ctx, client)
    assert sorted(client.asked) == ALL
    got = items.load(ctx.items_path("arc_agi_2"))
    assert list(got) == [f"arc_agi_2/{i}" for i in ALL]
    assert got == {f"arc_agi_2/{i}": _want(i) for i in ALL}
    reps = {d["id"]: d for d in map(json.loads, ctx.side.joinpath("arc_agi_2-replies.jsonl").open())}
    assert set(reps) == set(ALL)
    assert all(set(d) == {"id", "parsed", "correct", "finish", "completion_tokens"} for d in reps.values())
    a0 = _tid(0)
    assert reps[f"{a0}/0"] == {"id": f"{a0}/0", "parsed": True, "correct": True, "finish": "stop", "completion_tokens": 10}
    assert reps[f"{_tid(1)}/0"] == {"id": f"{_tid(1)}/0", "parsed": True, "correct": False, "finish": "stop", "completion_tokens": 11}
    assert reps[f"{_tid(2)}/0"]["parsed"] is False and reps[f"{_tid(2)}/0"]["correct"] is False
    assert reps[f"{_tid(3)}/0"]["parsed"] is False and reps[f"{_tid(3)}/0"]["correct"] is False  # an empty reply is wrong
    for p in (ctx.side / "arc_agi_2-replies.jsonl", ctx.items_path("arc_agi_2")):
        assert "[[" not in p.read_text(), p
    k = sum(map(_want, ALL))
    assert f"arc_agi_2: {k}/167 correct" in (ctx.side / "run.log").read_text()


@pytest.mark.parametrize("thinking", [None, "off", "on"])
def test_reply_cap_is_thinking_max_tokens_either_way(arc_home, thinking):
    ctx = arc_home
    ctx.thinking, ctx.think_min_tokens, ctx.quick = thinking, 12345, True
    client = Stub(max_tokens=12345)
    arc.run(ctx, client)
    assert len(client.asked) == 17


def test_rerun_asks_only_for_the_missing_items(arc_home):
    ctx = arc_home
    with pytest.raises(OSError):
        arc.run(ctx, Stub(fail_on={ALL[150]}))
    assert not ctx.items_path("arc_agi_2").is_file()
    have = {json.loads(line)["id"] for line in ctx.side.joinpath("arc_agi_2-replies.jsonl").open()}
    assert ALL[150] not in have and len(have) >= 1
    again = Stub()
    arc.run(ctx, again)
    assert sorted(again.asked) == sorted(set(ALL) - have)
    assert items.load(ctx.items_path("arc_agi_2")) == {f"arc_agi_2/{i}": _want(i) for i in ALL}
    third = Stub()
    arc.run(ctx, third)
    assert third.asked == []


def test_rerun_after_a_torn_last_line(arc_home):
    ctx = arc_home
    arc.run(ctx, Stub())
    f = ctx.side / "arc_agi_2-replies.jsonl"
    lines = f.read_text().splitlines()
    torn = json.loads(lines[-1])["id"]
    f.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])  # killed mid-write
    again = Stub()
    arc.run(ctx, again)
    assert again.asked == [torn]
    last = Stub()
    arc.run(ctx, last)
    assert last.asked == []
    assert items.load(ctx.items_path("arc_agi_2")) == {f"arc_agi_2/{i}": _want(i) for i in ALL}


def test_deadline_stops_the_run_and_keeps_the_answers(arc_home):
    import time
    from qualitybench.context import DeadlineHit
    ctx = arc_home
    ctx.concurrency = 1

    class Late(Stub):
        def chat(self, *a, **k):
            r = super().chat(*a, **k)
            if len(self.asked) == 10:
                ctx.deadline = time.time() - 1
            return r
    client = Late()
    with pytest.raises(DeadlineHit):
        arc.run(ctx, client)
    assert len(client.asked) == 10 and not ctx.items_path("arc_agi_2").is_file()
    kept = [json.loads(line)["id"] for line in (ctx.side / "arc_agi_2-replies.jsonl").open()]
    assert sorted(kept) == sorted(client.asked)


def test_quick_asks_the_fixed_seventeen(arc_home):
    ctx, client = arc_home, Stub()
    ctx.quick = True
    arc.run(ctx, client)
    want = [i.removeprefix("arc_agi_2/") for i in quick_pick([f"arc_agi_2/{i}" for i in ALL])]
    assert len(want) == 17 and sorted(client.asked) == want
    assert list(items.load(ctx.items_path("arc_agi_2"))) == [f"arc_agi_2/{i}" for i in want]


# ------------------------------------------------------------------ prepare: the pinned tarball, read in memory

def _tar(tasks, ids=None, top=TOP, extra=(), listing=True):
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers={"comment": arc.SHA}) as t:
        def add(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        for d in (f"{top}/", f"{top}/data/", f"{top}/data/evaluation/"):
            ti = tarfile.TarInfo(d)
            ti.type = tarfile.DIRTYPE
            t.addfile(ti)
        add(f"{top}/README.md", b"# ARC-AGI-2\n")
        add(f"{top}/data/training/ffffffff.json", json.dumps(_task(999, 1)).encode())  # not the evaluation set
        add(f"{top}/data/training.txt", b"ffffffff\n")
        for tid, task in tasks.items():
            add(f"{top}/data/evaluation/{tid}.json", json.dumps(task).encode())
        if listing:
            add(f"{top}/data/evaluation.txt", "".join(f"{i}\n" for i in (sorted(tasks) if ids is None else ids)).encode())
        for name, data in extra:
            add(name, data)
    return b.getvalue()


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def download(tmp_path, monkeypatch):
    """prepare over a small set (3 tasks, 5 test inputs) served from memory by a fake urlopen."""
    monkeypatch.setenv("QUALITYBENCH_HOME", str(tmp_path / "qbhome"))
    monkeypatch.setattr(arc, "N_TASKS", 3)
    monkeypatch.setattr(arc, "TASK", dataclasses.replace(arc.TASK, items=5))
    ctx = Ctx(model="m", endpoint="x", side=tmp_path / "r.qb")
    ctx.load_state()
    calls = []

    def serve(body):
        def urlopen(url, timeout=None, **k):
            calls.append((url if isinstance(url, str) else url.full_url, timeout))
            if isinstance(body, BaseException):
                raise body
            return Resp(body)
        monkeypatch.setattr("urllib.request.urlopen", urlopen)
    serve.ctx, serve.calls = ctx, calls
    return serve


SMALL = {_tid(0): _task(0, 1), _tid(1): _task(1, 2), _tid(2): _task(2, 2)}


def test_prepare_downloads_once_without_extracting(download, tmp_path):
    download(_tar(SMALL))
    arc.prepare(download.ctx)
    assert download.calls == [(arc.URL, download.calls[0][1])] and download.calls[0][1]
    assert arc.URL == f"https://codeload.github.com/arcprize/ARC-AGI-2/tar.gz/{arc.SHA}"
    assert arc.SHA == "f3283f727488ad98fe575ea6a5ac981e4a188e49"
    f = arc.inputs_file()
    assert json.loads(f.read_text()) == SMALL
    assert sorted(p.name for p in f.parent.iterdir()) == ["evaluation.json"]  # nothing extracted, no temp left
    assert not list(tmp_path.rglob("ARC-AGI-2-*"))
    download(OSError("must not download again"))
    arc.prepare(download.ctx)
    assert len(download.calls) == 1


def test_prepare_temp_file_is_this_processes(download, monkeypatch):
    from pathlib import Path
    seen, real = [], Path.replace

    def replace(self, target):
        seen.append(self.name)
        return real(self, target)
    monkeypatch.setattr(Path, "replace", replace)
    download(_tar(SMALL))
    arc.prepare(download.ctx)
    assert seen == [f"evaluation.json.{os.getpid()}.tmp"]


def test_prepare_removes_its_temp_file_when_the_rename_fails(download, monkeypatch):
    from pathlib import Path

    def replace(self, target):
        assert self.is_file()
        raise OSError("disk full")
    monkeypatch.setattr(Path, "replace", replace)
    download(_tar(SMALL))
    with pytest.raises(OSError, match="disk full"):
        arc.prepare(download.ctx)
    assert not arc.inputs_file().exists() and not list(arc.inputs_file().parent.iterdir())


def test_prepare_with_a_cached_file_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("QUALITYBENCH_HOME", str(tmp_path / "qbhome"))
    _seed()

    def boom(*a, **k):
        raise AssertionError("downloaded")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    arc.prepare(Ctx(model="m", endpoint="x", side=tmp_path / "r.qb"))


def _bad(task, **change):
    return dict(SMALL, **{_tid(1): dict(task, **change)})


@pytest.mark.parametrize("body,match", [
    (urllib.error.URLError("no route to host"), "could not download ARC-AGI-2.*no route to host"),
    (urllib.error.HTTPError(arc.URL, 404, "Not Found", {}, None), "could not download ARC-AGI-2.*404"),
    (TimeoutError("timed out"), "could not download ARC-AGI-2.*timed out"),
    (b"<html>not a tarball</html>", "not a readable tarball"),
    (_tar(dict(list(SMALL.items())[:2])), "2 evaluation tasks, not 3"),
    (_tar(SMALL, ids=[_tid(0), _tid(1), _tid(7)]), "evaluation.txt"),
    (_tar(SMALL, ids=[_tid(0), _tid(1), _tid(1), _tid(2)]), "evaluation.txt"),
    (_tar(SMALL, listing=False), "has no data/evaluation.txt"),
    (_tar(SMALL, extra=[("elsewhere/x.json", b"{}")]), "one top folder"),
    (_tar({**SMALL, _tid(2): _task(2, 1)}), "4 test inputs, not 5"),
    (_tar(_bad(SMALL[_tid(1)], test=[{"input": [[1, "a"]], "output": [[1]]}, {"input": [[1]], "output": [[1]]}])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], train=[{"input": [1, 2], "output": [[1]]}])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], train=[{"input": [[]], "output": [[1]]}])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], train=[{"input": [], "output": [[1]]}])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], train=[{"input": [[10]], "output": [[1]]}])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], train=[{"input": [[1.0]], "output": [[1]]}])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], train=[{"input": [[True]], "output": [[1]]}])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], train=[])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], test=[])), "bad grid"),
    (_tar(_bad(SMALL[_tid(1)], test=[{"input": [[1]]}, {"input": [[1]], "output": [[1]]}])), "bad grid"),
    (_tar(SMALL, extra=[(f"{TOP}/data/evaluation/{_tid(5)}.json", b"{not json")]), "not JSON"),
])
def test_prepare_failures_say_why(download, body, match):
    download(body)
    with pytest.raises(TaskFailed, match=match) as e:
        arc.prepare(download.ctx)
    assert len(str(e.value)) <= 150
    assert not arc.inputs_file().exists()
    assert not arc.inputs_file().parent.is_dir() or not list(arc.inputs_file().parent.iterdir())


def test_precheck_skips_arc_without_the_data(download):
    from qualitybench.runner import precheck_inputs
    download(urllib.error.URLError("no route to host"))
    ctx = download.ctx
    assert precheck_inputs(ctx, ["gsm8k", "arc_agi_2"]) == ["gsm8k"]
    why = ctx.state["partials"]["arc_agi_2"]
    assert why.startswith("arc_agi_2: skipped before running (could not download ARC-AGI-2") and why.endswith(")")


# ------------------------------------------------------------------ a run through the CLI against the fake server

def test_run_arc_against_fake_server(home, tmp_path, capsys):  # noqa: F811
    _seed()
    fake = Fake()  # its replies hold no grid: every item is wrong
    try:
        out = tmp_path / "r.json"
        assert main(["run", "--endpoint", fake.url, "--model", "fake", "--out", str(out), "--tasks", "arc_agi_2",
                     "--skip-setup", "--concurrency", "4"]) == 0
        o = capsys.readouterr().out
        assert 'qb:progress {"task":"arc_agi_2","state":"scored","correct":0,"total":167' in o
        doc = json.loads(out.read_text())
        row = next(r for r in doc["tasks"] if r["id"] == "arc_agi_2")
        assert row["label"] == "ARC-AGI-2" and row["off"]["n"] == 167
        assert "items:arc_agi_2" in doc["hashes"]
        publishable(doc)
        asked = [r[2] for r in fake.requests if r[1].endswith("/chat/completions") and "--Test Input--" in r[2]["messages"][-1]["content"]]
        assert len(asked) == 167 and {b["max_tokens"] for b in asked} == {32768}
        assert main(["run", "--endpoint", fake.url, "--model", "fake", "--out", str(out), "--tasks", "arc_agi_2",
                     "--skip-setup", "--concurrency", "4"]) == 0
        assert sum(1 for r in fake.requests if r[1].endswith("/chat/completions") and "--Test Input--" in r[2]["messages"][-1]["content"]) == 167
    finally:
        fake.close()
