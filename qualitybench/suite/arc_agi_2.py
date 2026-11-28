"""ARC-AGI-2, run natively: the 120 tasks of the public evaluation set (arcprize/ARC-AGI-2 at a pinned commit), one
item per test input, ARC Prize's own prompt and answer parser (arc-agi-benchmarking), one attempt, exact grid match.
The grids stay in the QualityBench home: neither the results nor the sidecar carry them."""
import json
import os
import re

from ..taskdef import Task

SHA = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
URL = f"https://codeload.github.com/arcprize/ARC-AGI-2/tar.gz/{SHA}"
N_TASKS = 120
TIMEOUT_S = 120
# json.loads recurses once per nesting level: in a worker thread a reply nested ~1500 deep crashes the process (musl)
# or raises RecursionError (glibc). Deeper than this is no grid; a grid is 2 deep.
MAX_DEPTH = 64

# arc-agi-benchmarking's prompt template, verbatim
PROMPT = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your response:"""


def inputs_file():
    from .. import envs
    return envs.home() / "inputs" / "arc_agi_2" / "evaluation.json"


def prompt(train: list[dict], test_input: list) -> str:
    """arc-agi-benchmarking's prompt_manager, byte for byte."""
    training_examples = ""
    for i, pair in enumerate(train):
        training_examples += f"--Example {i}-- \n\n INPUT: \n\n"
        training_examples += json.dumps(pair["input"]) + "\n\n"
        training_examples += "OUTPUT: \n\n"
        training_examples += json.dumps(pair["output"]) + "\n\n"
    return PROMPT.format(training_examples=training_examples, test_input=json.dumps(test_input))


def _too_deep(s: str) -> bool:
    """Whether the text's brackets outside JSON strings nest deeper than MAX_DEPTH."""
    depth, in_str, esc = 0, False, False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                return True
        elif c in "]}":
            depth -= 1
    return False


def _loads(s: str):
    """json.loads, or None for text nested too deep to parse safely."""
    return None if _too_deep(s) else json.loads(s)


def extract_from_boxed(s: str) -> list | None:
    """arc-agi-benchmarking's extract_from_boxed: the first \\boxed{...}, if it holds a list of lists."""
    m = re.search(r"\\boxed\{(.*?)\}", s, re.DOTALL)
    if not m:
        return None
    try:
        got = _loads(m.group(1).strip())
    except ValueError:
        return None
    return got if isinstance(got, list) and all(isinstance(row, list) for row in got) else None


def backscan_json_parser(s: str) -> list | None:
    """arc-agi-benchmarking's backscan_json_parser: the bracket group that ends at the text's last ']' or '}', if it
    is a non-empty list of lists."""
    end = max(s.rfind("]"), s.rfind("}"))
    if end < 0:
        return None
    close = s[end]
    open_ = "[" if close == "]" else "{"
    depth, start = 1, -1
    for i in range(end - 1, -1, -1):
        if s[i] == close:
            depth += 1
        elif s[i] == open_:
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        return None
    try:
        got = _loads(s[start:end + 1])
    except ValueError:
        return None
    return got if isinstance(got, list) and got and all(isinstance(row, list) for row in got) else None


def parse_grid(reply: str | None) -> list | None:
    """The reply's answer grid outside any <think> block: a \\boxed{} grid, else the last bracket group, else None."""
    reply = re.sub(r"(?s)<think>.*?</think>", "", reply or "")
    reply = reply.rsplit("</think>", 1)[-1]
    reply = reply.split("<think>", 1)[0]  # thinking that never closed, e.g. cut at the token cap
    got = extract_from_boxed(reply)
    return got if got is not None else backscan_json_parser(reply)


def _grid(g) -> bool:
    return (isinstance(g, list) and bool(g) and all(isinstance(r, list) and r for r in g)
            and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))


def _check(tid: str, task) -> None:
    from ..context import TaskFailed
    ok = isinstance(task, dict) and all(isinstance(task.get(s), list) and task[s] for s in ("train", "test"))
    if not ok or not all(isinstance(p, dict) and _grid(p.get("input")) and _grid(p.get("output"))
                         for p in task["train"] + task["test"]):
        raise TaskFailed(f"ARC-AGI-2 task {tid[:40]} has a bad grid or pair")


def _read(data: bytes) -> dict:
    """{task_id: task} from the tarball's data/evaluation/*.json, checked against data/evaluation.txt."""
    import io
    import tarfile
    import zlib
    from ..context import TaskFailed
    tasks, listed = {}, None
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            members = t.getmembers()
            tops = {m.name.split("/", 1)[0] for m in members}
            if len(tops) != 1:
                raise TaskFailed(f"the ARC-AGI-2 tarball has {len(tops)} top folders, not one top folder")
            for m in members:
                rest = m.name.split("/", 1)[1] if "/" in m.name else ""
                if not m.isfile():
                    continue
                if rest == "data/evaluation.txt":
                    listed = t.extractfile(m).read().decode().split()
                elif (k := re.fullmatch(r"data/evaluation/([^/]+)\.json", rest)):
                    try:
                        tasks[k.group(1)] = json.loads(t.extractfile(m).read())
                    except ValueError:
                        raise TaskFailed(f"ARC-AGI-2 task {k.group(1)[:40]} is not JSON")
    except (tarfile.TarError, EOFError, OSError, zlib.error) as e:
        raise TaskFailed(f"the ARC-AGI-2 download is not a readable tarball ({type(e).__name__})")
    except UnicodeDecodeError:
        raise TaskFailed("the ARC-AGI-2 data/evaluation.txt is not text")
    if len(tasks) != N_TASKS:
        raise TaskFailed(f"the ARC-AGI-2 tarball has {len(tasks)} evaluation tasks, not {N_TASKS}")
    if listed is None:
        raise TaskFailed("the ARC-AGI-2 tarball has no data/evaluation.txt")
    if sorted(listed) != sorted(tasks):
        raise TaskFailed("the ARC-AGI-2 evaluation tasks differ from the ids in data/evaluation.txt")
    for tid, task in tasks.items():
        _check(tid, task)
    n = sum(len(task["test"]) for task in tasks.values())
    if n != TASK.items:
        raise TaskFailed(f"the ARC-AGI-2 evaluation set has {n} test inputs, not {TASK.items}")
    return tasks


def prepare(ctx):
    import urllib.error
    import urllib.request
    from ..context import TaskFailed, log_to
    f = inputs_file()
    if f.is_file():
        return
    log_to(ctx, f"arc_agi_2: downloading arcprize/ARC-AGI-2 at {SHA[:12]} (one time)")
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT_S) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise TaskFailed(f"could not download ARC-AGI-2 (HTTP {e.code})")
    except (urllib.error.URLError, OSError) as e:
        why = getattr(e, "reason", None) or e
        raise TaskFailed(f"could not download ARC-AGI-2 ({type(e).__name__}: {str(why)[:80]})")
    tasks = _read(data)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")  # two runs sharing a home never write the same file
    try:
        tmp.write_text(json.dumps(tasks))
        tmp.replace(f)
    finally:
        tmp.unlink(missing_ok=True)


def run(ctx, client):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from .. import items
    from ..context import log_to
    from ..progress import emit
    from ..taskdef import quick_pick
    prepare(ctx)
    tasks = json.loads(inputs_file().read_text())
    ids = sorted(f"{t}/{k}" for t, task in tasks.items() for k in range(len(task["test"])))
    pinned = quick_pick(ids) if ctx.quick else ids
    out = ctx.path("arc_agi_2-replies.jsonl")
    have: dict[str, dict] = {}
    if out.is_file():
        for line in out.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:  # a line cut short by a kill
                continue
            have[d["id"]] = d
    todo = [i for i in pinned if i not in have]
    lock = threading.Lock()
    with lock:
        if out.is_file() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
            with open(out, "a") as f:
                f.write("\n")  # a line torn by a kill: the next reply starts a line of its own
    done = len(pinned) - len(todo)
    emit(TASK.id, done=done, total=len(pinned))

    def one(i):
        nonlocal done
        ctx.check_deadline()
        tid, k = i.rsplit("/", 1)
        task, test = tasks[tid], tasks[tid]["test"][int(k)]
        # The --thinking-max-tokens cap with thinking on or off: frontier APIs reason hidden under it.
        r = client.chat(ctx.model, [{"role": "user", "content": prompt(task["train"], test["input"])}],
                        max_tokens=ctx.think_min_tokens, temperature=0)
        ch = r["choices"][0]
        grid = parse_grid((ch.get("message") or {}).get("content"))  # an empty reply (content filter) is wrong
        d = {"id": i, "parsed": grid is not None, "correct": grid == test["output"], "finish": ch.get("finish_reason") or "",
             "completion_tokens": (r.get("usage") or {}).get("completion_tokens") or 0}
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(d) + "\n")
            have[i] = d
            done += 1
            emit(TASK.id, throttle=2, done=done, total=len(pinned))

    with ThreadPoolExecutor(max(1, ctx.concurrency)) as ex:
        futs = [ex.submit(one, i) for i in todo]
        try:
            for f in futs:
                f.result()
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    k, n = items.write(ctx.items_path(TASK.id), [{"id": f"arc_agi_2/{i}", "correct": have[i]["correct"]} for i in pinned])
    log_to(ctx, f"arc_agi_2: {k}/{n} correct")


# Opt-in until L80's suite.go carries it. Downloads its data in prepare (stdlib only), so `setup` installs nothing.
TASK = Task(
    id="arc_agi_2",
    label="ARC-AGI-2",
    items=167,
    order=78,
    default=False,
    cost=70,
    how="120 public evaluation tasks, 167 test grids, ARC Prize's prompt, 1 attempt, exact grid match",
    about="Abstract visual puzzles, easy for people and hard for models: infer the rule from a few example pairs of "
          "colored grids, then produce the test output grid exactly (ARC-AGI-2).",
    scored=True,
    resumable=True,
    overhead_s=10,
    needs=frozenset(),
    # A guess until a run of this model is in the history: 167 replies of up to --thinking-max-tokens tokens.
    ref_minutes=40,
    run=run,
    prepare=prepare,
)
