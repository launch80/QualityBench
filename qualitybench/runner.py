"""`qualitybench run`: one endpoint, the pinned suite, a results file plus an HTML report."""
from __future__ import annotations

import json
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path

import sys

from . import __version__, compare, envs, items, report, results
from . import estimate as qest
from . import http as qhttp
from .context import Ctx, DeadlineHit, TaskFailed, log_to, parse_deadline, sidecar_for
from .progress import Tracker, emit, emit_line, track_run
from .proxy import Proxy
from .suite import BY_ID, SCORED, parse_tasks, run_order
from .tasks import harness, native


def preflight(ctx: Ctx, client: qhttp.Client) -> None:
    """The first request after start: a warmup (it pays any model load) and a check, never scored."""
    log_to(ctx, "warmup + preflight")
    r = client.chat(ctx.model, [{"role": "user", "content": "Reply with the single word OK."}],
                    max_tokens=32 if ctx.thinking != "on" else 4096, temperature=0, log=lambda m: log_to(ctx, m))
    m = dict(r["choices"][0]["message"])
    reasoning = m.get("reasoning") or m.get("reasoning_content")
    content = m.get("content") or ""
    if "<think>" in content or "</think>" in content:  # a server with no reasoning parser: thinking inline
        inline = re.sub(r"(?s)^.*</think>", "", content) if "</think>" in content else ""
        reasoning = reasoning or content[: len(content) - len(inline)] or content
        m["content"] = inline.strip()
    log_to(ctx, f"preflight content: {m.get('content')!r} reasoning: {'yes' if reasoning else 'no'}")
    if ctx.thinking is None and reasoning:
        raise TaskFailed("preflight: this model thinks by default, and the harnesses' token limits would cut its "
                         "replies off mid-thought. Pass --thinking off (the usual baseline) or --thinking on")
    if not m.get("content"):
        raise TaskFailed("preflight: the model returned no content")
    if ctx.thinking == "off" and reasoning:
        raise TaskFailed("preflight: thinking is still on (reasoning text returned with enable_thinking=false)")
    if ctx.thinking == "on" and not reasoning:
        raise TaskFailed("preflight: thinking is not on (no reasoning text). The server needs a reasoning parser "
                         "(vLLM --reasoning-parser) and a chat template that honours enable_thinking")


OUTPUTS = {"logprobs": "logprobs.npz", "needle": "needle.json", "spec_parity": "parity.json"}


def finished(ctx: Ctx, task: str) -> bool:
    if task in ctx.state["partials"]:
        return False
    if task in SCORED:
        return items_ok(ctx, task)
    return (ctx.side / OUTPUTS[task]).is_file()


def items_ok(ctx: Ctx, task: str) -> bool:
    p = ctx.items_path(task)
    return p.is_file() and p.stat().st_size > 0


def run_task(ctx: Ctx, task: str, client: qhttp.Client) -> None:
    if task in ("gsm8k", "ifeval", "mmlu_pro"):
        harness.run_lmeval(ctx, task)
    elif task in ("humaneval", "mbpp"):
        harness.run_evalplus(ctx, task)
    elif task == "bfcl":
        harness.run_bfcl(ctx)
    elif task == "lcb":
        harness.run_lcb(ctx)
    elif task == "logprobs":
        native.run_logprobs(ctx, client)
    elif task == "needle":
        native.run_needle(ctx, client)
    elif task == "spec_parity":
        native.run_parity(ctx, client)
    if task in SCORED:
        it = items.load(ctx.items_path(task))
        emit(task, state="scored", correct=sum(it.values()), total=len(it))


def write_env(ctx: Ctx, server_models: dict) -> None:
    p = ctx.side / "ENV.json"
    old = json.loads(p.read_text()) if p.is_file() else {}
    env = {"date": old.get("date") or datetime.now().astimezone().isoformat(), "model": ctx.model,
           **harness.harness_versions(), "sandbox": ctx.sandbox,
           "thinking": ctx.thinking or "server default",
           "thinking_on_min_max_tokens": ctx.think_min_tokens if ctx.thinking == "on" else 0,
           "evalplus_image": envs.EVALPLUS_IMAGE, "container": _runtime(ctx), "endpoint": ctx.endpoint, "host": socket.gethostname(),
           "qualitybench": __version__, "server_models": server_models}
    if old.get("date"):
        env["resumed"] = datetime.now().astimezone().isoformat()
    ctx.path("ENV.json").write_text(json.dumps(env, indent=1))


def _runtime(ctx: Ctx) -> str | None:
    if ctx.sandbox == "native":
        return None
    try:
        return envs.container()
    except SystemExit:
        return None


def run(a) -> int:
    out = Path(a.out).absolute()
    if not out.name.endswith(".json"):
        raise SystemExit("--out must name a .json results file")
    tasks = parse_tasks(a.tasks)
    deadline = parse_deadline(a.deadline)
    side = sidecar_for(out)
    side.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(model=a.model, endpoint=a.endpoint, side=side, quick=a.quick, thinking=a.thinking,
              think_min_tokens=a.thinking_max_tokens, concurrency=a.concurrency, sandbox=a.sandbox, deadline=deadline,
              tokenizer=a.tokenizer, eval_set=a.eval_set, api_key=os.environ.get("QUALITYBENCH_API_KEY"))
    ctx.load_state()
    prev = ctx.state.get("run")
    now = {"model": a.model, "quick": a.quick, "thinking": a.thinking or ""}
    if prev and prev != now:
        raise SystemExit(f"{side.name}/ holds a run with {prev}, not {now}. Rerun the same command to resume it, "
                         "or pick another --out")
    ctx.state["run"], ctx.state["quick"] = now, a.quick
    ctx.save_state()
    log_to(ctx, f"qualitybench {__version__}: {a.model} at {a.endpoint}, tasks {','.join(tasks)}"
                f"{' (quick)' if a.quick else ''}, results {out}")

    todo = [t for t in run_order(tasks, deadline is not None) if not finished(ctx, t)]
    for t in tasks:
        if t not in todo:
            log_to(ctx, f"{t}: finished earlier, skipping")
    failed: list[str] = []
    interrupted = False
    est: dict | None = None
    probe: dict | None = None
    tracker: Tracker | None = None
    stop_tracker = None
    walls: dict[str, dict] = {}
    t_run = time.time()
    with Proxy(a.endpoint, ctx.api_key, a.thinking, a.thinking_max_tokens if a.thinking == "on" else 0) as px:
        ctx.api = px.url
        client = qhttp.Client(px.url)
        try:
            if todo:
                models = qhttp.wait_ready(client, attempts=6)
                write_env(ctx, models)
                preflight(ctx, client)
                est, probe = _estimate(ctx, a, todo, client)
                px.take_stats()  # the warmup and the speed probe are not part of any task
                gate = _gate(a, est)
                if gate is not None:
                    return gate
                if not a.skip_setup:
                    envs.setup(todo, a.sandbox)
                tracker = Tracker({r["id"]: r["min"] for r in est["tasks"]}, todo)
                stop_tracker = track_run(tracker)
            elif not (side / "ENV.json").is_file():
                write_env(ctx, {})
            for i, t in enumerate(todo):
                if ctx.past_deadline():
                    for rest in todo[i:]:
                        ctx.mark_partial(rest, f"{rest}: not started, deadline reached")
                        emit(rest, state="partial", done=0, total=BY_ID[rest].items)
                    log_to(ctx, "deadline reached: starting no more tasks")
                    break
                log_to(ctx, f"{t}: start")
                t0 = time.time()
                if tracker:
                    tracker.start(t)
                    tracker.emit()
                ctx.clear_partial(t)  # a task may mark itself partial again (e.g. a needle length the server refuses)
                try:
                    run_task(ctx, t, client)
                    log_to(ctx, f"{t}: done in {(time.time() - t0) / 60:.1f} min")
                    walls[t] = {"items": qest.items_for(t, a.quick), "wall_s": round(time.time() - t0)}
                except DeadlineHit:
                    ctx.mark_partial(t, f"{t}: cut by the deadline after {(time.time() - t0) / 60:.0f} min; rerun to finish")
                    emit(t, state="partial", done=0, total=BY_ID[t].items)
                    log_to(ctx, f"{t}: cut by the deadline")
                except (TaskFailed, qhttp.HTTPError, OSError, ValueError) as e:
                    failed.append(t)
                    ctx.mark_partial(t, f"{t}: failed ({str(e)[:150]})")
                    emit(t, state="failed")
                    log_to(ctx, f"{t}: FAILED: {e}")
                finally:
                    _record_caps(ctx, t, px.take_stats())
                    if tracker:
                        tracker.end(t)
                        tracker.emit()
        except KeyboardInterrupt:
            interrupted = True
            qhttp.cancel_all()
            log_to(ctx, "interrupted: keeping what finished; rerun the same command to resume")
        except (TaskFailed, qhttp.HTTPError, RuntimeError) as e:
            log_to(ctx, f"FAILED: {e}")
            return 1
        finally:
            if stop_tracker:
                stop_tracker.set()
            track_run(None)

    if walls:
        _finish_timings(ctx, a, walls, probe, est, time.time() - t_run)

    if not (side / "ENV.json").is_file():
        return 1
    if a.reference:
        own = side / "run.json"
        doc = results.build(side, title=a.title, thresholds=a.thresholds, notes=a.note)
        _write(ctx, doc, own)
        ref = json.loads(Path(a.reference).read_text())
        ref_side = sidecar_for(Path(a.reference))
        if ref["model"] == a.model and a.thinking == "on" and not ref.get("env", {}).get("thinking_on_min_max_tokens"):
            # Same model, thinking on vs the reference's thinking off: pair them task by task.
            doc = results.build(ref_side, title=ref.get("title"), thresholds=a.thresholds, notes=a.note,
                                on_side=side, on_src=out.stem)
            doc["env"] = ref.get("env", doc["env"])
        else:
            doc, warn = compare.build(Path(a.reference), own, a.thresholds, a.arm, a.arm_label, ref_side=ref_side,
                                      test_side=side)
            for w in warn:
                log_to(ctx, "compare: " + w)
            for line in compare.summary_lines(doc, doc["gate_arm"]):
                log_to(ctx, line)
    else:
        doc = results.build(side, title=a.title, thresholds=a.thresholds, notes=a.note)
    _write(ctx, doc, out)
    for p in results.validate_shape(doc):
        log_to(ctx, f"note: launch80 will not accept this file yet: {p}")
    if interrupted:
        return 130
    return 1 if failed else 0


RESUMABLE = ("humaneval", "mbpp", "lcb")  # a rerun asks only for the missing items, so counts add up


def _record_caps(ctx: Ctx, task: str, st: dict) -> None:
    """Keep how many of a task's replies hit the token cap (state.json), and say so in the log."""
    if task not in SCORED or not st["replies"]:
        return
    caps = ctx.state.setdefault("caps", {})
    old = caps.get(task) if task in RESUMABLE else None
    if old:
        st = {k: st[k] + old.get(k, 0) for k in st}
    caps[task] = {**st, "thinking": ctx.thinking or "", "max_tokens": ctx.think_min_tokens if ctx.thinking == "on" else 0}
    ctx.save_state()
    if st["length"]:
        log_to(ctx, f"{task}: {st['length']} of {st['replies']} replies stopped at the token cap (scored as wrong)")
    if ctx.thinking is None and st["reasoning"]:
        log_to(ctx, f"{task}: WARNING: {st['reasoning']} of {st['replies']} replies had reasoning, but --thinking was "
                    "not set, so the harnesses' short token limits apply. Rerun with --thinking on "
                    "--thinking-max-tokens 32768, or --thinking off")


def _write(ctx: Ctx, doc: dict, out: Path) -> None:
    for w in results.write(doc, out):
        log_to(ctx, "results: " + w)
    html = report.write(doc, out)
    log_to(ctx, f"wrote {out} and {html.name}")


def _estimate(ctx: Ctx, a, todo: list[str], client: qhttp.Client) -> tuple[dict, dict | None]:
    """Measure the server (unless --no-probe) and estimate the run; print it as a qb:estimate line too."""
    probe, warn = None, None
    if not getattr(a, "no_probe", False):
        probe = qest.cached_probe(ctx.endpoint, ctx.model)
        if probe is None:
            log_to(ctx, "measuring the server for the time estimate")
            try:
                # Straight to the server, not through the proxy: with --thinking on the proxy raises every
                # max_tokens to the thinking floor, and with ignore_eos the 128-token probe ran for that long.
                direct = qhttp.Client(ctx.endpoint, ctx.api_key, attempts=1)
                probe = qest.probe(direct, ctx.model, concurrency=min(16, ctx.concurrency))
            except (OSError, qhttp.HTTPError, ValueError) as e:
                warn = f"could not measure the server ({str(e)[:160]}); the estimate uses the reference run"
            else:
                qest._save_probe(ctx.endpoint, ctx.model, probe)
    est = qest.estimate(todo, a.quick, a.thinking, ctx.model, probe, a.sandbox)
    if warn:
        est["warning"] = warn
    emit_line("estimate", est)
    for line in qest.summary(est):
        log_to(ctx, line)
    _env_update(ctx, estimate={"total_min": est["total_min"], "range_min": est["range_min"], "source": est["source"]})
    return est, probe


def _gate(a, est: dict) -> int | None:
    """None to go on; an exit status to stop with."""
    hi = est["range_min"][1]
    if a.max_minutes is not None and hi > a.max_minutes:
        print(f"qualitybench: not started: this run could take up to {qest.fmt_min(hi)}, over --max-minutes "
              f"{a.max_minutes:g}. Narrow it (--quick, --tasks) or raise the limit.", file=sys.stderr)
        return 3
    if a.yes or not sys.stdin.isatty() or not qest.needs_confirm(est, qest.confirm_mode()):
        return None
    sys.stderr.write("Start? [y/N] ")
    sys.stderr.flush()
    try:
        ans = sys.stdin.readline().strip().lower()
    except (OSError, KeyboardInterrupt):
        ans = ""
    if ans in ("y", "yes"):
        return None
    print("not started. (--yes or QUALITYBENCH_CONFIRM=never skips this question)", file=sys.stderr)
    return 0


def _env_update(ctx: Ctx, **kv) -> None:
    """Sidecar-only facts (ENV.json); results.build publishes only the env fields the schema allows."""
    p = ctx.side / "ENV.json"
    try:
        env = json.loads(p.read_text())
    except (OSError, ValueError):
        return
    env.update(kv)
    p.write_text(json.dumps(env, indent=1))


def _finish_timings(ctx: Ctx, a, walls: dict, probe: dict | None, est: dict | None, wall_s: float) -> None:
    _env_update(ctx, wall_min=round(wall_s / 60, 1))
    notes = {k: v for k, v in (n.split("=", 1) for n in a.note if "=" in n) if k in ("gpu", "engine", "precision")}
    try:
        qest.record_timings(ctx.model, a.thinking, a.quick, walls, probe, est, notes or None)
    except OSError as e:
        log_to(ctx, f"note: could not record timings: {e}")
    if est:
        log_to(ctx, f"took {qest.fmt_min(wall_s / 60)} (estimated {qest.fmt_min(est['total_min'])})")
