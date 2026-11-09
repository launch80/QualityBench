"""Tasks scored by third-party harnesses: lm-eval, evalplus, BFCL and LiveCodeBench.

Each harness runs in its own locked environment (envs.py) and talks only to the local proxy. Each task
writes items/<task>.jsonl last and atomically, so a rerun skips finished tasks, and evalplus and LCB also
skip finished shards and problems.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .. import envs, items
from ..context import Ctx, TaskFailed, log_to
from ..progress import emit
from ..suite import BY_ID, quick_pick

# ------------------------------------------------------------------ lm-eval: gsm8k, ifeval, mmlu_pro

def _mmlu_samples() -> dict[str, list[int]]:
    return json.loads(envs.pkg_file("data", "mmlu-pro-samples.json").read_text())


def lmeval_samples(task: str, quick: bool) -> dict | None:
    """The --samples map for lm-eval, or None for the full task."""
    if task == "mmlu_pro":
        s = _mmlu_samples()
        if quick:
            flat = quick_pick([(k, i) for k, v in s.items() for i in v])
            s = {}
            for k, i in flat:
                s.setdefault(k, []).append(i)
        return s
    if quick:
        return {task: quick_pick(list(range(BY_ID[task].items)))}
    return None


def run_lmeval(ctx: Ctx, task: str) -> None:
    out = ctx.path("lmeval", task)
    samples = lmeval_samples(task, ctx.quick)
    tasks = ",".join(samples) if task == "mmlu_pro" else task
    total = sum(map(len, samples.values())) if samples else BY_ID[task].items
    margs = (f"model={ctx.model},base_url={ctx.api}/v1/chat/completions,num_concurrent={ctx.concurrency},"
             f"max_retries=3,tokenized_requests=False,timeout=3600")
    cmd = [envs.venv_bin("lmeval", "lm-eval"), "run", "--model", "local-chat-completions", "--model_args", margs,
           "--apply_chat_template", "--num_fewshot", "0", "--log_samples", "--output_path", out,
           "--gen_kwargs", "temperature=0,max_gen_toks=2048", "--tasks", tasks]
    if samples:
        cmd += ["--samples", json.dumps(samples)]
    emit(task, done=0, total=total)
    ctx.sh(cmd, task, env=envs.harness_env({"OPENAI_API_KEY": "local"}), total=total)
    recs = []
    for t in tasks.split(","):
        f = _newest(out, f"samples_{t}_*.jsonl")
        recs += items.from_lmeval(f, t)
    k, n = items.write(ctx.items_path(task), recs)
    stock = f" (lm-eval's stock scoring: {sum(r['stock'] for r in recs)}/{n})" if task == "gsm8k" else ""
    log_to(ctx, f"{task}: {k}/{n} correct{stock}")


def _newest(root: Path, pattern: str) -> Path:
    found = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise TaskFailed(f"no {pattern} under {root.name}/ (see run.log)")
    return found[-1]


# ------------------------------------------------------------------ evalplus: humaneval, mbpp

_CODEGEN = re.compile(r"Codegen: \S+")


def run_evalplus(ctx: Ctx, ds: str) -> None:
    lm_py = envs.venv_python("lmeval")
    env = envs.harness_env({"OPENAI_API_KEY": "local"})
    import subprocess
    rargs = [lm_py, envs.pkg_file("helpers", "evalplus_ranges.py"), ds, str(ctx.concurrency)] + (["--quick"] if ctx.quick else [])
    ranges = [tuple(l.split()) for l in subprocess.run(list(map(str, rargs)), env=env, capture_output=True, text=True,
                                                       check=True).stdout.split("\n") if l.strip()]
    root = ctx.path("evalplus", f".shards-{ds}")
    # Shard ranges are deterministic for a given (dataset, concurrency, tier), so an interrupted run resumes:
    # evalplus skips problems already in each shard's output file.
    cmds = [[lm_py, envs.pkg_file("helpers", "evalplus_codegen.py"), "--model", ctx.model, "--dataset", ds,
             "--backend", "openai", "--base-url", f"{ctx.api}/v1", "--greedy", "--id_range", f"[{lo},{hi}]",
             "--root", root / f"{lo}-{hi}"] for lo, hi in ranges]
    total = BY_ID[ds].items if not ctx.quick else len(ranges)
    emit(ds, done=0, total=total)
    ctx.sh_many(cmds, ds, ctx.concurrency, env=env, total=total, counter=_CODEGEN, shared={})
    dest = ctx.path("evalplus", ds, "x").parent
    shard_files = sorted(root.glob(f"*/{ds}/*_temp_0.0.jsonl"))
    shard_files = [f for f in shard_files if not f.name.endswith(".raw.jsonl")]
    if not shard_files:
        raise TaskFailed(f"evalplus {ds}: no samples written (see run.log)")
    name = shard_files[0].name
    for fname in (name, name[:-6] + ".raw.jsonl"):
        with open(dest / fname, "w") as out:
            for f in sorted(root.glob(f"*/{ds}/{fname}")):
                out.write(f.read_text())
    samples = dest / name
    ids_file = dest / f"{ds}-answered.txt"
    ctx.sh([lm_py, envs.pkg_file("helpers", "evalplus_pad.py"), ds, samples, ids_file], ds, env=env)
    answered = set(ids_file.read_text().split())
    log_to(ctx, f"evalplus {ds}: {len(answered)} samples, scoring with no network")
    emit(ds, state="scoring")
    if ctx.sandbox == "native":
        ctx.sh([envs.venv_bin("lmeval", "evalplus.evaluate"), "--dataset", ds, "--samples", samples,
                "--i-just-wanna-run"], ds, env=env)
    else:
        cache = envs.home() / "xdg" / "evalplus"
        ctx.sh(envs.container_run("none", [(dest.parent, "/work", "rw"), (cache, "/cache", "ro")],
                                  ["--memory", "8g", "--pids-limit", "1024", "--cap-drop", "ALL", "--security-opt",
                                   "no-new-privileges", "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp/xc"])
               + [envs.EVALPLUS_IMAGE, "bash", "-c",
                f"mkdir -p /tmp/xc/evalplus && cp /cache/* /tmp/xc/evalplus/ && "
                f"evalplus.evaluate --dataset {ds} --samples /work/{ds}/{name} --i-just-wanna-run"], ds)
    res = dest / (name[:-6] + "_eval_results.json")
    k, n = items.write(ctx.items_path(ds), [i for i in items.from_evalplus(res) if i["id"] in answered])
    log_to(ctx, f"{ds}: {k}/{n} correct")


# ------------------------------------------------------------------ BFCL

BFCL_CATEGORIES = "simple_python,multiple,parallel,parallel_multiple"


def bfcl_ids(quick: bool) -> dict[str, list[str]]:
    ids = json.loads(envs.pkg_file("data", "bfcl-ids.json").read_text())
    if quick:
        ids = {c: quick_pick(v, key=lambda s: int(s.rsplit("_", 1)[1])) for c, v in ids.items()}
    return ids


def run_bfcl(ctx: Ctx) -> None:
    root = ctx.path("bfcl", "x").parent
    ids = bfcl_ids(ctx.quick)
    (root / "test_case_ids_to_generate.json").write_text(json.dumps(ids, indent=1))
    total = sum(map(len, ids.values()))
    env = envs.harness_env({"BFCL_PROJECT_ROOT": str(root), "OPENAI_BASE_URL": f"{ctx.api}/v1", "OPENAI_API_KEY": "local"})
    py, shim = envs.venv_python("bfcl"), envs.pkg_file("helpers", "bfcl_run.py")
    emit("bfcl", done=0, total=total)
    ctx.sh([py, shim, ctx.model, "generate", "--run-ids", "--temperature", "0", "--num-threads", ctx.concurrency],
           "bfcl", env=env, total=total)
    emit("bfcl", state="scoring")
    ctx.sh([py, shim, ctx.model, "evaluate", "--test-category", BFCL_CATEGORIES, "--partial-eval"], "bfcl", env=env)
    score = sorted((root / "score").glob("*-FC"))
    if not score:
        raise TaskFailed("bfcl: no score directory written (see run.log)")
    k, n = items.write(ctx.items_path("bfcl"), items.from_bfcl(score[0]))
    log_to(ctx, f"bfcl: {k}/{n} correct")


# ------------------------------------------------------------------ LiveCodeBench

def lcb_ids(quick: bool) -> list[str]:
    ids = envs.pkg_file("data", "lcb-ids.txt").read_text().split()
    return quick_pick(ids) if quick else ids


_LCB_GEN = re.compile(r"lcb: generated \d+/\d+")


def _lcb_pad(work: Path) -> None:
    """lcb_runner scores only a complete v6 slice; problems a quick run skipped get empty code (scored as fails
    and dropped from the results)."""
    custom = json.loads((work / "custom.json").read_text())
    have = {c["question_id"] for c in custom}
    custom += [{"question_id": q, "code_list": [""]} for q in lcb_ids(False) if q not in have]
    (work / "custom.json").write_text(json.dumps(custom))


def run_lcb(ctx: Ctx) -> None:
    work = ctx.path("lcb", "x").parent
    ids = lcb_ids(ctx.quick)
    (work / "ids.txt").write_text("\n".join(ids) + "\n")
    helpers = envs.pkg_file("helpers")
    done = sum(1 for _ in open(work / "raw.jsonl")) if (work / "raw.jsonl").is_file() else 0
    emit("lcb", done=min(done, len(ids)), total=len(ids))
    gen_args = ["--base", ctx.api, "--model", ctx.model, "--workers", str(ctx.concurrency)]
    score_args = ["--scenario", "codegeneration", "--release_version", "v6", "--num_process_evaluate", "8", "--timeout", "6"]
    if ctx.sandbox == "native":
        py, src = envs.home() / "venvs" / "lcb" / "bin" / "python", envs.home() / "lcb-src"
        env = envs.harness_env({"HF_DATASETS_OFFLINE": "1"})
        ctx.sh([py, helpers / "lcb_gen.py", *gen_args, "--out", work, "--ids", work / "ids.txt"], "lcb", env=env, cwd=src,
               total=len(ids), counter=_LCB_GEN, shared={"done": done})
        _lcb_pad(work)
        emit("lcb", state="scoring")
        ctx.sh([py, "-m", "lcb_runner.runner.custom_evaluator", "--custom_output_file", work / "custom.json", *score_args],
               "lcb", env=env, cwd=src)
    else:
        # generation: host network only to reach the local proxy; scoring: no network, capped, unprivileged
        base = envs.lcb_docker_base("host", [(helpers, "/k", "ro"), (work, "/work", "rw")], ["-e", "HF_DATASETS_OFFLINE=1"])
        ctx.sh(base + [envs.LCB_IMAGE, "python", "/k/lcb_gen.py", *gen_args, "--out", "/work", "--ids", "/work/ids.txt"],
               "lcb", total=len(ids), counter=_LCB_GEN, shared={"done": done})
        _lcb_pad(work)
        emit("lcb", state="scoring")
        base = envs.lcb_docker_base("none", [(work, "/work", "rw")],
                                    ["-e", "HF_DATASETS_OFFLINE=1", "--memory", "16g", "--pids-limit", "2048",
                                     "--cap-drop", "ALL", "--security-opt", "no-new-privileges"])
        ctx.sh(base + [envs.LCB_IMAGE, "python", "-m", "lcb_runner.runner.custom_evaluator",
                       "--custom_output_file", "/work/custom.json", *score_args], "lcb")
    res = work / "custom_codegeneration_output_eval_all.json"
    if not res.is_file():
        raise TaskFailed("lcb: scorer wrote no results (see run.log)")
    answered = set(ids)
    k, n = items.write(ctx.items_path("lcb"), [i for i in items.from_lcb(res) if i["id"] in answered])
    log_to(ctx, f"lcb: {k}/{n} correct")


def harness_versions() -> dict:
    """Pinned harness versions, read from the shipped locks (what setup installs)."""
    def pin(lock, pkg):
        for line in envs.pkg_file("locks", lock).read_text().splitlines():
            if line.lower().startswith(pkg + "=="):
                return line.split("==", 1)[1].strip()
        return None
    return {"lm_eval": pin("lmeval.txt", "lm-eval"), "evalplus": pin("lmeval.txt", "evalplus"),
            "bfcl_eval": pin("bfcl.txt", "bfcl-eval")}


def cleanup_shards(ctx: Ctx, ds: str) -> None:
    shutil.rmtree(ctx.side / "evalplus" / f".shards-{ds}", ignore_errors=True)
