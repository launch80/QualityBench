"""Tasks QualityBench runs itself: teacher-forced logprobs, needle retrieval and spec parity."""
from __future__ import annotations

import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import envs
from .. import http as qhttp
from ..context import Ctx, TaskFailed, log_to
from ..progress import emit
from ..suite import quick_pick

TOPK = 64
SETS = ("ppl-wiki", "ppl-code", "kld")
CHUNK = 16


# ------------------------------------------------------------------ fixed inputs (tokenizer-specific)

def inputs_dir(tokenizer: str) -> Path:
    return envs.home() / "inputs" / hashlib.sha256(tokenizer.encode()).hexdigest()[:16]


def ensure_inputs(ctx: Ctx, what: str) -> Path:
    """Build eval-set.json / needle.json for this tokenizer once (in the lmeval env) and cache them."""
    if what == "evalset" and ctx.eval_set:
        return Path(ctx.eval_set)
    tok = ctx.tokenizer or ctx.model
    d = inputs_dir(tok)
    f = d / ("eval-set.json" if what == "evalset" else "needle.json")
    if not f.is_file():
        envs.ensure_venv("lmeval")
        log_to(ctx, f"{what}: building the fixed inputs for tokenizer {tok} (one time)")
        try:
            ctx.sh([envs.venv_python("lmeval"), envs.pkg_file("helpers", "build_inputs.py"), "--tokenizer", tok,
                    "--out", d, "--what", what], "logprobs" if what == "evalset" else "needle", env=envs.harness_env())
        except TaskFailed as e:
            why = getattr(e, "last", "")
            if why.startswith("tokenizer: "):
                why = why.removeprefix("tokenizer: ")
                if not ctx.tokenizer:
                    why += ". It came from the model id; pass --tokenizer <HF repo or local dir>"
                raise TaskFailed(f"{why} (see {ctx.side.name}/run.log)")
            if ctx.tokenizer:
                raise TaskFailed(f"{e}. It failed building inputs for --tokenizer {tok}")
            raise TaskFailed(f"{e}. The tokenizer comes from --tokenizer, else the model id ({tok}); pass "
                             "--tokenizer <HF repo or local dir> if the served name is not a Hugging Face repo")
    return f


# ------------------------------------------------------------------ logprobs

def _score_window(client: qhttp.Client, model: str, ids: list[int]):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": TOPK,
            "add_special_tokens": False}
    try:
        r = client.post("/v1/completions", body)
    except qhttp.HTTPError as e:
        if e.status == 400 and "logprobs" in e.body:
            raise TaskFailed(f"the server refused prompt_logprobs={TOPK}: start vLLM with --max-logprobs {TOPK} "
                             f"(and speculative decoding off). Server said: {e.body[:200]}")
        raise
    pl = r["choices"][0].get("prompt_logprobs")
    if not pl:
        raise TaskFailed("the server returned no prompt_logprobs: logprobs needs vLLM (or a server with the same "
                         "extension) started with --max-logprobs 64 and speculative decoding off. The task scores still count; "
                         "to skip this task, pass --tasks without logprobs (e.g. "
                         "--tasks gsm8k,ifeval,mmlu_pro,humaneval,mbpp,bfcl,lcb,needle)")
    L = len(ids)
    tgt = np.empty(L - 1, np.float32)
    tid = np.zeros((L - 1, TOPK), np.int32)
    tlp = np.full((L - 1, TOPK), -np.inf, np.float16)
    for p in range(1, L):
        d = pl[p]  # {token_id: {"logprob": x, "rank": r, ...}}; includes the true token
        tgt[p - 1] = d[str(ids[p])]["logprob"]
        top = sorted(((int(t), v["logprob"]) for t, v in d.items() if v.get("rank", 0) <= TOPK), key=lambda x: -x[1])[:TOPK]
        for j, (t, lp) in enumerate(top):
            tid[p - 1, j] = t
            tlp[p - 1, j] = lp
    if not np.isfinite(tgt).all():
        raise TaskFailed("non-finite prompt logprobs: is speculative decoding on?")
    return tgt, tid, tlp


def run_logprobs(ctx: Ctx, client: qhttp.Client) -> None:
    es_path = ensure_inputs(ctx, "evalset")
    es = json.loads(Path(es_path).read_text())
    sha = es["meta"]["sha256_ids"]
    picks = {s: (quick_pick(list(range(len(es[s])))) if ctx.quick else list(range(len(es[s])))) for s in SETS}
    total = sum(map(len, picks.values()))
    done = 0
    lock = threading.Lock()
    emit("logprobs", done=0, total=total)
    arrays = {}
    for s in SETS:
        idx = picks[s]
        parts = []
        for c0 in range(0, len(idx), CHUNK):
            chunk = idx[c0:c0 + CHUNK]
            f = ctx.path("logprobs", f"{s}-{sha[:12]}-{'q' if ctx.quick else 'f'}-{c0:04d}.npz")
            if f.is_file():  # finished on an earlier attempt
                z = np.load(f)
                parts.append((z["target"], z["ids"], z["lp"]))
                done += len(chunk)
                emit("logprobs", done=done, total=total)
                continue
            ctx.check_deadline()

            def one(i):
                nonlocal done
                r = _score_window(client, ctx.model, es[s][i])
                with lock:
                    done += 1
                    emit("logprobs", throttle=2, done=done, total=total)
                return r

            with ThreadPoolExecutor(min(4, ctx.concurrency)) as ex:
                res = list(ex.map(one, chunk))
            t, i_, l_ = (np.stack([r[k] for r in res]) for k in range(3))
            tmp = f.with_name(f.name + ".tmp.npz")
            np.savez(tmp, target=t, ids=i_, lp=l_)
            tmp.replace(f)
            parts.append((t, i_, l_))
        arrays[f"{s}.target"] = np.concatenate([p[0] for p in parts])
        arrays[f"{s}.ids"] = np.concatenate([p[1] for p in parts])
        arrays[f"{s}.lp"] = np.concatenate([p[2] for p in parts])
        log_to(ctx, f"logprobs {s}: {len(idx)} windows, ppl {math.exp(-arrays[f'{s}.target'].astype(np.float64).mean()):.4f}")
    meta = {"model": ctx.model, "topk": TOPK, "eval_set_sha256": sha, "quick": ctx.quick,
            "windows": {s: picks[s] for s in SETS}, "vocab": es["meta"].get("vocab")}
    out = ctx.path("logprobs.npz")
    tmp = out.with_name("logprobs.tmp.npz")
    np.savez_compressed(tmp, **arrays, meta=json.dumps(meta))
    tmp.replace(out)
    emit("logprobs", state="scored", done=total, total=total)


def logprob_summary(npz: Path) -> tuple[dict, dict]:
    """The `logprobs` and `eval_meta` blocks of the results (the study's qb_report.py, unchanged)."""
    z = np.load(npz)
    meta = json.loads(str(z["meta"]))
    lp = {}
    for s, label in (("ppl-wiki", "wikitext-2"), ("ppl-code", "code"), ("kld", "KLD set")):
        if f"{s}.target" not in z:
            continue
        nll = -z[f"{s}.target"].astype(np.float64)
        mass = np.exp(z[f"{s}.lp"].astype(np.float64)).sum(-1)
        win = np.exp(nll.mean(1))
        edges = np.linspace(0, nll.shape[1], 33).astype(int)
        lp[s] = {"label": label, "windows": int(nll.shape[0]), "positions": int(nll.shape[1]),
                 "ppl": float(np.exp(nll.mean())), "win_ppl": [round(float(v), 4) for v in win],
                 "nll_by_pos": [[int(edges[i]), int(edges[i + 1]), float(nll[:, edges[i]:edges[i + 1]].mean())] for i in range(32)],
                 "mass_mean": float(mass.mean()), "mass_p01": float(np.quantile(mass, 0.01)), "mass_p50": float(np.median(mass)),
                 "window_tokens": int(nll.shape[1]) + 1}
    em = {"model": meta.get("model"), "topk": meta.get("topk"), "eval_set_sha256": meta.get("eval_set_sha256")}
    return lp, em


# ------------------------------------------------------------------ needle

def run_needle(ctx: Ctx, client: qhttp.Client) -> None:
    prompts = json.loads(ensure_inputs(ctx, "needle").read_text())["prompts"]
    if ctx.quick:
        prompts = prompts[::10]
    out = ctx.path("needle-items.jsonl")
    have = {}
    if out.is_file():
        for line in open(out):
            d = json.loads(line)
            have[(d["length"], d["depth"])] = d
    emit("needle", done=len(have), total=len(prompts))
    skipped = []
    for p in prompts:
        key = (p["length"], p["depth"])
        if key in have:
            continue
        if any(s == p["length"] for s in skipped):
            continue
        ctx.check_deadline()
        try:
            r = client.chat(ctx.model, [{"role": "user", "content": p["prompt"]}], max_tokens=24, temperature=0)
        except qhttp.HTTPError as e:
            if e.status == 400:  # longer than the server's context: skip this length, say so
                skipped.append(p["length"])
                log_to(ctx, f"needle {p['length']}: the server refused the prompt ({e}); skipping this length")
                continue
            raise
        ans = (r["choices"][0]["message"].get("content") or "").strip()
        d = {"length": p["length"], "depth": p["depth"], "correct": p["number"] in ans, "answer": ans[:40],
             "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens", 0)}
        with open(out, "a") as f:
            f.write(json.dumps(d) + "\n")
        have[key] = d
        emit("needle", done=len(have), total=len(prompts))
    if skipped:
        ctx.mark_partial("needle", f"needle: {', '.join(f'{s // 1024}k' for s in sorted(set(skipped)))} longer than the server's context")
    its = [have[(p["length"], p["depth"])] for p in prompts if (p["length"], p["depth"]) in have]
    lengths = sorted({d["length"] for d in its})
    summary = {str(L): sum(d["correct"] for d in its if d["length"] == L) / max(1, sum(1 for d in its if d["length"] == L))
               for L in lengths}
    ctx.path("needle.json").write_text(json.dumps({"model": ctx.model, "summary": summary, "items": its}, indent=1))
    emit("needle", state="scored", correct=sum(d["correct"] for d in its), total=len(its))


# ------------------------------------------------------------------ spec parity

def run_parity(ctx: Ctx, client: qhttp.Client) -> None:
    prompts = json.loads(envs.pkg_file("data", "parity-prompts.json").read_text())
    out = []
    emit("spec_parity", done=0, total=len(prompts))
    for i, p in enumerate(prompts):
        ctx.check_deadline()
        r = client.chat(ctx.model, [{"role": "user", "content": p}], max_tokens=256, temperature=0.0,
                        logprobs=True, top_logprobs=5, seed=608)
        ch = r["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        out.append({"prompt": p, "text": ch["message"].get("content") or "", "tokens": [t["token"] for t in lp],
                    "top": [{u["token"]: u["logprob"] for u in t["top_logprobs"]} for t in lp]})
        emit("spec_parity", throttle=2, done=i + 1, total=len(prompts))
    ctx.path("parity.json").write_text(json.dumps(out))
    emit("spec_parity", state="scored", done=len(prompts), total=len(prompts))


def parity_compare(test: list, ref: list) -> dict:
    """Greedy text of `test` against `ref`; at the first differing token, test's top-1 minus top-2 logprob."""
    same, rows = 0, []
    for i, (p, q) in enumerate(zip(test, ref)):
        if p["text"] == q["text"]:
            same += 1
            continue
        tp, tq = p["tokens"], q["tokens"]
        k = next((j for j in range(min(len(tp), len(tq))) if tp[j] != tq[j]), min(len(tp), len(tq)))
        top = sorted(p["top"][k].items(), key=lambda kv: -kv[1])[:2] if k < len(p["top"]) else []
        row = {"i": i, "token": k, "a": (tp[k] if k < len(tp) else "")[:80], "b": (tq[k] if k < len(tq) else "")[:80]}
        if len(top) == 2:
            row["a_top2_gap"] = round(top[0][1] - top[1][1], 4)
        rows.append(row)
    return {"identical": same == len(test), "n": len(test), "mismatches": rows[:64]}
