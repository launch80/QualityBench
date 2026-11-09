"""Assemble the results file (a `bench.qualitybench.v1` document, schema_version 1) from a run's sidecar."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from . import SCHEMA_VERSION, __version__, items, stats
from .suite import BY_ID, SCORED

MAX_BYTES = 256 * 1024
CAP_SHARE = 0.01  # a task whose replies hit the token cap this often gets a caveat
ENV_KEYS = ("date", "model", "precision", "engine", "gpu", "lm_eval", "evalplus", "bfcl_eval", "evalplus_image",
            "sandbox", "thinking_on_min_max_tokens", "endpoint", "host")
NOTE_ENV = {"precision": "precision", "engine": "engine", "gpu": "gpu"}

BASE_CAVEATS = {
    "lcb": "LiveCodeBench v6 problems may predate the model's training cutoff, so absolute scores may be inflated by "
           "contamination. This affects every run equally.",
    "gsm8k": "GSM8K (0-shot chat) is scored by the reply's final answer (\\boxed{} or '####', else the last bold "
             "quantity or 'Answer:', else the final sentence's last amount), compared numerically, so $7.00 matches 7. "
             "lm-eval's stock flexible-extract score, kept per item, misreads such replies and reads several points lower.",
    "subsets": "MMLU-Pro is a fixed 1000-question subset and BFCL a 400-item subset of four categories.",
}


def item_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def file_hash(p: str | Path | None) -> str | None:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).is_file() else None


def parse_thresholds(path: str | Path) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 3 and re.fullmatch(r"T\d+", c[0]):
            rows.append({"id": c[0][:8], "metric": c[1][:300], "rule": c[2][:300]})
    return rows[:24]


def task_rows(side: Path, on_side: Path | None = None, on_src: str | None = None) -> list[dict]:
    rows = []
    for t in SCORED:
        R = items.load(side / "items" / f"{t}.jsonl")
        if R is None:
            continue
        row = {"id": t, "label": BY_ID[t].label, "how": BY_ID[t].how, "off": stats.score(R)}
        if on_side is not None:
            T = items.load(on_side / "items" / f"{t}.jsonl")
            if T is not None and T.keys() == R.keys():
                row["on"] = stats.score(T, on_src)
                row["paired"] = stats.paired(R, T)
        rows.append(row)
    return rows


_HOME = re.compile(r"(/home/|/Users/|/root/|~/|[A-Za-z]:\\Users\\)")
_LAN = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|\[?::1|[^/\s]*\.local\b)")


def scrub(doc: dict) -> list[str]:
    """Remove machine details L80 refuses to publish (home paths, LAN URLs) outside env.endpoint/env.host.

    A model served from a local directory reports its path as the model id; that becomes the directory name.
    Returns what was changed, for the log.
    """
    changed = []

    def walk(v, path):
        if isinstance(v, dict):
            return {k: (x if path == "/env" and k in ("endpoint", "host") else walk(x, f"{path}/{k}")) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, f"{path}/{i}") for i, x in enumerate(v)]
        if isinstance(v, str) and (_HOME.search(v) or _LAN.search(v)):
            changed.append(path)
            if _LAN.search(v):
                return _LAN.sub("<local>", v)
            return re.sub(r"(?:/home/|/Users/|/root/|~/|[A-Za-z]:\\Users\\)\S*?([^/\\\s]+)(?=$|[\s\"')])", r"\1", v)
        return v

    new = walk(doc, "")
    doc.clear()
    doc.update(new)
    return changed


def build(side: Path, *, title: str | None = None, thresholds: str | None = None, notes: list[str] | None = None,
          on_side: Path | None = None, on_src: str | None = None) -> dict:
    env_raw = json.loads((side / "ENV.json").read_text())
    state = json.loads((side / "state.json").read_text()) if (side / "state.json").is_file() else {}
    quick = bool(state.get("quick"))
    model = env_raw["model"]
    for sd in (side, on_side):
        if sd is not None:
            items.regrade_gsm8k(sd)
    tasks = task_rows(side, on_side, on_src)
    env = {k: env_raw[k] for k in ENV_KEYS if env_raw.get(k) is not None}
    caveats = []
    for n in notes or []:
        k, _, v = n.partition("=")
        if k.strip().lower() in NOTE_ENV and v.strip():
            env[NOTE_ENV[k.strip().lower()]] = v.strip()[:120]
        elif n.strip():
            caveats.append(f"Note: {n.strip()}"[:600])
    ran = {t["id"] for t in tasks}
    if not ran & {"humaneval", "mbpp"}:
        env.pop("evalplus_image", None)
    if quick:
        caveats.insert(0, "Quick run: a fixed tenth of each task's items. Compare it only with other quick runs; "
                          "intervals are about three times wider than a full run's.")
    if env_raw.get("request_params"):
        from .proxy import describe
        caveats.insert(0, (f"Request params on every chat request: {describe(env_raw['request_params'])}. "
                           "Compare it with runs that used the same params.")[:600])
    if env_raw.get("thinking") == "on":
        caveats.insert(0, "This run had thinking ON (max_tokens at least "
                          f"{env_raw.get('thinking_on_min_max_tokens')}); its scores sit in the thinking-off slot "
                          "because it has no thinking-off run to pair with.")
    caps = state.get("caps", {})
    cap_notes = []
    for t in SCORED:
        c = caps.get(t)
        if t not in ran or not c or not c.get("replies"):
            continue
        share = c["length"] / c["replies"]
        if share >= CAP_SHARE:
            fix = ("raise --thinking-max-tokens" if c.get("thinking") == "on" else
                   "use --thinking on (with --thinking-max-tokens) or --thinking off")
            cap_notes.append(f"{BY_ID[t].label}: {c['length']} of {c['replies']} replies ({share:.0%}) stopped at the "
                             f"token cap and count as wrong, so this score understates the model; {fix}.")
    if not env_raw.get("thinking") or env_raw.get("thinking") == "server default":
        seen = sum(c.get("reasoning", 0) for c in caps.values())
        if seen:
            cap_notes.insert(0, f"The model reasoned in {seen} replies, but thinking was left at the server's default, "
                                "so the harnesses' short token limits (2048, 4096 for LiveCodeBench) applied. Rerun with "
                                "--thinking on or --thinking off.")
    caveats[:0] = cap_notes
    if "lcb" in ran:
        caveats.append(BASE_CAVEATS["lcb"])
    if "gsm8k" in ran:
        caveats.append(BASE_CAVEATS["gsm8k"])
    if ran & {"mmlu_pro", "bfcl"}:
        caveats.append(BASE_CAVEATS["subsets"])
    doc = {"schema_version": SCHEMA_VERSION, "qualitybench_version": __version__,
           "title": (title or f"{model}: QualityBench{' quick run' if quick else ''}")[:160], "model": model[:200],
           "generated": datetime.now().astimezone().isoformat(timespec="minutes"), "env": env,
           "stats": {"resamples": stats.RESAMPLES, "seed": stats.SEED}, "tasks": tasks, "logprobs": {}}
    hashes = {f"items:{t['id']}": item_hash(items.load(side / "items" / f"{t['id']}.jsonl")) for t in tasks}
    if (side / "logprobs.npz").is_file():
        from .tasks.native import logprob_summary
        doc["logprobs"], doc["eval_meta"] = logprob_summary(side / "logprobs.npz")
        hashes["eval-set ids"] = doc["eval_meta"].get("eval_set_sha256")
    else:
        caveats.append("No logprobs in this run (not asked for, or the server has no prompt_logprobs, a vLLM "
                       "extension), so perplexity, KLD and top-1 agreement are n/a. Task scores are unaffected.")
    if (side / "needle.json").is_file():
        nd = json.loads((side / "needle.json").read_text())
        doc["needle"] = {"model": nd.get("model", model)[:200], "summary": nd["summary"], "items": nd["items"]}
    if thresholds:
        doc["thresholds"] = parse_thresholds(thresholds)
        hashes["THRESHOLDS.md"] = file_hash(thresholds)
    doc["hashes"] = hashes
    if on_src:
        doc["think_dirs"] = [on_src[:80]]
    doc["partials"] = [p[:200] for p in state.get("partials", {}).values()][:32]
    doc["caveats"] = caveats[:12]
    return doc


def write(doc: dict, out: Path) -> list[str]:
    """Write the results file atomically; returns warnings (scrubbed fields, size)."""
    warn = [f"removed a local path or LAN URL from {p}" for p in scrub(doc)]
    data = json.dumps(doc, separators=(",", ":"))
    if len(data) > MAX_BYTES:
        warn.append(f"results are {len(data) // 1024} KB, over launch80's 256 KB limit")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(data)
    tmp.replace(out)
    return warn


def validate_shape(doc: dict) -> list[str]:
    """What launch80 would refuse, beyond the schema: the server needs at least one scored task.

    The logprobs block may be empty (servers without prompt_logprobs); the page shows KLD and perplexity as n/a.
    """
    probs = []
    if not doc.get("tasks"):
        probs.append("no scored task (launch80 needs at least one of gsm8k, ifeval, mmlu_pro, humaneval, mbpp, bfcl, lcb)")
    return probs
