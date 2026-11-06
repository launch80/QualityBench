"""How long a run will take, before it starts.

Three sources, best first:

- history: earlier runs of this model with the same thinking setting (``timings.jsonl`` in the
  QualityBench home). Their per-task wall time, scaled by item count and, when both runs were
  probed, by the probe speeds.
- probe: a short speed test of the server (16 parallel chats plus one long prefill), compared with
  the same probe on the reference run's server. Each task's reference time is scaled by that.
- reference: the reference run's per-task times as they are, when the server can't be measured.

A task's minutes = fixed overhead (harness start, scoring) + the scalable part × items fraction ×
reference speed ÷ this server's speed, × a thinking factor for generated-text tasks.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

from . import envs
from . import http as qhttp
from .suite import BY_ID

#: The run the reference table was measured on.
REF_RUN = "Qwopus3.8-27B INT4 on one Arc Pro B65, vLLM 0.30, thinking off (2026-11-02)"

#: Full-run minutes per task on the reference run (thinking off). L80 keeps the same table
#: (internal/qualitybench/suite.go RefMinutes) for when this tool can't be asked.
REF_MINUTES = {"gsm8k": 19, "ifeval": 9, "mmlu_pro": 42, "humaneval": 22, "mbpp": 32, "bfcl": 4, "lcb": 17,
               "logprobs": 16, "needle": 6, "spec_parity": 2}

#: The probe on the reference run's server (output tokens/s across 16 parallel chats; prompt tokens/s
#: for one long prompt): the same model and box, measured three times on 2026-11-06 (229–235, 1267–1290).
REF_DECODE_TPS = 232.0
REF_PREFILL_TPS = 1280.0

#: Seconds per task that don't scale with the server: harness start, scoring, sandboxes. From quick
#: runs where almost all the time was this: a 0.6B model on this machine (gsm8k 18 s, ifeval 22 s,
#: mmlu_pro 28 s, bfcl 7 s) and a 1.7B on a rented 3090 through a tunnel (about 1.5-3x those).
OVERHEAD_S = {"gsm8k": 15, "ifeval": 18, "mmlu_pro": 25, "humaneval": 25, "mbpp": 30, "bfcl": 5, "lcb": 35,
              "logprobs": 10, "needle": 10, "spec_parity": 5}

#: Seconds per item no server goes below. A logprobs window returns 2048 × 64 logprobs (about 0.7 MB
#: of JSON), so over a tunnel it is bound by the network: 3.6 s a window on that quick run.
FLOOR_S = {"logprobs": 1.5}

#: Tasks bound by prompt processing rather than by generated text.
PREFILL_TASKS = {"logprobs", "needle"}

#: Thinking on writes several times the text. A guess until a thinking-on run is in the history.
THINK_FACTOR = 4.0

#: A probe measures speed, not how much a model writes (a small model answers briefly) nor a slow
#: tunnel or a busy sandbox host, hence the wide range until this model has history.
RANGES = {"history": (0.85, 1.25), "probe": (0.45, 2.0), "reference": (0.4, 2.5)}

PROBE_TTL_S = 600


def items_for(task: str, quick: bool) -> int:
    n = BY_ID[task].items
    return math.ceil(n / 10) if quick else n


# --- history -------------------------------------------------------------------------------------
def timings_path() -> Path:
    return envs.home() / "timings.jsonl"


def load_history(model: str, thinking: str | None) -> list[dict]:
    p = timings_path()
    if not p.is_file():
        return []
    want = thinking or "server default"
    out = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("model") == model and r.get("thinking") == want:
            out.append(r)
    return out


def record_timings(model: str, thinking: str | None, quick: bool, tasks: dict, probe: dict | None,
                   estimate: dict | None, notes: dict | None = None) -> None:
    """Append one run's per-task wall times. Local only; never published."""
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model, "thinking": thinking or "server default",
           "quick": quick, "tasks": tasks, "probe": probe,
           "estimate_min": (estimate or {}).get("total_min"), **({"notes": notes} if notes else {})}
    p = timings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _from_history(task: str, items: int, hist: list[dict], probe: dict | None) -> float | None:
    for r in reversed(hist):
        t = (r.get("tasks") or {}).get(task)
        if not t or not t.get("items") or not t.get("wall_s"):
            continue
        ov = OVERHEAD_S.get(task, 0)
        scal = max(0.0, t["wall_s"] - ov) * items / t["items"]
        then = r.get("probe") or {}
        key = "prefill_tps" if task in PREFILL_TASKS else "decode_tps"
        if probe and then.get(key) and probe.get(key):
            scal *= then[key] / probe[key]
        return (ov + scal) / 60
    return None


# --- probe ---------------------------------------------------------------------------------------
_PROBE_PROMPT = ("Write a short, plain explanation of how a hash table handles collisions, then give a "
                 "worked example with five keys. ") * 6
_PREFILL_TEXT = ("The quick brown fox jumps over the lazy dog while the committee reviews the budget. ") * 220


def _cache_path() -> Path:
    return envs.home() / "probe-cache.json"


def cached_probe(endpoint: str, model: str) -> dict | None:
    try:
        d = json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return None
    e = d.get(f"{endpoint}|{model}")
    if e and time.time() - e.get("at", 0) < PROBE_TTL_S:
        return e["probe"]
    return None


def _save_probe(endpoint: str, model: str, probe: dict) -> None:
    p = _cache_path()
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        d = {}
    d = {k: v for k, v in d.items() if time.time() - v.get("at", 0) < PROBE_TTL_S}
    d[f"{endpoint}|{model}"] = {"at": time.time(), "probe": probe}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def _completion_tokens(r: dict) -> int:
    u = r.get("usage") or {}
    if u.get("completion_tokens"):
        return int(u["completion_tokens"])
    m = (r.get("choices") or [{}])[0].get("message") or {}
    text = (m.get("content") or "") + (m.get("reasoning") or m.get("reasoning_content") or "")
    return max(1, len(text) // 4)


def probe(client: qhttp.Client, model: str, concurrency: int = 16, max_tokens: int = 128) -> dict:
    """Measure the server: aggregate output tokens/s at `concurrency`, and prompt tokens/s.

    Assumes the model is loaded (run it after the warmup). Takes seconds on a fast server.
    """
    msgs = [{"role": "user", "content": _PROBE_PROMPT}]
    extra = {"ignore_eos": True}
    try:  # servers that reject ignore_eos still work without it (the replies just may end early)
        client.post("/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 4, "temperature": 0,
                                             **extra}, timeout=600)
    except qhttp.HTTPError:
        extra = {}
    toks, errs = [0] * concurrency, []

    def one(i):
        try:
            r = client.post("/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": max_tokens,
                                                     "temperature": 0, **extra}, timeout=600)
            toks[i] = _completion_tokens(r)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    t0 = time.monotonic()
    th = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(concurrency)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.monotonic() - t0
    if errs and not sum(toks):
        raise errs[0]
    decode = sum(toks) / max(wall, 1e-3)

    # A fresh prefix each time, so a server's prefix cache can't answer the prefill from memory.
    text = f"[{os.urandom(8).hex()}] " + _PREFILL_TEXT
    t0 = time.monotonic()
    r = client.post("/v1/chat/completions", {"model": model, "max_tokens": 1, "temperature": 0,
                                             "messages": [{"role": "user", "content": text}]}, timeout=600)
    lat = time.monotonic() - t0
    ptoks = int((r.get("usage") or {}).get("prompt_tokens") or len(_PREFILL_TEXT) // 4)
    return {"decode_tps": round(decode, 1), "prefill_tps": round(ptoks / max(lat, 1e-3), 1),
            "concurrency": concurrency}


# --- setup ---------------------------------------------------------------------------------------
def setup_minutes(tasks: list[str], sandbox: str) -> float:
    """Rough one-time install time for what these tasks still need (0 once set up)."""
    m = 0.0
    lm = set(tasks) & {"gsm8k", "ifeval", "mmlu_pro", "humaneval", "mbpp", "logprobs", "needle"}
    if lm and not _stamped("lmeval"):
        m += 4
    if "bfcl" in tasks and not _stamped("bfcl"):
        m += 2
    if set(tasks) & {"humaneval", "mbpp"} and sandbox == "docker" and not _image(envs.EVALPLUS_IMAGE):
        m += 2
    if "lcb" in tasks:
        if sandbox == "native":
            m += 0 if _stamped("lcb") else 5
        elif not _image(envs.LCB_IMAGE):
            m += 6
    return m


def _stamped(name: str) -> bool:
    return envs._stamp(name).is_file()


def _image(image: str) -> bool:
    try:
        return envs.docker_image_exists(image)
    except OSError:
        return False


# --- the estimate ----------------------------------------------------------------------------------
def estimate(tasks: list[str], quick: bool, thinking: str | None, model: str, probe_result: dict | None,
             sandbox: str = "docker", history: list[dict] | None = None) -> dict:
    hist = load_history(model, thinking) if history is None else history
    rows, sources = [], set()
    for t in tasks:
        n = items_for(t, quick)
        m = _from_history(t, n, hist, probe_result)
        src = "history"
        if m is None:
            ov = OVERHEAD_S.get(t, 0) / 60
            scal = max(0.0, REF_MINUTES.get(t, 2) - ov) * n / BY_ID[t].items
            if probe_result:
                if t in PREFILL_TASKS:
                    scal *= REF_PREFILL_TPS / max(probe_result.get("prefill_tps") or REF_PREFILL_TPS, 1)
                else:
                    scal *= REF_DECODE_TPS / max(probe_result.get("decode_tps") or REF_DECODE_TPS, 1)
                src = "probe"
            else:
                src = "reference"
            if thinking == "on" and t not in PREFILL_TASKS:
                scal *= THINK_FACTOR
            m = max(ov + scal, n * FLOOR_S.get(t, 0) / 60)
        sources.add(src)
        rows.append({"id": t, "label": BY_ID[t].label, "min": round(m, 1), "items": n, "source": src})
    worst = "reference" if "reference" in sources else "probe" if "probe" in sources else "history"
    lo_f, hi_f = RANGES[worst]
    if thinking == "on" and worst != "history":
        lo_f, hi_f = lo_f * 0.75, hi_f * 1.5
    setup = setup_minutes(tasks, sandbox)
    total = sum(r["min"] for r in rows)
    basis = {"history": "from earlier runs of this model",
             "probe": "measured on this server just now",
             "reference": f"from the reference run: {REF_RUN}"}[worst]
    doc = {"source": worst, "total_min": round(total + setup, 1),
           "range_min": [round(total * lo_f + setup, 1), round(total * hi_f + setup, 1)],
           "tasks": rows, "probe": None, "setup_min": setup, "basis": basis}
    if probe_result:
        doc["probe"] = {**probe_result,
                        "speed": round(probe_result.get("decode_tps", 0) / REF_DECODE_TPS, 2)}
    return doc


def measure(endpoint: str, model: str, api_key: str | None = None, *, use_cache: bool = True,
            log=None) -> tuple[dict | None, str | None]:
    """Probe the endpoint (or reuse a probe from the last 10 minutes). Returns (probe, warning)."""
    if use_cache:
        c = cached_probe(endpoint, model)
        if c:
            return c, None
    client = qhttp.Client(endpoint, api_key, attempts=1)
    try:
        qhttp.wait_ready(client, attempts=3, pause=5)
        client.post("/v1/chat/completions", {"model": model, "max_tokens": 4, "temperature": 0,
                                             "messages": [{"role": "user", "content": "Reply OK."}]}, timeout=1800)
        p = probe(client, model)
    except (OSError, qhttp.HTTPError, RuntimeError, ValueError) as e:
        return None, f"could not measure the server ({str(e)[:160]}); the estimate uses the reference run"
    _save_probe(endpoint, model, p)
    return p, None


def fmt_min(m: float) -> str:
    if m < 1:
        return "<1 min"
    m = round(m)
    return f"{m // 60}h{m % 60:02d}" if m >= 60 else f"{m} min"


def summary(doc: dict) -> list[str]:
    lo, hi = doc["range_min"]
    lines = [f"Estimated time: about {fmt_min(doc['total_min'])} ({fmt_min(lo)}–{fmt_min(hi)}), {doc['basis']}"]
    lines.append("  " + " · ".join(f"{r['label']} {fmt_min(r['min'])}" for r in doc["tasks"]))
    if doc.get("setup_min"):
        lines.append(f"  includes about {fmt_min(doc['setup_min'])} of one-time setup")
    if doc.get("warning"):
        lines.append("  note: " + doc["warning"])
    return lines


# --- the gate ------------------------------------------------------------------------------------
def confirm_mode() -> str:
    v = (os.environ.get("QUALITYBENCH_CONFIRM") or "always").strip().lower()
    return v if v in ("always", "never") or v.startswith("over:") else "always"


def needs_confirm(doc: dict, mode: str) -> bool:
    if mode == "never":
        return False
    if mode.startswith("over:"):
        from .context import parse_deadline
        try:
            lim = (parse_deadline(mode[5:], now=0) or 0) / 60
        except ValueError:
            try:
                lim = float(mode[5:])
            except ValueError:
                return True
        return doc["total_min"] > lim
    return True
