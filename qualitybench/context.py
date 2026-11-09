"""Run state shared by every task: paths in the sidecar, the deadline, and supervised subprocesses."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import http as qhttp
from .progress import emit, log, tqdm_counts


class DeadlineHit(Exception):
    """The deadline passed while this task ran; what it finished is kept."""


class TaskFailed(Exception):
    pass


def parse_deadline(s: str | None, now: float | None = None) -> float | None:
    """RFC 3339 time or Go-style duration ("6h", "90m", "1h30m", "45s") -> epoch seconds."""
    if not s:
        return None
    now = time.time() if now is None else now
    if re.fullmatch(r"(\d+(\.\d+)?(h|m|s|ms))+", s):
        mult = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
        return now + sum(float(n) * mult[u] for n, _, u in re.findall(r"(\d+(\.\d+)?)(ms|h|m|s)", s))
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"--deadline {s!r}: want an RFC 3339 time or a duration like 6h or 90m")
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


def sidecar_for(out: Path) -> Path:
    """Per-item records, raw harness output and logs live next to the results file, never in it."""
    stem = out.name[:-5] if out.name.endswith(".json") else out.name
    return out.parent / f"{stem}.qb"


@dataclass
class Ctx:
    model: str
    endpoint: str            # as given by the user (kept only in env.endpoint, which L80 strips)
    api: str = ""            # the local proxy every request goes through
    side: Path = Path(".")
    quick: bool = False
    thinking: str | None = None
    think_min_tokens: int = 32768
    concurrency: int = 16
    sandbox: str = "docker"
    deadline: float | None = None
    tokenizer: str | None = None
    eval_set: str | None = None
    api_key: str | None = None
    params: dict = field(default_factory=dict)  # --request-param, merged into every chat request by the proxy
    state: dict = field(default_factory=dict)

    # ---- sidecar layout
    def path(self, *p: str) -> Path:
        q = self.side.joinpath(*p)
        q.parent.mkdir(parents=True, exist_ok=True)
        return q

    def items_path(self, task: str) -> Path:
        return self.path("items", f"{task}.jsonl")

    def logf(self):
        return open(self.path("run.log"), "a")

    # ---- partials (state.json), so a rerun knows what was cut and clears it once finished
    def load_state(self) -> None:
        p = self.side / "state.json"
        self.state = json.loads(p.read_text()) if p.is_file() else {}
        self.state.setdefault("partials", {})

    def save_state(self) -> None:
        self.path("state.json").write_text(json.dumps(self.state, indent=1))

    def mark_partial(self, task: str, why: str) -> None:
        self.state["partials"][task] = why
        self.save_state()

    def clear_partial(self, task: str) -> None:
        if self.state["partials"].pop(task, None) is not None:
            self.save_state()

    # ---- deadline
    def past_deadline(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    def check_deadline(self) -> None:
        if self.past_deadline():
            raise DeadlineHit()

    # ---- supervised subprocesses
    def sh(self, cmd: list, task: str, env: dict | None = None, cwd: Path | None = None,
           total: int | None = None, counter: re.Pattern | None = None, shared: dict | None = None) -> None:
        """Run one harness command, logging to run.log and turning its progress into qb:progress lines.

        The process gets its own session, so the deadline or Ctrl-C stops it and all its children; its
        HTTP connections close with it, which cancels its generations on the server.
        """
        self.check_deadline()
        cmd = list(map(str, cmd))
        with self.logf() as lf:
            lf.write(f"\n$ {' '.join(cmd)}\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                 start_new_session=True, text=True, errors="replace", bufsize=1)
            hit = threading.Event()

            def watchdog():
                while p.poll() is None:
                    if self.past_deadline() or qhttp.CANCEL.is_set():
                        hit.set()
                        _kill(p)
                        return
                    time.sleep(1)

            threading.Thread(target=watchdog, daemon=True).start()
            try:
                for line in p.stdout:
                    lf.write(line)
                    if counter is not None and shared is not None and counter.search(line):
                        with _count_lock:
                            shared["done"] = shared.get("done", 0) + 1
                            emit(task, throttle=2, done=min(shared["done"], total or shared["done"]), total=total)
                    elif total:
                        c = tqdm_counts(line)
                        if c:
                            emit(task, throttle=2, done=min(c[0], total), total=total)
                p.wait()
            except BaseException:
                _kill(p)
                raise
        if hit.is_set():
            if qhttp.CANCEL.is_set() and not self.past_deadline():
                raise KeyboardInterrupt()
            raise DeadlineHit()
        if p.returncode != 0:
            raise TaskFailed(f"{Path(cmd[0]).name} exited with status {p.returncode} (see {self.side.name}/run.log)")

    def sh_many(self, cmds: list[list], task: str, parallel: int, **kw) -> None:
        """Run commands side by side, at most `parallel` at a time; the first failure is raised after all stop."""
        errors: list[BaseException] = []
        sem = threading.Semaphore(max(1, parallel))
        shared = kw.pop("shared", {})

        def one(c):
            with sem:
                if errors:
                    return
                try:
                    self.sh(c, task, shared=shared, **kw)
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

        threads = [threading.Thread(target=one, args=(c,), daemon=True) for c in cmds]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for kind in (KeyboardInterrupt, DeadlineHit):
            for e in errors:
                if isinstance(e, kind):
                    raise e
        if errors:
            raise errors[0]


_count_lock = threading.Lock()


def _kill(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def log_to(ctx: Ctx, msg: str) -> None:
    log(msg)
    with ctx.logf() as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
