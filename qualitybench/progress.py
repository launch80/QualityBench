"""Machine-readable progress lines for L80's console (docs/contract.md, "Progress lines")."""
from __future__ import annotations

import json
import re
import sys
import threading
import time

_lock = threading.Lock()
_last: dict[str, float] = {}


def emit(task: str, throttle: float = 0.0, **fields) -> None:
    """Print `qb:progress {...}` to stdout. With throttle, at most one line per task per that many seconds."""
    now = time.monotonic()
    with _lock:
        if throttle and now - _last.get(task, 0) < throttle:
            return
        _last[task] = now
        if TRACKER is not None:
            TRACKER.update(task, fields)
            if task in TRACKER.started:
                fields = {**fields, "elapsed_s": round(now - TRACKER.started[task])}
        _write("qb:progress " + json.dumps({"task": task, **fields}, separators=(",", ":")))


def _write(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit_line(kind: str, obj: dict) -> None:
    """Print `qb:<kind> {...}` (e.g. qb:estimate)."""
    with _lock:
        _write(f"qb:{kind} " + json.dumps(obj, separators=(",", ":")))


class Tracker:
    """The whole run's progress and time left, from per-task estimates and the task lines.

    The running task's time left comes from its own rate once it is a few percent in; tasks not yet
    started use their estimate, corrected by how the finished tasks compared with theirs.
    """

    def __init__(self, est_min: dict[str, float], order: list[str], clock=time.monotonic):
        self.clock = clock
        self.est = {t: max(0.05, est_min.get(t, 1.0)) * 60 for t in order}
        self.order = list(order)
        self.t0 = clock()
        self.started: dict[str, float] = {}
        self.ended: dict[str, float] = {}
        self.frac: dict[str, float] = {}
        self.current: str | None = None

    def start(self, task: str) -> None:
        self.current = task
        self.started[task] = self.clock()
        self.frac[task] = 0.0

    def end(self, task: str) -> None:
        if task in self.started:
            self.ended[task] = self.clock()
        self.frac[task] = 1.0

    def update(self, task: str, f: dict) -> None:
        if f.get("total") and f.get("done") is not None:
            # Generation is most of a task; scoring (state=scoring) is the rest.
            self.frac[task] = max(self.frac.get(task, 0.0), 0.9 * min(1.0, f["done"] / f["total"]))
        if f.get("state") == "scoring":
            self.frac[task] = max(self.frac.get(task, 0.0), 0.9)
        if f.get("state") in ("scored", "partial", "failed"):
            self.frac[task] = 1.0

    def ratio(self) -> float:
        """Observed ÷ estimated over the finished tasks (1 until one finishes)."""
        obs = sum(self.ended[t] - self.started[t] for t in self.ended)
        est = sum(self.est[t] for t in self.ended)
        return min(3.0, max(0.3, obs / est)) if est > 0 and obs > 0 else 1.0

    def snapshot(self) -> dict:
        now = self.clock()
        r = self.ratio()
        total = sum(self.est.values())
        done_w, left = 0.0, 0.0
        for t in self.order:
            e = self.est[t]
            if t in self.ended:
                done_w += e
                continue
            f = self.frac.get(t, 0.0)
            if t in self.started:
                el = now - self.started[t]
                done_w += e * f
                if f >= 0.05 and el > 5:
                    left += el / f - el
                else:
                    left += max(e * r - el, e * r * (1 - f) * 0.2)
            else:
                left += e * r
        idx = self.order.index(self.current) + 1 if self.current in self.order else 0
        return {"task": self.current, "task_index": idx, "tasks": len(self.order),
                "elapsed_s": round(now - self.t0), "eta_s": round(left),
                "est_total_s": round(total), "fraction": round(min(1.0, done_w / total) if total else 0.0, 3)}

    def emit(self) -> None:
        with _lock:
            _write("qb:progress " + json.dumps({"run": self.snapshot()}, separators=(",", ":")))


TRACKER: Tracker | None = None


def track_run(tr: Tracker | None, every: float = 10.0) -> threading.Event:
    """Install `tr` and emit its run line every `every` seconds until the returned event is set."""
    global TRACKER
    TRACKER = tr
    stop = threading.Event()
    if tr is None:
        return stop

    def loop():
        while not stop.wait(every):
            tr.emit()

    threading.Thread(target=loop, daemon=True).start()
    return stop


def log(msg: str) -> None:
    sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    sys.stdout.flush()


# tqdm bars as the harnesses print them: "...| 593/1319 [01:02<...". Used to turn a harness's
# own progress into qb:progress lines while it runs.
_TQDM = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


def tqdm_counts(line: str) -> tuple[int, int] | None:
    m = None
    for m in _TQDM.finditer(line):
        pass
    return (int(m.group(1)), int(m.group(2))) if m else None
