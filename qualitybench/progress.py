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
        sys.stdout.write("qb:progress " + json.dumps({"task": task, **fields}, separators=(",", ":")) + "\n")
        sys.stdout.flush()


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
