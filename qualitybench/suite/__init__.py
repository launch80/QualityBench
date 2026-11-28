"""The v1 suite: one module per task here, and every module in this package must define `TASK = Task(...)`.
L80 keeps the same table in internal/qualitybench/suite.go: the two must not drift."""
from __future__ import annotations

import importlib
import pkgutil

from ..taskdef import Task


def build(found: dict[str, Task]) -> list[Task]:
    """Module name -> its TASK, to the suite in order; two tasks sharing an id or an order fail here, at import."""
    for key in ("id", "order"):
        seen: dict = {}
        for mod, t in found.items():
            v = getattr(t, key)
            if v in seen:
                raise ValueError(f"suite modules {seen[v]} and {mod} share {key} {v!r}")
            seen[v] = mod
    return sorted(found.values(), key=lambda t: t.order)


SUITE = build({m.name: importlib.import_module(f"{__name__}.{m.name}").TASK for m in pkgutil.iter_modules(__path__)})
BY_ID = {t.id: t for t in SUITE}
#: Tasks scored per item as right/wrong (the "tasks" block of the results).
SCORED = tuple(t.id for t in SUITE if t.scored)


def default_ids() -> list[str]:
    return [t.id for t in SUITE if t.default]


def parse_tasks(spec: str | None) -> list[str]:
    """'gsm8k, ifeval' -> suite ids in suite order. None or empty -> the default set."""
    if not spec or not spec.strip():
        return default_ids()
    want = {p.strip().lower() for p in spec.split(",") if p.strip()}
    unknown = sorted(want - BY_ID.keys())
    if unknown:
        raise ValueError(f"unknown task(s) {', '.join(unknown)} (tasks: {', '.join(BY_ID)})")
    return [t.id for t in SUITE if t.id in want]


def run_order(ids: list[str], deadline: bool) -> list[str]:
    """Suite order normally; cheapest first when a deadline may cut the run short."""
    return sorted(ids, key=lambda i: BY_ID[i].cost) if deadline else list(ids)
