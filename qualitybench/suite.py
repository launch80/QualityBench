"""The v1 suite. L80 keeps the same table in internal/qualitybench/suite.go: the two must not drift."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    items: int
    how: str
    default: bool = True
    # Rough cost rank, cheapest first: with --deadline, cheap tasks run first.
    cost: int = 0


SUITE: list[Task] = [
    Task("gsm8k", "GSM8K", 1319, "final answer in the reply, compared numerically, 0-shot chat", cost=19),
    Task("ifeval", "IFEval", 541, "prompt-level strict", cost=9),
    Task("mmlu_pro", "MMLU-Pro", 1000, "fixed 1000-question subset, 14 subjects", cost=42),
    Task("humaneval", "HumanEval+", 164, "evalplus base + plus tests, sandboxed", cost=22),
    Task("mbpp", "MBPP+", 378, "evalplus base + plus tests, sandboxed", cost=32),
    Task("bfcl", "BFCL", 400, "4 categories, function calling via the server tool parser", cost=4),
    Task("lcb", "LiveCodeBench v6", 175, "fixed 175-problem slice, sandboxed", cost=17),
    Task("logprobs", "Logprobs", 592, "teacher-forced top-64 over fixed 2048-token windows", cost=16),
    Task("needle", "Needle", 20, "7-digit needle at 10 depths, 32k and 128k context", cost=10),
    Task("spec_parity", "Spec parity", 32, "greedy text, 256 tokens, compared across runs", default=False, cost=2),
]

#: Tasks scored per item as right/wrong (the "tasks" block of the results).
SCORED = ("gsm8k", "ifeval", "mmlu_pro", "humaneval", "mbpp", "bfcl", "lcb")
BY_ID = {t.id: t for t in SUITE}


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


def quick_pick(ids: list, key=None) -> list:
    """The fixed quick subset: every tenth item of the sorted list, starting at the first.

    Fixed rather than sampled, so any two quick runs answer exactly the same items.
    """
    return sorted(ids, key=key)[::10]
