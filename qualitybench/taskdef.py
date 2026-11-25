"""What a task is: one module in qualitybench/suite/ defines one `TASK = Task(...)`."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class Task:
    id: str
    label: str
    items: int
    how: str
    # Position in the suite: the suite, the results and the report list tasks in this order.
    order: int
    # Full-run minutes on the reference run (estimate.REF_RUN, thinking off). L80 keeps a copy
    # (internal/qualitybench/suite.go RefMinutes) for when this tool can't be asked.
    ref_minutes: float
    default: bool = True
    # Rough cost rank, cheapest first: with --deadline, cheap tasks run first.
    cost: int = 0
    # What the benchmark measures, for people who do not know it: the report shows it on hover and in a list.
    about: str = ""
    # Scored per item as right/wrong (the "tasks" block of the results).
    scored: bool = False
    # The sidecar file whose presence marks a non-scored task finished.
    output: str | None = None
    # A rerun asks only for the missing items, so reply counts add up.
    resumable: bool = False
    # Seconds that don't scale with the server: harness start, scoring, sandboxes. From quick runs where almost
    # all the time was this: a 0.6B model on this machine (gsm8k 18 s, ifeval 22 s, mmlu_pro 28 s, bfcl 7 s) and
    # a 1.7B on a rented 3090 through a tunnel (about 1.5-3x those).
    overhead_s: float = 0
    # Seconds per item no server goes below.
    floor_s: float = 0
    # Bound by prompt processing rather than by generated text.
    prefill: bool = False
    # What `qualitybench setup` installs for it: "lmeval", "mrcr_inputs", "bfcl", "evalplus_image", "lcb".
    needs: frozenset[str] = field(default_factory=frozenset)
    # Its lines of the dataset prefetch script (envs.PREFETCH), run in the lmeval venv with `want` set.
    prefetch: str = ""
    # lm-eval (filter, metric) that scores its samples.
    lmeval_pick: tuple[str, str] | None = None
    # Fixed inputs built from a readable tokenizer before it can run (native.ensure_inputs).
    inputs: str | None = None


def quick_pick(ids: list, key=None) -> list:
    """The fixed quick subset: every tenth item of the sorted list, starting at the first.

    Fixed rather than sampled, so any two quick runs answer exactly the same items.
    """
    return sorted(ids, key=key)[::10]
