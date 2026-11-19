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
    # What the benchmark measures, for people who do not know it: the report shows it on hover and in a list.
    about: str = ""


SUITE: list[Task] = [
    Task("gsm8k", "GSM8K", 1319, "final answer in the reply, compared numerically, 0-shot chat", cost=19,
         about="Grade-school maths word problems (OpenAI, 2021). The model works out a number and only the final answer is checked, so it tests multi-step arithmetic reasoning."),
    Task("ifeval", "IFEval", 541, "prompt-level strict", cost=9,
         about="Instruction following (Google, 2023). Each prompt comes with rules a program can check, such as a word limit, a required format or a word to avoid; a reply passes only if it keeps every rule."),
    Task("mmlu_pro", "MMLU-Pro", 1000, "fixed 1000-question subset, 14 subjects", cost=42,
         about="Knowledge and reasoning (TIGER-Lab, 2024): multiple-choice questions with up to ten options across 14 subjects, from law and medicine to physics and computer science. A harder successor of MMLU."),
    Task("humaneval", "HumanEval+", 164, "evalplus base + plus tests, sandboxed", cost=22,
         about="Short Python programming tasks (OpenAI's HumanEval, 2021) with EvalPlus's extra tests. The model writes a function from its description, and the code is run against hidden tests: it passes or fails."),
    Task("mbpp", "MBPP+", 378, "evalplus base + plus tests, sandboxed", cost=32,
         about="Basic Python programming problems (Google's MBPP, 2021) with EvalPlus's extra tests. The model writes a small function, and the code is run against hidden tests."),
    Task("bfcl", "BFCL", 400, "4 categories, function calling via the server tool parser", cost=4,
         about="Function calling (Berkeley Function Calling Leaderboard). Given tool definitions, the model has to choose the right function and fill in its arguments correctly, including cases with several calls or no fitting tool."),
    Task("lcb", "LiveCodeBench v6", 175, "fixed 175-problem slice, sandboxed", cost=17,
         about="Competition programming problems (LiveCodeBench) from LeetCode, AtCoder and Codeforces, collected after most models' training data. Solutions are run against hidden tests."),
    Task("logprobs", "Logprobs", 592, "teacher-forced top-64 over fixed 2048-token windows", cost=16,
         about="How confidently the model predicts fixed text, token by token (perplexity), and how far a quantized build's predictions drift from the original's (KL divergence, top-1 agreement). The most sensitive view of quantization loss."),
    Task("needle", "Needle", 20, "7-digit needle at 10 depths, 32k and 128k context", cost=10,
         about="A seven-digit number hidden at ten depths in 32k and 128k tokens of filler text. Checks that one fact can still be found in a very long input."),
    Task("spec_parity", "Spec parity", 32, "greedy text, 256 tokens, compared across runs", default=False, cost=2,
         about="Whether the greedy output is identical with speculative decoding on and off, on fixed prompts. Any difference means the speculative path changes what the model says."),
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
