from ..taskdef import Task

TASK = Task(
    id="spec_parity", label="Spec parity", items=32, order=100, default=False, cost=2,
    how="greedy text, 256 tokens, compared across runs",
    about="Whether the greedy output is identical with speculative decoding on and off, on fixed prompts. Any difference means the speculative path changes what the model says.",
    output="parity.json", ref_minutes=2, overhead_s=5,
)
