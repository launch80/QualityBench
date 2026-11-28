from ..taskdef import Task


def run(ctx, client):
    from ..tasks import harness
    harness.run_lmeval(ctx, "gsm8k")


TASK = Task(
    id="gsm8k", label="GSM8K", items=1319, order=10, cost=19,
    how="final answer in the reply, compared numerically, 0-shot chat",
    about="Grade-school maths word problems (OpenAI, 2021). The model works out a number and only the final answer is checked, so it tests multi-step arithmetic reasoning.",
    scored=True, resumable=True, ref_minutes=19, overhead_s=15, needs=frozenset({"lmeval"}),
    prefetch='if "gsm8k" in want: print("gsm8k", len(load_dataset("openai/gsm8k", "main", split="test")))\n',
    lmeval_pick=("flexible-extract", "exact_match"),
    run=run,
)
