from ..taskdef import Task


def run(ctx, client):
    from ..tasks import harness
    harness.run_lmeval(ctx, "mmlu_pro")


TASK = Task(
    id="mmlu_pro", label="MMLU-Pro", items=1000, order=30, cost=42,
    how="fixed 1000-question subset, 14 subjects",
    about="Knowledge and reasoning (TIGER-Lab, 2024): multiple-choice questions with up to ten options across 14 subjects, from law and medicine to physics and computer science. A harder successor of MMLU.",
    scored=True, resumable=True, ref_minutes=42, overhead_s=25, needs=frozenset({"lmeval"}),
    prefetch='if "mmlu_pro" in want: print("mmlu_pro", len(load_dataset("TIGER-Lab/MMLU-Pro", split="test")))\n',
    lmeval_pick=("custom-extract", "exact_match"),
    run=run,
)
