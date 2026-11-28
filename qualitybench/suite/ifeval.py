from ..taskdef import Task


def run(ctx, client):
    from ..tasks import harness
    harness.run_lmeval(ctx, "ifeval")


TASK = Task(
    id="ifeval", label="IFEval", items=541, order=20, cost=9,
    how="prompt-level strict",
    about="Instruction following (Google, 2023). Each prompt comes with rules a program can check, such as a word limit, a required format or a word to avoid; a reply passes only if it keeps every rule.",
    scored=True, resumable=True, ref_minutes=9, overhead_s=18, needs=frozenset({"lmeval"}),
    prefetch='if "ifeval" in want: print("ifeval", len(load_dataset("google/IFEval", split="train")))\n',
    lmeval_pick=("none", "prompt_level_strict_acc"),
    run=run,
)
