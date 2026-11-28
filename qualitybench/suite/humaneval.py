from ..taskdef import Task


def run(ctx, client):
    from ..tasks import harness
    harness.run_evalplus(ctx, "humaneval")


TASK = Task(
    id="humaneval",
    label="HumanEval+",
    items=164,
    order=40,
    cost=22,
    how="evalplus base + plus tests, sandboxed",
    about="Short Python programming tasks (OpenAI's HumanEval, 2021) with EvalPlus's extra tests. The model writes a function from its description, and the code is run against hidden tests: it passes or fails.",
    scored=True,
    resumable=True,
    ref_minutes=22,
    overhead_s=25,
    needs=frozenset({"lmeval", "evalplus_image"}),
    # the same lines as mbpp's: one evalplus load fetches both
    prefetch='if want & {"humaneval", "mbpp"}:\n'
             '    from evalplus.data import get_human_eval_plus, get_mbpp_plus\n'
             '    print("humaneval+", len(get_human_eval_plus()), "mbpp+", len(get_mbpp_plus()))\n',
    run=run,
)
