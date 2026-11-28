from ..taskdef import Task


def run(ctx, client):
    from ..tasks import harness
    harness.run_evalplus(ctx, "mbpp")


TASK = Task(
    id="mbpp",
    label="MBPP+",
    items=378,
    order=50,
    cost=32,
    how="evalplus base + plus tests, sandboxed",
    about="Basic Python programming problems (Google's MBPP, 2021) with EvalPlus's extra tests. The model writes a small function, and the code is run against hidden tests.",
    scored=True,
    resumable=True,
    ref_minutes=32,
    overhead_s=30,
    needs=frozenset({"lmeval", "evalplus_image"}),
    # the same lines as humaneval's: one evalplus load fetches both
    prefetch='if want & {"humaneval", "mbpp"}:\n'
             '    from evalplus.data import get_human_eval_plus, get_mbpp_plus\n'
             '    print("humaneval+", len(get_human_eval_plus()), "mbpp+", len(get_mbpp_plus()))\n',
    run=run,
)
