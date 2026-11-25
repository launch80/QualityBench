from ..taskdef import Task


def run(ctx, client):
    from ..tasks import native
    native.run_needle(ctx, client)


def prepare(ctx):
    from ..tasks import native
    native.ensure_inputs(ctx, "needle")


TASK = Task(
    id="needle", label="Needle", items=20, order=90, cost=10,
    how="7-digit needle at 10 depths, 32k and 128k context",
    about="A seven-digit number hidden at ten depths in 32k and 128k tokens of filler text. Checks that one fact can still be found in a very long input.",
    output="needle.json", ref_minutes=6, overhead_s=10, prefill=True, needs=frozenset({"lmeval"}),
    run=run, prepare=prepare,
)
