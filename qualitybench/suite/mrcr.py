from ..taskdef import Task


def run(ctx, client):
    from ..tasks import native
    native.run_mrcr(ctx, client)


# Opt-in until L80's suite.go carries it: 200 conversations of 33k-128k tokens, prefill-heavy, so it runs last.
TASK = Task(
    id="mrcr",
    label="MRCR 8-needle",
    items=200,
    order=110,
    default=False,
    cost=60,
    how="openai/mrcr: 8 identical asks in a 64k or 128k conversation, SequenceMatcher ratio of the asked-for reply",
    about="Long-context retrieval (OpenAI MRCR): eight identical asks hidden in a 64k- or 128k-token conversation, graded as the reply's match to the asked-for text. Where needle saturates at 100%, quantization damage first shows.",
    output="mrcr.json",
    overhead_s=15,
    prefill=True,
    needs=frozenset({"lmeval", "mrcr_inputs"}),
    # Not part of the reference run: 14.3M prompt tokens at its 1280 tok/s prefill is about 3h05, plus the
    # replies. A guess until a run of this model is in the history.
    ref_minutes=200,
    run=run,
)
