from ..taskdef import Task


def run(ctx, client):
    from ..tasks import native
    native.run_logprobs(ctx, client)


TASK = Task(
    id="logprobs", label="Logprobs", items=592, order=80, cost=16,
    how="teacher-forced top-64 over fixed 2048-token windows",
    about="How confidently the model predicts fixed text, token by token (perplexity), and how far a quantized build's predictions drift from the original's (KL divergence, top-1 agreement). The most sensitive view of quantization loss.",
    output="logprobs.npz", ref_minutes=16, overhead_s=10, prefill=True, needs=frozenset({"lmeval"}), inputs="evalset",
    # A window returns 2048 × 64 logprobs (about 0.7 MB of JSON), so over a tunnel it is bound by the
    # network: 3.6 s a window on that quick run.
    floor_s=1.5,
    run=run,
)
