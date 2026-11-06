# Pre-registered quality thresholds: Launch80/Qwopus3.8-27B-Coder-Flash-Int4-baked

Registered 2026-11-01, BEFORE any quantization or evaluation ran. Do not edit; any change
goes in a dated addendum below with a reason. The sha256 of this file is recorded in
THRESHOLDS.sha256 at registration time.

Reference: Jackrong/Qwopus3.8-27B-Coder-Flash in BF16 (vLLM, CUDA, rented GPU).
Arms: A = INT4 body + dense lm_head/MTP · B = A + baked INT4 lm_head/MTP · C = B served with MTP k=6.
Peer bar: Jackrong/Qwopus3.8-27B-Coder-Flash-GGUF Q4_K_M (llama.cpp).

| # | Metric | Pass if (arm B, vs BF16) |
|---|--------|--------------------------|
| T1 | Perplexity, wikitext-2 test (40 x 2048) | delta <= +3.0 % |
| T2 | Perplexity, code corpus (fixed, published) | delta <= +3.0 % |
| T3 | Mean KL divergence, next-token, 512 x 2048 fixed tokens, top-64 | <= 0.05 nats |
| T4 | Top-1 token agreement, same tokens | >= 93 % |
| T5 | Paired task pass@1 / accuracy, greedy (HumanEval+, MBPP+, LiveCodeBench slice, GSM8K, MMLU-Pro subset, IFEval, BFCL subset) | per task: delta >= -2.0 pts OR 95 % bootstrap CI of the paired delta includes 0 |
| T6 | Mean KLD vs Q4_K_M GGUF peer | ours <= peer |
| T7 | Spec decode: arm C greedy output == arm B greedy output | identical on all parity prompts |

Reported but not gated: needle-in-haystack 32k/128k, vision set, MTP acceptance rate,
per-linear relative output error, BetterBench quick throughput.

Overall verdict: PASS if T1-T7 all pass. Any failure is reported as-is, alongside the numbers.

## Addenda
(none)

### Addendum 2026-11-03: T6 measurement method
The registered bar is unchanged (mean KLD, ours <= Q4_K_M peer). The method changed from `llama-perplexity --kl-divergence`
against a BF16 GGUF base on a rented GPU to: Q4_K_M top-64 logprobs from llama.cpp (b11723 SYCL, B65) on the same eval set,
scored by the same `kld.py` against the same vLLM BF16 reference as the INT4 arms. Q8_0 is run the same way to size the
cross-engine term. Reason: no rental needed and identical metric code for both sides. Decided before T6 ran.
