# QualityBench default thresholds

The bars `qualitybench compare` scores a test run on when no `--thresholds` file is given. They are the
pre-registered bars of the Qwopus 3.8 INT4 study (2026-11-01), without its GGUF peer row. Copy this file,
change the numbers, and pass it with `--thresholds` to use your own; its sha256 is recorded in the results.
Ids have fixed meanings; the first number in each rule is the bar.

| # | Metric | Pass if (test vs reference) |
|---|--------|-----------------------------|
| T1 | Perplexity, wikitext-2 test (40 x 2048) | delta <= +3.0 % |
| T2 | Perplexity, code corpus (40 x 2048) | delta <= +3.0 % |
| T3 | Mean KL divergence, next-token, 512 x 2048 fixed tokens, top-64 | <= 0.05 nats |
| T4 | Top-1 token agreement, same tokens | >= 93 % |
| T5 | Paired task accuracy, greedy (every task in the reference) | per task: delta >= -2.0 pts OR 95 % bootstrap CI of the paired delta includes 0 |
| T7 | Spec parity: test greedy output == reference greedy output | identical on all parity prompts |
