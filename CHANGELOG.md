# Changelog

## 0.1.0 (2026-11-06)

First release. QualityBench is extracted from the kit that ran the Qwopus3.8-27B BF16-vs-INT4 study.

- `qualitybench run`: the v1 suite (GSM8K, IFEval, MMLU-Pro subset, HumanEval+, MBPP+, BFCL subset, LiveCodeBench v6 slice,
  teacher-forced logprobs, needle at 32k/128k, optional spec parity) against any OpenAI-compatible endpoint. It writes a
  `bench.qualitybench.v1` results file, a self-contained HTML report, and per-item data in `<name>.qb/`.
- `--quick` (a fixed tenth of every task), `--tasks`, `--thinking on|off` through a local proxy with a preflight check,
  `--deadline` with partial results, and resume by rerunning the same command.
- `qualitybench compare`: KLD, top-1 agreement, perplexity change and paired task deltas (exact McNemar p and
  paired-bootstrap CI) against a reference run, with verdicts against a thresholds file.
- Request lifecycle: one-hour timeouts, at most 3 attempts, and abandoned requests cancelled on the server.
- `qualitybench setup`: pinned harness environments (lm-eval 0.4.13, evalplus 0.3.1, bfcl-eval 2026.3.23,
  LiveCodeBench 28fef95e), each locked separately.
- `qb:progress` lines for the L80 console.
- Regression test: re-scoring the study's per-item files reproduces its published task numbers exactly.
