# Changelog

## 0.2.0 (2026-11-06)

- **Time estimate before a run.** `run` measures the server for a few seconds after the warmup (16 parallel
  chats and one long prompt), scales the reference run's per-task times by the speed difference and prints an
  estimate with a range. After a run, it uses that model's real times from `timings.jsonl` in the QualityBench
  home, which never leaves the machine.
- **Approval.** On a terminal, `run` asks `Start? [y/N]` before the first task. `--yes` skips it,
  `QUALITYBENCH_CONFIRM=never|over:30m` changes the default, and runs without a terminal are never asked.
  `--max-minutes N` refuses runs that could take longer (exit 3). Harness setup now happens after the question.
- **`qualitybench estimate`** prints the estimate without running anything, with `--json` for L80 and agents.
- **Progress:** a `qb:estimate` line before the first task, `elapsed_s` on task lines, and a run line every
  10 seconds with the overall fraction and time left (`eta_s`). L80 draws them as an overall bar.
- The estimate and the run's wall time are kept in `<name>.qb/ENV.json`, not in the results file (the
  `bench.qualitybench.v1` env has no room for them).

## 0.1.1 (2026-11-06)

- Preflight: a model that thinks by default, run without `--thinking`, now stops at once and asks for
  `--thinking off` (the usual baseline) or `--thinking on`. Before, it failed with "the model returned no
  content", because the reasoning used up the preflight's tokens, and the harnesses' token limits would
  have cut its replies off mid-thought anyway.
- README: links to the intro video and the launch post.

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
