# Changelog

## Unreleased

- **The MRCR match is on the page again.** A merge in 0.4.x lost the report's "MRCR 8-needle match" tile, so the
  number was only in the MRCR section's table view; the tile is back, and "What the benchmarks measure" lists MRCR.
- **Removed: `run --reference`, and run's `--arm` and `--arm-label`.** `run` now writes only its own results
  to `--out` (no more `<name>.qb/run.json`). To compare two runs, run both, then
  `qualitybench compare REF.json TEST.json` (it keeps `--arm`, `--arm-label` and `--thresholds`). Thinking off
  vs on of one model is the same: `compare off.json on.json`, which gives the paired change per task; when the
  two runs' thinking settings differ, the arm label and title name them (`my-model, thinking on`). A
  `--thinking on` run's caveat now says its scores are its thinking-on scores, instead of promising a pair.
  Pages and results files made with `--reference` before still render as they did. Sending the removed flags
  exits 2.
- **Each task is one module (developers).** Everything about a task (its metadata, its estimate, what setup
  installs for it and how it runs) is declared in one module in `qualitybench/suite/`. Adding a task scored
  right/wrong per item is one new file plus its rows in the suite table in `docs/contract.md` and in the L80
  table test. Nothing changes for users.
- **A prompt the server's content filter refuses is scored wrong instead of killing the task.** Azure's filter
  answers a few MMLU-Pro prompts, and a few IFEval prompts it takes for jailbreaks, with HTTP 400 every time; the
  harness retried each until it gave up and took the whole task down. The proxy now answers such a refusal
  with an empty reply (`finish_reason: content_filter`), so the item counts as wrong and the task finishes.
  The end of the task logs how many, and the results carry a caveat such as `MMLU-Pro: the server's content
  filter refused 10 of its prompts; they count as wrong, so this score understates the model.`
- **gsm8k, ifeval and mmlu_pro resume instead of starting over.** lm-eval now saves each reply as it arrives
  (`lmeval/<task>-replies_rank0.db` in the sidecar), so rerunning after a failure or Ctrl-C sends only the
  requests that have no reply yet. Before, one failed request near the end threw away every reply of the task.
- **A run warns up front when a task cannot be built.** logprobs and needle need a readable tokenizer for their
  fixed inputs; if none is available (e.g. the served model has no Hugging Face repo), the run now says so before
  any server request, skips just those tasks with the reason listed in `partials`, and continues with the rest —
  instead of hours into the run failing at each one. A missing `--eval-set` file is caught the same way.
- **The report explains each benchmark.** Hovering a task name, a bar or a tile says in a sentence or two what the
  benchmark measures and where it comes from (GSM8K, IFEval, MMLU-Pro, HumanEval+, MBPP+, BFCL, LiveCodeBench,
  logprobs, needle, spec parity), and a "What the benchmarks measure" list under the task chart collects them.
  The text lives with each task in `qualitybench/suite/` (one module per task); `qualitybench tasks` prints it,
  and `tasks --json` carries it as `about`.
- **The mean-accuracy tile shows both means of a paired report.** It averaged only the thinking-off scores,
  so a run paired with a no-thinking reference read as the reference's number (78.9% instead of its own
  88.9% on one run). It now reads "78.9% / 88.9%" over "thinking off / thinking on". Single runs are unchanged.
- **New opt-in task `mrcr` (MRCR 8-needle).** The `needle` task scores 100% at both lengths on current 27B
  models, so it cannot see what a quantized KV cache or INT4 weights do to long context. `mrcr` runs the
  openai/mrcr 8-needle conversations: eight identical asks hidden in a 64k or 128k conversation, and the
  model must return the asked-for piece with a random prefix. It is graded exactly as the dataset card does:
  0 unless the reply starts with the prefix, else the SequenceMatcher ratio of the reply to the expected text.
  The score is the mean ratio in points. Items are the two length bins that fit a 128k context, kept whole
  (100 each), pinned by the sha256 of their prompt in `qualitybench/data/mrcr-ids.json`; the quick tier is
  every tenth. The text is fixed, so no tokenizer is needed; `setup` or the first run downloads about 460 MB
  of parquet once and keeps the 200 conversations under `<home>/inputs/mrcr`. A conversation over the server's
  context is skipped and listed in `partials`; with thinking on, a reply gets `--thinking-max-tokens` when the
  context has room for it after the prompt, else the room left (at least 2048), so a 128k server runs the 64k
  half whole and most of the 128k half instead of refusing prompt + floor. Results carry an `mrcr` block (per-length summary, per-item
  ratios), the report shows a tile and a section, and `compare` prints a paired delta with a bootstrap CI over
  the shared items (no McNemar: the outcome is a ratio, not right or wrong). It is in no default threshold.
  Greedy, two conversations in flight at a time (prefill-bound), run last under `--deadline`. The time estimate
  is a guess (about 3h20 on the reference box) until a run is in the history. The task stays opt-in until L80's
  `suite.go` and the server's `bench-qualitybench-v1` schema carry it (docs/contract.md).
- **Output tokens per reply.** The proxy now records each chat reply's `usage.completion_tokens` as the
  server reported it, for the scored tasks, in `<name>.qb/state.json`. A task's counts are published only
  when every item's reply was counted; a task resumed after a kill, or from a 0.4.0 `state.json`, has none.
  The results schema is unchanged: mean and median per task travel as one caveat, shown under Caveats in the
  report, and each task's end logs a line such as
  `gsm8k: 1319 replies, 1,234 output tokens per reply (mean), 987 (median)`. `compare` prints the change
  under each task line and adds one caveat for the arm. When only one of the two runs has counts for a task
  (the other was made with 0.4.0 or earlier, its server sent no `usage`, or not every reply was counted),
  `compare` warns `output tokens not compared for <tasks>: ...` and prints no tokens line for it. When the
  server also reports `usage.completion_tokens_details.reasoning_tokens` for every reply of a task, the
  caveat and `compare` add a thinking part. The report gets a "How much the model writes" card for
  non-technical readers: per task the typical reply (median) and the whole benchmark (all tokens), as
  "thinking off → thinking on" or "reference → arm" with a multiplier, an "All tasks" total and the share of
  thinking tokens; its data is written to `<name>.qb/tokens.json` so `qualitybench report` redraws it.
- **`compare` refuses a comparison file as the reference.** Its `.qb` folder is never the reference run's
  data, so deltas were scored against the wrong run; pass the run's own results file.

## 0.4.0 (2026-11-11)

- **MMLU-Pro reads answers written in markdown.** lm-eval looks for `answer is (D)` or `answer is D`, so a
  reply ending "the answer is **(D)**" counted as no answer. When lm-eval finds no answer, QualityBench now
  takes the first letter after "answer is", skipping the `**`, `_` or `:` around it. A letter lm-eval did find
  is kept as it is. On a Qwen3.8-27B thinking-on run this moved 3 of 1000 items (82.9% to 83.2%).
  **MMLU-Pro scores from 0.3.0 and earlier are not comparable with 0.4.0 scores.** If you published results
  made with 0.3.0 or earlier, please retest with 0.4.0 and republish. The README explains why.
- **`--request-param KEY=VALUE`** adds a field to every chat request through the proxy, e.g.
  `reasoning_effort=high` or `chat_template_kwargs.reasoning_budget=8192`. Dotted keys nest and values parse
  as JSON. Sampling, token-limit and request-shape fields are refused. The params are part of the run (a
  resume with other params is refused), kept in `<name>.qb/ENV.json`, shown as a caveat in the results, and
  `compare` warns when two runs differ. `estimate` takes the flag too.
- **Fix: the server measurement no longer takes minutes with `--thinking on`.** It went through the proxy,
  which raised its 128-token requests to the thinking floor. It now talks to the endpoint directly.
- **Tokenizer failures say why.** When the logprobs or needle inputs can't be built because the tokenizer repo
  is gated (with or without `HF_TOKEN`) or isn't a repo at all, the failure now says so. The README explains
  when `--tokenizer` is needed.

Thanks to @tomaszzmuda for the MMLU-Pro, `--request-param`, probe and tokenizer changes (#4, #5, #6, #7).

## 0.3.0 (2026-11-09)

- **GSM8K grading.** GSM8K is now scored by the reply's final answer: the last `\boxed{}` or `#### N`, else
  whichever comes last of a bold quantity (`**$132.00**`) and `Answer: N`, else the final sentence's last money
  amount or number. It is compared numerically, so `$7.00` matches 7 and "Judy makes $7,425 in 1 week." reads
  7425, not 1. lm-eval's stock flexible-extract plus exact_match misread those replies. Re-grading the six full
  Qwopus study runs adds 126–133 correct items per thinking-off run (about +9.5 points; 63 with thinking on),
  and loses at most 2 items that the stock scoring had counted correct. The prompt is unchanged. Each item keeps the stock result as `stock`, and the log prints both counts.
  Older runs are re-graded from their lm-eval samples when built or compared. `compare` warns when one side
  can't be re-graded.
- **Token caps.** The proxy counts chat replies that stopped at the token cap (`finish_reason: length`) and
  replies with reasoning, per task. Any task where 1% or more were cut gets a caveat in the results and a log
  line. So does a model that reasoned while `--thinking` was left at the server default. The preflight now
  also catches reasoning returned inline as `<think>` by a server with no reasoning parser.
  `--thinking-max-tokens` now defaults to 32768 (was 16384).
- **Podman.** The code tasks run with docker or podman. `QUALITYBENCH_CONTAINER=podman` picks podman, which is
  also used when docker is missing. Podman runs get `--userns=keep-id` and `--security-opt label=disable`.
  The evalplus image is fully qualified (`docker.io/ganler/evalplus@sha256:...`).
- **Task scores without logprobs.** A results file with an empty `logprobs` block is no longer reported as
  unpublishable; launch80 accepts task-only runs and shows perplexity and KLD as n/a. The results carry a
  caveat, and the logprobs error on servers without `prompt_logprobs` says how to skip the task.

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
