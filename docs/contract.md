# QualityBench ↔ L80 contract

`L80 qualitybench` runs the `qualitybench` console script from this repository. L80 owns
task selection, endpoint discovery and publishing. The tool owns generation,
scoring, statistics and the HTML report. This file is what each side may rely on.
A change to it is a change to both.

Status: implemented by QualityBench 0.1.0. L80 pins a release commit in its tool manifest;
`L80_QUALITYBENCH` runs a local build. The same file lives in L80-CLI as `docs/qualitybench.md`.

## Invocation

```
qualitybench --version                       prints a bare version, e.g. 0.1.0
qualitybench run --endpoint URL --model ID --out PATH
    [--quick] [--tasks a,b,c] [--thinking on|off]
    [--reference REF.json] [--thresholds T.md] [--deadline T]
    [--note KEY=VALUE]...  [extra flags passed through]
qualitybench compare REF.json TEST.json [--thresholds T.md] [--out PATH]
qualitybench setup [--tasks a,b,c] [--sandbox docker|native]
qualitybench tasks [--json]
```

Flags L80 does not send by name, but may pass through: `--title`, `--concurrency N`
(default 16), `--sandbox docker|native`, `--tokenizer`, `--eval-set`, `--arm`,
`--arm-label`, `--thinking-max-tokens`.

- `--reference REF.json`: when the reference is the same model with thinking off
  and this run has `--thinking on`, the two are paired task by task (`on`/`paired`).
  Otherwise this run becomes an arm of the reference's document. Either way `--out`
  holds the combined document and this run's own results go to `<name>.qb/run.json`.
  The reference's `<ref>.qb/` folder must sit next to it.
- `--note precision=…`, `engine=…` and `gpu=…` fill `env`; other notes become caveats.
- `run` installs the harness environments it needs on first use (`setup` does it
  ahead of time); a run without network after setup needs no downloads.

- `--endpoint` is a base URL including `/v1`.
- `--tasks` takes ids from the suite table below, in any order. L80 sends it
  only for a subset; with no `--tasks` the tool runs its own default set, which
  must match the table.
- `--quick` runs a fixed tenth of each task's items. The lists are fixed, not
  sampled, so quick runs compare with each other. The results say so.
- `--deadline` takes an RFC 3339 time or a Go-style duration (`6h`, `90m`).
  The tool orders tasks so cheap ones go first, starts no work after the
  deadline, and marks cut tasks in `partials` instead of dropping them.
- Environment: `QUALITYBENCH_API_KEY` is the endpoint's bearer key, which never
  arrives as a flag. `QUALITYBENCH_NO_UPDATE_CHECK=1` means L80 owns version alignment.
  `QUALITYBENCH_CONTAINER=docker|podman` picks the container CLI for the code tasks
  (default: docker if installed, else podman). (0.3.0)

- `--yes` starts without asking about the estimated time. Without it, a run on
  a terminal asks `Start? [y/N]` (a "no" exits 0 and runs nothing);
  `QUALITYBENCH_CONFIRM=always|never|over:<duration>` sets when to ask. A run
  without a terminal never asks. L80 asks itself and passes `--yes`. (0.2.0)
- `--max-minutes N` refuses to start, exit 3, when the estimate's upper bound
  is over N minutes. `--no-probe` skips the server speed test. (0.2.0)

Exit status: 0 when the results file is complete or partial by deadline; 1 when
a task failed (the results file is still written, with the task in `partials`);
2 on usage; 3 when `--max-minutes` stopped the run before it started; 130 when
interrupted. On any exit the tool keeps what it finished.

### `qualitybench estimate` (0.2.0)

```
qualitybench estimate --model M [--endpoint URL] [--quick] [--tasks a,b] [--thinking on|off] [--no-probe] [--json]
```

Runs nothing. With `--endpoint` it measures the server (16 parallel chats and
one long prompt, a few seconds once the model is loaded; reused for 10
minutes). With `--json` it prints one object:

```json
{"source": "probe", "total_min": 38.2, "range_min": [17.2, 76.4],
 "tasks": [{"id": "gsm8k", "label": "GSM8K", "min": 6.1, "items": 132, "source": "probe"}],
 "probe": {"decode_tps": 2140.0, "prefill_tps": 9800.0, "concurrency": 16, "speed": 9.2},
 "setup_min": 0, "basis": "measured on this server just now"}
```

`source` is the weakest source any task used: `history` (earlier runs of this
model with the same thinking setting, from `timings.jsonl` in the QualityBench
home), `probe` or `reference` (the reference run's times, when the server
can't be measured). If the server can't be reached it still exits 0, with
source `reference` and a `warning`. `setup_min` is the one-time install still
needed. `run` prints the same object as a `qb:estimate` line before the first
task, and keeps it (with the run's `wall_min`) in `<name>.qb/ENV.json`. It is
not part of the results file.

## Suite (v1)

| id | items | default |
|---|---|---|
| gsm8k | 1319 | yes |
| ifeval | 541 | yes |
| mmlu_pro | 1000 (fixed subset, 14 subjects) | yes |
| humaneval | 164 | yes |
| mbpp | 378 | yes |
| bfcl | 400 (4 categories) | yes |
| lcb | 175 (v6 slice) | yes |
| logprobs | 592 windows × 2048 | yes |
| needle | 20 (32k and 128k × 10 depths) | yes |
| spec_parity | 32 | no, only when named |

`internal/qualitybench/suite.go` holds the same table, with the measured
minutes the TUI uses for estimates. The two must not drift: a task added to
the tool is added here in the same release that pins it.

## Behaviour L80 relies on

These come from the 2026-11 Qwopus study, where each one failed at least once:
- **Resume:** rerunning the same command skips finished tasks and finished
  shards. L80 tells users to "rerun the same command".
- **Request lifecycle:**
  - Each request gets a long timeout (at least an hour with thinking on) and
    at most 3 attempts.
  - The tool **cancels** a generation it abandons, by closing the connection
    or sending an explicit abort.
  - It never resends a request that is still running server-side. evalplus
    0.3.1 kills a request after 100 s and retries it forever; behind a thinking
    proxy that filled the KV cache with orphaned generations and stalled a run
    for 3.5 h.
- **Warmup:** the first request after start is a warmup and is never scored.
  On llama-swap it pays the model load.
- **Concurrency:** every harness runs concurrently. Sequential harnesses such
  as evalplus codegen are sharded by item range.
- **Sandbox:** code is executed with no network.

## Outputs

- `--out PATH`: the results file, described below.
- `PATH` with `.json` replaced by `.html`: a self-contained HTML report that
  works offline.
- Per-item records, graded text, logprob arrays and `run.log` go in `<name>.qb/`
  next to the results file. They are never part of the publish payload, and
  `compare` needs them.

### Results file

The results file is a `bench.qualitybench.v1` document **as the server
validates it**, minus `$template`, which L80 adds. Requirements:
- It has `schema_version: 1` and `model`. L80 refuses other versions with a
  pointer to `L80 update`.
- It stays under **256 KB** (the server's limit), so summaries and per-window
  series only.
- It may carry `env.endpoint` and `env.host`. L80 strips both. It must not
  carry any other machine detail: L80 refuses to publish a file with a
  home-directory path (`/home/`, `/Users/`, `~/`, `C:\Users\`) or a URL to
  localhost, a private address or a `.local` host anywhere in it.
- `partials` lists every task cut short.

The server requires at least one scored task. The `logprobs` block may be empty: servers
without vLLM's `prompt_logprobs` extension (llama.cpp, radiance) can submit task scores alone, and
the page shows perplexity, KLD and top-1 agreement as n/a (QualityBench 0.3.0 adds a caveat saying
so). A run with no scored task (only `needle`, for example) writes a valid local file that launch80
will not accept; the tool says so at the end of the run.

The schema itself lives with the server template
(`lib/artifacts/schemas/bench-qualitybench-v1.ts` in the launch80 repo).
`lib/skills/sample-qualitybench.json` there is a real example, and
`internal/qualitybench/testdata/results-bf16.json` here is the same file with
`env.endpoint` and `env.host` added back.

### Progress lines (for the console)

While running, the tool prints one line per update to stdout:

```
qb:progress {"task":"gsm8k","done":120,"total":1319}
qb:progress {"task":"gsm8k","state":"scored","correct":1147,"total":1319}
qb:progress {"task":"lcb","state":"partial","done":88,"total":175}
```

From 0.2.0, task lines carry `elapsed_s` (seconds since the task started), and
a run line comes every 10 seconds and at each task change:

```
qb:progress {"run":{"task":"ifeval","task_index":2,"tasks":9,"elapsed_s":812,"eta_s":1460,"est_total_s":2300,"fraction":0.36}}
```

`task_index` is 1-based. `fraction` is the share of the run done, weighted by
each task's estimated minutes. `eta_s` uses the running task's own rate once it
is a few percent in, and the estimates of the tasks still to come, corrected by
how the finished tasks compared with theirs. The `qb:estimate` line is
described under `qualitybench estimate`.

The console's QualityBench screen parses these for its per-task progress
view. Any other output is shown as log text.

## Not in this contract yet

- Running on rented GPUs. That comes through `L80 vast` with a
  `benchmark: qualitybench` spec block, and teardown must never wait on the
  user.
- Verifying quality runs (`L80 verify`). It needs a reproducibility rule:
  the same items, and a tolerance band taken from a reference-vs-reference
  noise floor.
