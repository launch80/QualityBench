# QualityBench

> **How to:** watch the [2-minute intro video](https://youtu.be/m7z_u1MZNiU) or read the [launch post](https://www.launch80.com/blog/qualitybench-launch) for a walkthrough of a run, the report and compare, with and without the L80 CLI.

QualityBench measures how good a model's answers are, and how much quality a quantized build or a different
engine loses, against any OpenAI-compatible endpoint. It is the quality sibling of
[BetterBench](https://github.com/GGZ14/BetterBench), which measures speed.

It runs a fixed, pinned suite: the same items, the same token windows, greedy decoding with seed 608. That
way two runs compare item by item, and the comparison gives paired deltas with intervals, not two headline
numbers side by side.

![QualityBench: a quick run, the HTML report, and compare](docs/media/qualitybench.gif)

| Task | Items | Scored as |
|---|---|---|
| `gsm8k` | 1319 | last number in the reply, 0-shot chat (lm-eval) |
| `ifeval` | 541 | prompt-level strict (lm-eval) |
| `mmlu_pro` | 1000 | fixed subset over 14 subjects, exact letter (lm-eval) |
| `humaneval` | 164 | HumanEval+ base + plus tests, sandboxed (evalplus) |
| `mbpp` | 378 | MBPP+ base + plus tests, sandboxed (evalplus) |
| `bfcl` | 400 | 4 categories, function calling through the server's tool parser (BFCL) |
| `lcb` | 175 | LiveCodeBench v6, fixed slice, sandboxed |
| `logprobs` | 592 windows | teacher-forced top-64 over 2048-token windows: perplexity, and KLD / top-1 agreement in compare |
| `needle` | 20 | a 7-digit needle at 10 depths in 32k and 128k of C4 text |
| `spec_parity` | 32 | greedy text, 256 tokens (optional; for speculative decoding on vs off) |

The suite comes from the Qwopus3.8-27B BF16-vs-INT4 study (November 2026), which is also its regression test:
re-scoring that study's per-item files with this package reproduces its published numbers exactly.

## Install

```bash
uv tool install git+https://github.com/launch80/QualityBench@v0.1.0
qualitybench setup            # one time: harness environments, the LCB image, datasets
```

You need Python 3.10+, [uv](https://docs.astral.sh/uv/) and git. Docker is needed for the code tasks, which run
model-written code in containers with no network. The eval harnesses pin incompatible dependencies, so
`setup` installs each into its own locked environment under `~/.cache/qualitybench` (set `QUALITYBENCH_HOME`
to move it). A run also installs what it needs on first use.

If you use the [L80 CLI](https://github.com/launch80/L80-CLI), `L80 qualitybench` runs this tool and
publishes the results as a launch80 page.

## Run

```bash
qualitybench run --endpoint http://127.0.0.1:8000/v1 --model my-model --out results/bf16.json
```

This writes three things:
- `results/bf16.json`: the results file, a summary under 256 KB (the `bench.qualitybench.v1` format launch80 renders).
- `results/bf16.html`: a self-contained report with charts. It works offline.
- `results/bf16.qb/`: per-item records, raw harness output, logprob arrays and `run.log`. `compare` needs this
  folder, so keep it next to the results file. It is never part of a published payload.

Useful options:

| Option | What it does |
|---|---|
| `--quick` | a fixed tenth of each task's items. A smoke run; quick runs compare only with quick runs |
| `--tasks gsm8k,ifeval` | a subset of the suite (`qualitybench tasks` lists ids) |
| `--thinking on\|off` | forces thinking through a local proxy and checks that reasoning is present or absent |
| `--deadline 6h` | starts no work after the deadline; cut tasks are listed in `partials`, not dropped |
| `--reference ref.json` | also compares this run with a reference run; `--out` then holds the comparison |
| `--note precision=INT4 --note engine="vLLM 0.30" --note gpu="RTX PRO 6000"` | labels for the report |
| `--tokenizer org/repo` | the tokenizer for logprobs and needle, if the served name is not a Hugging Face repo |
| `--sandbox native` | runs model code directly on this machine. Use it only on a disposable box without docker |

The endpoint's bearer key goes in `QUALITYBENCH_API_KEY`, never on the command line.

### How long it will take

These runs take a while: about 2h40 for the full suite on a 27B INT4 model on one Arc Pro B65. Before the
first task, `run` measures the server for a few seconds and prints an estimate with a range:

```
Estimated time: about 38 min (17 min–1h16), measured on this server just now
  GSM8K 6 min · IFEval 3 min · MMLU-Pro 9 min · HumanEval+ 4 min · ...
Start? [y/N]
```

On a terminal it asks before starting. `--yes` skips the question, and `QUALITYBENCH_CONFIRM=never` (or
`over:30m`, to ask only for long runs) changes the default. Scripts without a terminal are never asked.
`--max-minutes 60` refuses to start (exit 3) when the run could take longer than that.

To estimate without running anything:

```bash
qualitybench estimate --endpoint http://127.0.0.1:8000/v1 --model my-model --quick [--json]
```

The first estimate for a model is rough: the probe measures the server's speed, not how much this model
writes. After one run, the estimate uses that run's real times, kept in `timings.jsonl` in the QualityBench
home (`~/.cache/qualitybench`). That file stays on this machine.

To resume an interrupted run, run the same command again. Finished tasks, evalplus shards, LCB problems and
logprob chunks are kept and skipped.

### Server requirements

- **logprobs** needs `prompt_logprobs` on `/v1/completions`, which vLLM provides. Start the server with
  `--max-logprobs 64` and speculative decoding off (prompt logprobs are not exact under MTP).
- **bfcl** scores the server's own tool-call parser, so enable it (for vLLM, `--enable-auto-tool-choice --tool-call-parser ...`).
- **needle** needs a 128k context for its long half. A shorter server skips that length and says so in `partials`.
- **--thinking on** needs a reasoning parser (vLLM `--reasoning-parser`) and a chat template that honours
  `enable_thinking`.

## Compare

```bash
qualitybench compare results/bf16.json results/int4.json --thresholds my-thresholds.md
```

This prints KLD, top-1 agreement, the perplexity change and a paired accuracy change per task (exact McNemar
p plus a paired-bootstrap 95% CI), then a verdict per threshold. It writes `int4-vs-bf16.json` and `.html`:
the reference's page with the test run drawn as an arm. Without `--thresholds`, compare uses the
[default bars](qualitybench/data/thresholds-default.md), which are the study's pre-registered ones. Each
threshold file's sha256 is recorded in the output.

Both runs must use the same tier and tokenizer. Compare refuses logprobs scored on different eval sets, and
it skips any task whose two runs answered different items.

## How it keeps runs honest

- **Fixed inputs.** Task subsets are pinned files in `qualitybench/data`. The quick tier is every tenth item
  of each list, not a sample. The logprob windows are built once per tokenizer from public corpora
  (wikitext-2, C4 validation shard 1, codeparrot-clean-valid) and identified by a sha256 of their token ids.
- **Request lifecycle.** Each request gets a one-hour timeout and at most 3 attempts. An abandoned request is
  cancelled by closing its connection, including through the proxy, so the server stops generating it.
  Nothing is resent while an earlier copy may still be running. evalplus's own 100-second kill-and-retry loop
  is patched out. Behind a thinking proxy, that loop once stalled a run for 3.5 hours.
- **Warmup.** The first request is a warmup plus preflight and is never scored.
- **Sandbox.** HumanEval+, MBPP+ and LiveCodeBench code runs in containers with `--network none`, capped
  memory and pids, and no capabilities.
- **Stats.** Accuracy comes with a 95% Wilson interval. Paired comparisons use an exact McNemar test and a
  10,000-resample paired bootstrap with seed 608.

## Progress output

While it runs, the tool prints one machine-readable line per update. [L80](https://github.com/launch80/L80-CLI)'s
console reads these lines; any other line is log text.

```
qb:estimate {"source":"probe","total_min":38.2,"range_min":[17.2,76.4],"tasks":[...],...}
qb:progress {"task":"gsm8k","done":120,"total":1319,"elapsed_s":95}
qb:progress {"task":"gsm8k","state":"scored","correct":1147,"total":1319,"elapsed_s":412}
qb:progress {"run":{"task":"ifeval","task_index":2,"tasks":9,"elapsed_s":812,"eta_s":1460,"est_total_s":2300,"fraction":0.36}}
```

The `run` line comes every 10 seconds and at each task change. L80 draws it as an overall bar with the time
left.

[docs/contract.md](docs/contract.md) is the full contract between QualityBench and L80: flags, exit codes,
outputs and the results format.

## Development

```bash
uv venv && uv pip install -e '.[dev]' && .venv/bin/pytest
```

The tests use no network and no GPU. A fake OpenAI server covers the run loop, the proxy and cancellation,
and the study's per-item files are the regression fixtures.

## License

Apache-2.0. The pinned harnesses (lm-eval, evalplus, BFCL, LiveCodeBench) and datasets keep their own licenses.

## Community

Questions, results, or a model you want tested? Join the Launch80 Discord: https://discord.gg/launch80

If this work is useful to you, please support the server: https://launch80.com/supporter
