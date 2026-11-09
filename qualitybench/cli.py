"""Command line: `qualitybench run | estimate | compare | report | setup | tasks`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .suite import SUITE


class Parser(argparse.ArgumentParser):
    def error(self, message):  # usage errors exit 2, as the contract says
        self.print_usage(sys.stderr)
        sys.stderr.write(f"qualitybench: error: {message}\n")
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = Parser(prog="qualitybench", description="Repeatable quality baseline and quality-loss bench for "
                                                 "OpenAI-compatible endpoints.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", parser_class=Parser)

    r = sub.add_parser("run", help="run the suite against one endpoint and write a results file + HTML report")
    r.add_argument("--endpoint", required=True, help="base URL of the OpenAI-compatible server, e.g. http://127.0.0.1:8000/v1")
    r.add_argument("--model", required=True, help="model id as the server lists it")
    r.add_argument("--out", required=True, help="results file (.json); the report goes next to it as .html and "
                                                "per-item data in <name>.qb/")
    r.add_argument("--quick", action="store_true", help="a fixed tenth of each task's items (a smoke run)")
    r.add_argument("--tasks", help="comma list of task ids (default: every default task); see `qualitybench tasks`")
    r.add_argument("--thinking", choices=["on", "off"], help="force thinking on or off through the local proxy "
                                                             "(default: the server's own default)")
    r.add_argument("--thinking-max-tokens", type=int, default=32768,
                   help="max_tokens floor with --thinking on (32768; thinking models can spend 10-30k tokens on hard code)")
    r.add_argument("--reference", help="results file of a reference run: also compare against it (KLD, paired "
                                       "deltas, thresholds); --out then holds the comparison")
    r.add_argument("--thresholds", help="thresholds markdown table (default: the built-in bars)")
    r.add_argument("--deadline", help="RFC 3339 time or a duration such as 6h: start no work after it, mark cut tasks partial")
    r.add_argument("--note", action="append", default=[], metavar="KEY=VALUE",
                   help="precision=, engine= and gpu= go in env; anything else becomes a caveat")
    r.add_argument("--title", help="report title")
    r.add_argument("--concurrency", type=int, default=16, help="requests in flight per harness (16)")
    r.add_argument("--sandbox", choices=["docker", "native"], default="docker",
                   help="where model-written code is executed: a network-less container (default; docker, or podman "
                        "with QUALITYBENCH_CONTAINER=podman) or directly on this machine (only on a disposable box)")
    r.add_argument("--tokenizer", help="HF repo or local dir of the model's tokenizer, for logprobs and needle "
                                       "(default: the model id)")
    r.add_argument("--eval-set", help="a prebuilt eval-set.json to score instead of building one for the tokenizer")
    r.add_argument("--arm", help="key of this run in the comparison (default: from the model name)")
    r.add_argument("--arm-label", help="label of this run in the comparison (default: the model id)")
    r.add_argument("--yes", "-y", action="store_true", help="start without asking about the estimated time")
    r.add_argument("--max-minutes", type=float, help="do not start (exit 3) if the run could take longer than this")
    r.add_argument("--no-probe", action="store_true", help="skip the short server speed test; estimate from history "
                                                          "or the reference run")
    r.add_argument("--skip-setup", action="store_true", help=argparse.SUPPRESS)

    e = sub.add_parser("estimate", help="say how long a run would take, without running it")
    e.add_argument("--endpoint", help="base URL of the server to measure (omit to use history or the reference run)")
    e.add_argument("--model", required=True, help="model id as the server lists it")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--tasks", help="comma list of task ids (default: every default task)")
    e.add_argument("--thinking", choices=["on", "off"])
    e.add_argument("--sandbox", choices=["docker", "native"], default="docker")
    e.add_argument("--no-probe", action="store_true", help="don't measure the server")
    e.add_argument("--json", action="store_true", help="print one JSON object")

    c = sub.add_parser("compare", help="compare a test run against a reference run")
    c.add_argument("ref", help="reference results file")
    c.add_argument("test", help="test results file")
    c.add_argument("--thresholds", help="thresholds markdown table (default: the built-in bars)")
    c.add_argument("--out", help="where to write the comparison (default: <test>-vs-<ref>.json)")
    c.add_argument("--arm", help="key of the test run in the comparison (default: from its model name)")
    c.add_argument("--arm-label", help="label of the test run (default: its model id)")

    rp = sub.add_parser("report", help="render the HTML report for a results file")
    rp.add_argument("results")
    rp.add_argument("--out", help="default: the results path with .html")

    s = sub.add_parser("setup", help="install the pinned harness environments and fetch datasets ahead of a run")
    s.add_argument("--tasks", help="only what these tasks need")
    s.add_argument("--sandbox", choices=["docker", "native"], default="docker")

    t = sub.add_parser("tasks", help="list the suite")
    t.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if not a.cmd:
        build_parser().print_help()
        return 2
    if a.cmd == "tasks":
        rows = [{"id": t.id, "label": t.label, "items": t.items, "measures": t.how, "default": t.default} for t in SUITE]
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                print(f"{r['id']:<12} {r['items']:>5}  {'default ' if r['default'] else 'optional'}  {r['label']}: {r['measures']}")
        return 0
    if a.cmd == "estimate":
        import os

        from . import estimate as qest
        from .suite import parse_tasks
        try:
            tasks = parse_tasks(a.tasks)
        except ValueError as ex:
            print(f"qualitybench: error: {ex}", file=sys.stderr)
            return 2
        probe, warn = None, None
        if a.endpoint and not a.no_probe:
            probe, warn = qest.measure(a.endpoint, a.model, os.environ.get("QUALITYBENCH_API_KEY"))
        doc = qest.estimate(tasks, a.quick, a.thinking, a.model, probe, a.sandbox)
        if warn:
            doc["warning"] = warn
        print(json.dumps(doc) if a.json else "\n".join(qest.summary(doc)))
        return 0
    if a.cmd == "setup":
        from . import envs
        from .suite import parse_tasks
        try:
            envs.setup(parse_tasks(a.tasks), a.sandbox)
        except ValueError as e:
            print(f"qualitybench: error: {e}", file=sys.stderr)
            return 2
        print("setup done")
        return 0
    if a.cmd == "report":
        from . import report
        doc = json.loads(Path(a.results).read_text())
        print(report.write(doc, Path(a.results), Path(a.out) if a.out else None))
        return 0
    if a.cmd == "compare":
        from . import compare, report, results
        doc, warn = compare.build(Path(a.ref), Path(a.test), a.thresholds, a.arm, a.arm_label)
        for w in warn:
            print("warning: " + w, file=sys.stderr)
        out = Path(a.out) if a.out else Path(a.test).with_name(f"{Path(a.test).stem}-vs-{Path(a.ref).stem}.json")
        for w in results.write(doc, out):
            print("warning: " + w, file=sys.stderr)
        html = report.write(doc, out)
        print("\n".join(compare.summary_lines(doc, doc["gate_arm"])))
        print(f"wrote {out} and {html}")
        return 0
    if a.cmd == "run":
        from .runner import run
        from .suite import parse_tasks
        try:
            parse_tasks(a.tasks)
            from .context import parse_deadline
            parse_deadline(a.deadline)
        except ValueError as e:
            print(f"qualitybench: error: {e}", file=sys.stderr)
            return 2
        if a.concurrency < 1:
            print("qualitybench: error: --concurrency must be at least 1", file=sys.stderr)
            return 2
        try:
            return run(a)
        except KeyboardInterrupt:
            return 130
    return 2


if __name__ == "__main__":
    sys.exit(main())
