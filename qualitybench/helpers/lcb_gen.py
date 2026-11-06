#!/usr/bin/env python3
"""LiveCodeBench v6 generation with lcb_runner's own prompt + extraction (runs in the LCB image or venv).

  lcb_gen.py --base http://host:port --model <id> --out <dir> --ids ids.txt [--workers 16] [--max-tokens 4096]

Prompts are lcb_runner's LMStyle.OpenAIChat messages, greedy, one sample; code is pulled out with lcb_runner's
extract_code. Replies are appended to <out>/raw.jsonl as they arrive, so a rerun only asks for the missing
problems. When all are in, writes <out>/custom.json for `lcb_runner.runner.custom_evaluator`. Scoring (which
executes the code) is a separate step with no network.
"""
import argparse, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts.code_generation import format_prompt_generation
from lcb_runner.utils.extraction_utils import extract_code

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--ids", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
want = set(open(a.ids).read().split())
probs = [p for p in load_code_generation_dataset("v6") if str(p.question_id) in want]
assert len(probs) == len(want), f"{len(probs)} problems found, {len(want)} ids listed"
raw = os.path.join(a.out, "raw.jsonl")
done = {}
if os.path.exists(raw):
    for line in open(raw):
        try:
            d = json.loads(line)
            done[d["question_id"]] = d
        except ValueError:
            pass  # a line cut short by an interrupt
lock = threading.Lock()
todo = [p for p in probs if str(p.question_id) not in done]
print(f"lcb: {len(done)} of {len(probs)} already generated", flush=True)


def gen(p):
    body = {"model": a.model, "messages": format_prompt_generation(p, LMStyle.OpenAIChat),
            "temperature": 0, "max_tokens": a.max_tokens}
    for attempt in range(1, 4):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                a.base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                {"Content-Type": "application/json"}), timeout=3600))
            break
        except Exception as ex:
            print(f"request failed (attempt {attempt}/3): {ex!r}", flush=True)
            if attempt == 3:
                raise
            time.sleep(10)
    text = r["choices"][0]["message"].get("content") or ""
    d = {"question_id": str(p.question_id), "raw": text, "finish_reason": r["choices"][0].get("finish_reason"),
         "code_list": [extract_code(text, LMStyle.OpenAIChat)]}
    with lock:
        with open(raw, "a") as f:
            f.write(json.dumps(d) + "\n")
        done[d["question_id"]] = d
        print(f"lcb: generated {len(done)}/{len(probs)}", flush=True)
    return d


t0 = time.time()
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(gen, todo))
res = [done[str(p.question_id)] for p in probs]
json.dump([{"question_id": r["question_id"], "code_list": r["code_list"]} for r in res],
          open(os.path.join(a.out, "custom.json"), "w"))
print(f"lcb: {len(res)} generated in {time.time() - t0:.0f}s, {sum(1 for r in res if not r['code_list'][0])} without a code block")
