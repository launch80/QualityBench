#!/usr/bin/env python3
"""evalplus.codegen with a sane request lifecycle; same CLI as the console script (runs in the lmeval env).

Upstream evalplus 0.3.1 (gen/util/openai_request.py) arms a 100 s SIGALRM per request and retries forever.
A thinking-on answer routinely takes longer, so every problem was killed and resent while the abandoned
generation kept running on the server. Here each request gets one hour and at most three tries, and a reply
with no content (thinking cut off by max_tokens) becomes "", scored as a failure instead of crashing the
shard. Greedy outputs are unchanged.
"""
import sys, time
import openai
import evalplus.gen.util.openai_request as req

_OpenAI = openai.OpenAI
openai.OpenAI = lambda *a, **k: _OpenAI(*a, **{"timeout": 3600, "max_retries": 0, **k})


def _no_none(resp):
    for ch in getattr(resp, "choices", []) or []:
        if getattr(ch, "message", None) is not None and ch.message.content is None:
            ch.message.content = ""
    return resp


def make_auto_request(*args, **kwargs):
    for attempt in range(1, 4):
        try:
            return _no_none(req.make_request(*args, **kwargs))
        except Exception as e:  # connection reset, 5xx, timeout: the client closed the connection
            print(f"request failed (attempt {attempt}/3): {e!r}", flush=True)
            if attempt == 3:
                raise
            time.sleep(10)


req.make_auto_request = make_auto_request

from evalplus.codegen import main  # noqa: E402  (import after the patch)

sys.exit(main())
