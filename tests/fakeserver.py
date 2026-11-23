"""A tiny OpenAI-compatible server for tests: deterministic chat, prompt_logprobs and a slow route."""
import json
import math
import re
import select
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Fake:
    def __init__(self, noise: float = 0.0, reasoning: bool = False, inline: bool = False, finish: str = "stop",
                  mrcr_truncate: bool = False, usage=None):
        self.noise = noise          # shifts logprobs, so a "quantized" server differs from the reference
        self.reasoning = reasoning
        self.inline = inline        # reasoning inside content as <think>…</think> (no reasoning parser)
        self.finish = finish        # finish_reason of chat replies ("length": cut at max_tokens)
        self.mrcr_truncate = mrcr_truncate  # answer MRCR asks with half the piece, like a degraded model
        self.usage = usage          # replaces the usage of chat replies (e.g. "n/a": a server that sends junk)
        self.requests = []
        self.aborted = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def _handler(fake):
        class H(BaseHTTPRequestHandler):
            def _json(self, obj, status=200):
                b = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                fake.requests.append(("GET", self.path, None, dict(self.headers)))
                self._json({"object": "list", "data": [{"id": "fake", "object": "model", "max_model_len": 131072}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(("POST", self.path, body, dict(self.headers)))
                if body.get("model") == "slow":
                    for _ in range(300):  # wait for the client to go away, like a long generation
                        r, _, _ = select.select([self.connection], [], [], 0.05)
                        if r and not self.connection.recv(1, 0x2):
                            fake.aborted.set()
                            return
                    return self._json({"choices": [{"message": {"content": "late"}}]})
                if self.path.endswith("/chat/completions"):
                    text = body["messages"][-1]["content"]
                    if sum(len(x.get("content") or "") for x in body["messages"]) > 100_000:
                        return self._json({"error": {"message": "context too long"}}, 400)
                    m = re.search(r"magic number for the blue lighthouse is (\d+)", text)
                    mr = re.search(r"Prepend (\S+) to the (\d+)\w\w \(1 indexed\)", text)
                    if mr:  # an MRCR ask: the k-th assistant turn, with the prefix asked for
                        asst = [x["content"] for x in body["messages"] if x["role"] == "assistant"]
                        k = int(mr.group(2)) - 1
                        piece = asst[k] if k < len(asst) else "nothing"
                        content = f"{mr.group(1)} {piece[: len(piece) // 2] if fake.mrcr_truncate else piece}"
                    else:
                        content = m.group(1) if m else ("OK" if "single word OK" in text else f"answer to: {text[:20]}")
                    msg = {"role": "assistant", "content": content}
                    if fake.reasoning:
                        msg["reasoning_content"] = "thinking..."
                    if fake.inline:
                        msg["content"] = "<think>thinking...</think>\n\n" + content
                    toks = content.split(" ")
                    lp = [{"token": t, "logprob": -0.1, "top_logprobs": [{"token": t, "logprob": -0.1},
                                                                          {"token": "x", "logprob": -2.5}]} for t in toks]
                    usage = {"prompt_tokens": len(text) // 4}
                    if mr:  # as vLLM counts: completion_tokens includes the thinking, reasoning_tokens says how much
                        usage["completion_tokens"] = len(content) // 4 + (7 if fake.reasoning else 0)
                        if fake.reasoning:
                            usage["completion_tokens_details"] = {"reasoning_tokens": 7}
                    else:
                        usage["completion_tokens"] = len(toks)
                        if fake.reasoning:  # as vLLM with a reasoning parser: the reasoning's share of completion_tokens
                            usage["completion_tokens_details"] = {"reasoning_tokens": len(msg["reasoning_content"].split())}
                    return self._json({"choices": [{"message": msg, "finish_reason": fake.finish, "logprobs": {"content": lp}}],
                                       "usage": usage if fake.usage is None else fake.usage})
                if self.path.endswith("/completions"):
                    ids = body["prompt"]
                    pl = [None]
                    for p in range(1, len(ids)):
                        d = {}
                        for r in range(1, 6):
                            tok = (ids[p] + r - 1) % 1000
                            d[str(tok)] = {"logprob": math.log(0.5 ** r) - fake.noise * ((p * r) % 3), "rank": r}
                        pl.append(d)
                    return self._json({"choices": [{"text": "", "prompt_logprobs": pl}]})
                self._json({"error": "not found"}, 404)

            def log_message(self, *a):
                pass

        return H
