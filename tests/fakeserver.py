"""A tiny OpenAI-compatible server for tests: deterministic chat, prompt_logprobs and a slow route."""
import json
import math
import re
import select
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Fake:
    def __init__(self, noise: float = 0.0, reasoning: bool = False, inline: bool = False, finish: str = "stop"):
        self.noise = noise          # shifts logprobs, so a "quantized" server differs from the reference
        self.reasoning = reasoning
        self.inline = inline        # reasoning inside content as <think>…</think> (no reasoning parser)
        self.finish = finish        # finish_reason of chat replies ("length": cut at max_tokens)
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
                self._json({"object": "list", "data": [{"id": "fake", "object": "model"}]})

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
                    m = re.search(r"magic number for the blue lighthouse is (\d+)", text)
                    content = m.group(1) if m else ("OK" if "single word OK" in text else f"answer to: {text[:20]}")
                    if m and len(text) > 100_000:
                        return self._json({"error": {"message": "context too long"}}, 400)
                    msg = {"role": "assistant", "content": content}
                    if fake.reasoning:
                        msg["reasoning_content"] = "thinking..."
                    if fake.inline:
                        msg["content"] = "<think>thinking...</think>\n\n" + content
                    toks = content.split(" ")
                    lp = [{"token": t, "logprob": -0.1, "top_logprobs": [{"token": t, "logprob": -0.1},
                                                                          {"token": "x", "logprob": -2.5}]} for t in toks]
                    return self._json({"choices": [{"message": msg, "finish_reason": fake.finish, "logprobs": {"content": lp}}],
                                       "usage": {"prompt_tokens": len(text) // 4}})
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
