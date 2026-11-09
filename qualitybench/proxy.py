"""A local pass-through in front of the endpoint that every harness talks to.

It does four things the harnesses cannot:
- forces thinking on or off (chat_template_kwargs.enable_thinking), for servers you cannot restart;
  with thinking on it raises max_tokens to a floor and drops `stop` (vLLM matches stop strings
  inside the reasoning too);
- counts chat replies that stopped at the token cap (finish_reason "length") and replies with reasoning,
  so a run can say when a task's scores are cut short (take_stats);
- adds the run's --request-param fields (e.g. reasoning_effort) to every chat request;
- adds the endpoint's bearer key, so the key never reaches a harness's command line;
- cancels the upstream generation when a harness drops its request (closes the upstream connection
  as soon as the client's socket closes), so abandoned generations never pile up on the server.
"""
from __future__ import annotations

import copy
import json
import select
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import http as qhttp

TALLY_CAP = 16 << 20  # bytes of one reply kept for counting; longer replies are not counted
HOP = {"host", "content-length", "accept-encoding", "connection", "transfer-encoding", "authorization"}

_GREEDY = "decoding is greedy with seed 608, so runs compare item by item"
_TASK = "each task sets it"
#: --request-param keys a run may not change, and why.
REFUSED = {**dict.fromkeys(("temperature", "top_p", "top_k", "min_p", "seed", "n", "best_of", "use_beam_search",
                            "presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias"), _GREEDY),
           **dict.fromkeys(("max_tokens", "max_completion_tokens"),
                           "each task sets its own; with --thinking on, raise the floor with --thinking-max-tokens"),
           **dict.fromkeys(("model", "messages", "stream", "stream_options", "stop", "logprobs", "top_logprobs", "tools",
                            "tool_choice", "parallel_tool_calls", "response_format"), _TASK),
           "chat_template_kwargs.enable_thinking": "use --thinking on|off"}


def flat(d: dict, prefix: str = ""):
    """(dotted key, value) for every leaf of a nested dict."""
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def parse_params(pairs: list[str]) -> dict:
    """--request-param KEY=VALUE pairs as one dict: dotted keys nest, values parse as JSON when they can."""
    out: dict = {}
    for pair in pairs:
        k, sep, v = pair.partition("=")
        parts = k.strip().split(".")
        if not sep or not all(parts):
            raise ValueError(f"--request-param {pair!r}: expected KEY=VALUE, e.g. reasoning_effort=high")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"--request-param {pair!r}: {p} is already set to a value, not an object")
        try:
            node[parts[-1]] = json.loads(v)
        except ValueError:
            node[parts[-1]] = v
    for key, _ in flat(out):
        why = REFUSED.get(key) or REFUSED.get(key.split(".")[0])
        if why:
            raise ValueError(f"--request-param {key} is not allowed: {why}")
    return out


def describe(params: dict) -> str:
    return ", ".join(f"{k}={v if isinstance(v, str) else json.dumps(v)}" for k, v in flat(params))


def merge(into: dict, extra: dict) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            merge(into[k], v)
        else:
            into[k] = copy.deepcopy(v)  # the request is edited afterwards; never share the run's params


class Proxy:
    def __init__(self, upstream: str, api_key: str | None = None, thinking: str | None = None,
                 min_max_tokens: int = 0, params: dict | None = None):
        up = upstream.rstrip("/")
        self.upstream = up[:-3] if up.endswith("/v1") else up
        self.api_key, self.thinking, self.min_max_tokens = api_key, thinking, min_max_tokens
        self._lock = threading.Lock()
        self._stats = {"replies": 0, "length": 0, "reasoning": 0}
        self.params = params or {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def take_stats(self) -> dict:
        """Chat replies seen since the last call: total, stopped by the token cap, with reasoning. Resets."""
        with self._lock:
            out, self._stats = self._stats, {"replies": 0, "length": 0, "reasoning": 0}
        return out

    def tally(self, body: bytes) -> None:
        """Count one chat/completions reply (JSON, or the SSE events of a streamed one)."""
        fins, reasoning = [], False
        text = body.decode("utf-8", "replace")
        docs = []
        if text.lstrip().startswith("{"):
            try:
                docs = [json.loads(text)]
            except ValueError:
                return
        else:
            for line in text.splitlines():
                if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                    try:
                        docs.append(json.loads(line[5:]))
                    except ValueError:
                        pass
        for d in docs:
            for ch in d.get("choices") or []:
                m = ch.get("message") or ch.get("delta") or {}
                if m.get("reasoning") or m.get("reasoning_content") or "<think>" in (m.get("content") or ""):
                    reasoning = True
                if ch.get("finish_reason"):
                    fins.append(ch["finish_reason"])
        if not fins:
            return
        with self._lock:
            self._stats["replies"] += 1
            self._stats["length"] += fins[-1] == "length"
            self._stats["reasoning"] += reasoning

    def rewrite(self, path: str, body: bytes) -> bytes:
        if not path.rstrip("/").endswith("/chat/completions") or not (self.thinking or self.params):
            return body
        try:
            d = json.loads(body)
        except ValueError:
            return body
        merge(d, self.params)
        if not self.thinking:
            return json.dumps(d).encode()
        d.setdefault("chat_template_kwargs", {})["enable_thinking"] = self.thinking == "on"
        if self.thinking == "on":
            if self.min_max_tokens:
                d.pop("max_completion_tokens", None)
                d["max_tokens"] = max(d.get("max_tokens") or 0, self.min_max_tokens)
            d.pop("stop", None)
        return json.dumps(d).encode()

    def _handler(self):
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _fwd(self, body: bytes | None):
                conn, _ = qhttp.connect(proxy.upstream, qhttp.TIMEOUT)
                qhttp.track(conn)
                done = threading.Event()

                def watch():  # the client hung up while we wait: drop the upstream request too
                    s = self.connection
                    while not done.is_set():
                        try:
                            r, _, _ = select.select([s], [], [], 1.0)
                            if r and not s.recv(1, 0x2):  # MSG_PEEK; b"" means the client closed
                                qhttp.hard_close(conn)
                                return
                        except (OSError, ValueError):
                            qhttp.hard_close(conn)
                            return

                threading.Thread(target=watch, daemon=True).start()
                try:
                    hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                    if proxy.api_key:
                        hdrs["Authorization"] = f"Bearer {proxy.api_key}"
                    elif self.headers.get("Authorization"):
                        hdrs["Authorization"] = self.headers["Authorization"]
                    conn.request(self.command, self.path, body, hdrs)
                    r = conn.getresponse()
                    self.send_response(r.status)
                    for k, v in r.getheaders():
                        if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                            self.send_header(k, v)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    keep = [] if r.status == 200 and self.path.rstrip("/").endswith("/chat/completions") else None
                    size = 0
                    while chunk := r.read1(65536) if hasattr(r, "read1") else r.read(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if keep is not None:
                            size += len(chunk)
                            if size <= TALLY_CAP:
                                keep.append(chunk)
                            else:
                                keep = None
                    if keep:
                        proxy.tally(b"".join(keep))
                except Exception as e:  # upstream failed or was cancelled
                    try:
                        msg = json.dumps({"error": {"message": f"qualitybench proxy: {e}"}}).encode()
                        self.send_response(502)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(msg)))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(msg)
                    except Exception:
                        pass
                finally:
                    done.set()
                    qhttp.untrack(conn)
                    self.close_connection = True

            def do_GET(self):
                self._fwd(None)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                self._fwd(proxy.rewrite(self.path, body))

            def log_message(self, *args):
                pass

        return H
