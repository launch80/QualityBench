"""A local pass-through in front of the endpoint that every harness talks to.

It does three things the harnesses cannot:
- forces thinking on or off (chat_template_kwargs.enable_thinking), for servers you cannot restart;
  with thinking on it raises max_tokens to a floor and drops `stop` (vLLM matches stop strings
  inside the reasoning too);
- adds the endpoint's bearer key, so the key never reaches a harness's command line;
- cancels the upstream generation when a harness drops its request (closes the upstream connection
  as soon as the client's socket closes), so abandoned generations never pile up on the server.
"""
from __future__ import annotations

import json
import select
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import http as qhttp

HOP = {"host", "content-length", "accept-encoding", "connection", "transfer-encoding", "authorization"}


class Proxy:
    def __init__(self, upstream: str, api_key: str | None = None, thinking: str | None = None,
                 min_max_tokens: int = 0):
        up = upstream.rstrip("/")
        self.upstream = up[:-3] if up.endswith("/v1") else up
        self.api_key, self.thinking, self.min_max_tokens = api_key, thinking, min_max_tokens
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

    def rewrite(self, path: str, body: bytes) -> bytes:
        if not path.rstrip("/").endswith("/chat/completions") or not self.thinking:
            return body
        try:
            d = json.loads(body)
        except ValueError:
            return body
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
                    while chunk := r.read1(65536) if hasattr(r, "read1") else r.read(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
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
