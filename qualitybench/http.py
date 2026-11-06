"""HTTP to the endpoint under test, with the request lifecycle the contract promises.

- Every request gets a long timeout (an hour by default) and at most 3 attempts.
- A request we give up on is cancelled by closing its connection: vLLM, SGLang and llama.cpp
  abort a generation whose client went away. Nothing is resent while the first copy may still run,
  because the only resend follows a closed connection.
- `cancel_all()` (deadline, Ctrl-C) closes every open connection, so nothing keeps generating
  on the server after the tool stops.
"""
from __future__ import annotations

import http.client
import json
import socket
import ssl
import threading
import time
import urllib.parse

TIMEOUT = 3600
ATTEMPTS = 3
RETRY_PAUSE = 10

CANCEL = threading.Event()
_open: set[http.client.HTTPConnection] = set()
_open_lock = threading.Lock()


class Cancelled(Exception):
    """The run was stopped (deadline or interrupt) while this request was pending."""


class HTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def connect(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    u = urllib.parse.urlsplit(url)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    path = u.path or "/"
    if u.query:
        path += "?" + u.query
    return conn, path


def track(conn) -> None:
    with _open_lock:
        _open.add(conn)


def untrack(conn) -> None:
    with _open_lock:
        _open.discard(conn)
    try:
        conn.close()
    except Exception:
        pass


def hard_close(conn) -> None:
    """Close a connection from another thread so a blocked read fails at once."""
    sock = getattr(conn, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def cancel_all() -> None:
    CANCEL.set()
    with _open_lock:
        conns = list(_open)
    for c in conns:
        hard_close(c)


def request(method: str, url: str, body: dict | None = None, headers: dict | None = None,
            timeout: float = TIMEOUT) -> tuple[int, bytes]:
    if CANCEL.is_set():
        raise Cancelled()
    conn, path = connect(url, timeout)
    track(conn)
    try:
        data = json.dumps(body).encode() if body is not None else None
        h = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, data, h)
        r = conn.getresponse()
        return r.status, r.read()
    except (OSError, http.client.HTTPException):
        if CANCEL.is_set():
            raise Cancelled()
        raise
    finally:
        untrack(conn)


class Client:
    """JSON client for one OpenAI-compatible base URL (with or without the trailing /v1)."""

    def __init__(self, base: str, api_key: str | None = None, timeout: float = TIMEOUT, attempts: int = ATTEMPTS):
        base = base.rstrip("/")
        self.base = base[:-3] if base.endswith("/v1") else base
        self.api_key = api_key
        self.timeout = timeout
        self.attempts = attempts

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def get(self, path: str, timeout: float = 60) -> dict:
        status, data = request("GET", self.base + path, headers=self._headers(), timeout=timeout)
        if status != 200:
            raise HTTPError(status, data.decode(errors="replace"))
        return json.loads(data)

    def post(self, path: str, body: dict, timeout: float | None = None, log=None) -> dict:
        last = None
        for attempt in range(1, self.attempts + 1):
            try:
                status, data = request("POST", self.base + path, body, self._headers(), timeout or self.timeout)
                if status == 200:
                    return json.loads(data)
                err = HTTPError(status, data.decode(errors="replace"))
                if 400 <= status < 500 and status not in (408, 429):
                    raise err  # the request itself is wrong: resending cannot help
                last = err
            except Cancelled:
                raise
            except HTTPError:
                raise
            except (OSError, http.client.HTTPException, ValueError) as e:
                last = e  # the connection is closed by now, so the server has dropped the request
            if log:
                log(f"request failed (attempt {attempt}/{self.attempts}): {last}")
            if attempt < self.attempts:
                if CANCEL.wait(RETRY_PAUSE):
                    raise Cancelled()
        raise last

    def chat(self, model: str, messages: list, log=None, **kw) -> dict:
        return self.post("/v1/chat/completions", {"model": model, "messages": messages, **kw}, log=log)


def wait_ready(client: Client, attempts: int = 30, pause: float = 10) -> dict:
    """GET /v1/models until it answers; returns the listing."""
    last = None
    for _ in range(attempts):
        try:
            return client.get("/v1/models")
        except (OSError, HTTPError, ValueError, http.client.HTTPException) as e:
            last = e
            if CANCEL.wait(pause):
                raise Cancelled()
    raise RuntimeError(f"endpoint did not answer /v1/models: {last}")
