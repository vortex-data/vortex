# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""A tiny local stand-in for the Hugging Face Hub used by the ``local_hub`` fixture.

Serves files from a directory the way the Hub's ``resolve`` endpoint does, including
HTTP Range support and optional bearer-token auth, so ``hf://`` integration tests need
no network access or Hugging Face account.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlsplit

import pyarrow as pa

import vortex as vx


@dataclass
class RequestLog:
    method: str
    path: str
    range: str | None
    status: int
    bytes_sent: int


@dataclass
class HubState:
    """Mutable per-server state: the served directory, auth requirement, and request log."""

    root: Path
    required_token: str | None = None
    requests: list[RequestLog] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def bytes_served(self) -> int:
        with self.lock:
            return sum(r.bytes_sent for r in self.requests)

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()

    def publish(self, repo_path: str, table: pa.Table) -> Path:
        """Write `table` as a Vortex file into the fake hub under a repository resolve path."""
        target = self.root / repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        vx.io.write(table, str(target))
        return target


class HubRequestHandler(BaseHTTPRequestHandler):
    """Serves files from a directory with HTTP Range support, like the Hub's resolve endpoint."""

    protocol_version = "HTTP/1.1"
    state: ClassVar[HubState]  # assigned on the handler subclass per server

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def _record(self, status: int, bytes_sent: int) -> None:
        with self.state.lock:
            self.state.requests.append(
                RequestLog(self.command, self.path, self.headers.get("Range"), status, bytes_sent)
            )

    def _error(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()
        self._record(status, 0)

    def _serve(self, send_body: bool) -> None:
        if self.state.required_token is not None:
            if self.headers.get("Authorization") != f"Bearer {self.state.required_token}":
                return self._error(401)

        relpath = unquote(urlsplit(self.path).path).lstrip("/")
        target = (self.state.root / relpath).resolve()
        if not target.is_relative_to(self.state.root.resolve()) or not target.is_file():
            return self._error(404)

        data = target.read_bytes()
        size = len(data)

        range_header = self.headers.get("Range")
        if range_header is None:
            start, end, status = 0, size - 1, 200
        else:
            spec = range_header.removeprefix("bytes=")
            if spec.startswith("-"):
                start, end = max(size - int(spec[1:]), 0), size - 1
            else:
                start_s, _, end_s = spec.partition("-")
                start, end = int(start_s), int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end:
                return self._error(416)
            status = 206

        body = data[start : end + 1]
        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if send_body:
            self.wfile.write(body)
        self._record(status, len(body) if send_body else 0)


def serve(root: Path) -> tuple[ThreadingHTTPServer, HubState]:
    """Start a fake hub server for `root` on a random port, returning it with its state."""
    state = HubState(root=root)
    handler = type("Handler", (HubRequestHandler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def sample_table(num_rows: int = 1000, seed: int = 0) -> pa.Table:
    """A small deterministic table of incompressible-ish data for read tests."""
    rng = random.Random(seed)
    return pa.table(
        {
            "x": [rng.randrange(1 << 40) for _ in range(num_rows)],
            "s": [rng.randbytes(8).hex() for _ in range(num_rows)],
        }
    )
