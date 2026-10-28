# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Read-only, loopback S3 subset for comparing real object-store clients.

Serves path-style /bucket/key requests from --root/key. Authentication is ignored.
Latency is charged per request, including HEAD and LIST. A shared bandwidth
budget and request semaphore model a client link and bounded in-flight requests.
This models transport costs, not S3's availability, TLS, retries, or tail latency.
"""

import argparse
import json
import mmap
import socket
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, unquote, urlsplit


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256

    def __init__(self, args: argparse.Namespace):
        super().__init__(("127.0.0.1", args.port), Handler)
        self.root = args.root.resolve()
        self.latency = args.latency_ms / 1000
        self.bandwidth = args.mib_per_second * 1024 * 1024
        self.slots = threading.Semaphore(args.concurrency)
        self.lock = threading.Lock()
        self.next_byte = 0.0
        self.trace = args.trace.open("w")
        self.files = {}

    def file_range(self, path: Path, start: int, end: int) -> memoryview:
        # Benchmark objects are immutable. Stream mapped pages instead of copying an entire
        # large response before its headers, which would add size-dependent first-byte delay.
        with self.lock:
            if path not in self.files:
                with path.open("rb") as source:
                    self.files[path] = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            return memoryview(self.files[path])[start:end]

    def transmit(self, output: BinaryIO, data: bytes | memoryview):
        # Small quanta allow concurrent responses to share the same link.
        for offset in range(0, len(data), 65536):
            chunk = data[offset : offset + 65536]
            with self.lock:
                # One quantum of credit absorbs scheduler oversleep. Without it, a single
                # response runs below the configured rate while concurrent responses do not.
                self.next_byte = (
                    max(time.monotonic() - 65536 / self.bandwidth, self.next_byte) + len(chunk) / self.bandwidth
                )
                ready = self.next_byte
            time.sleep(max(0, ready - time.monotonic()))
            output.write(chunk)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: Server

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *_args: object):
        pass

    def do_HEAD(self):
        self.serve(head=True)

    def do_GET(self):
        self.serve(head=False)

    def send_response(self, code: int, message: str | None = None):
        self.trace_state["status"] = code
        super().send_response(code, message)

    def serve(self, head: bool):
        self.trace_state = dict(
            method=self.command,
            path=self.path,
            range=self.headers.get("Range"),
            start=0,
            bytes=0,
            arrived=time.monotonic(),
        )
        try:
            self.serve_response(head)
        finally:
            self.trace_state["finished"] = time.monotonic()
            with self.server.lock:
                self.server.trace.write(json.dumps(self.trace_state) + "\n")
                self.server.trace.flush()

    def serve_response(self, head: bool):
        with self.server.slots:
            self.trace_state["admitted"] = time.monotonic()
            time.sleep(self.server.latency)
            parsed = urlsplit(self.path)
            parts = unquote(parsed.path).lstrip("/").split("/", 1)
            key = parts[1] if len(parts) > 1 else ""
            self.trace_state["key"] = key
            path = (self.server.root / key).resolve()
            if not path.is_relative_to(self.server.root):
                self.send_error(403)
                return
            query = parse_qs(parsed.query)
            start = 0
            if "list-type" in query or "prefix" in query:
                prefix = query.get("prefix", [""])[0]
                root = ET.Element("ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
                ET.SubElement(root, "Name").text = parts[0]
                ET.SubElement(root, "Prefix").text = prefix
                ET.SubElement(root, "IsTruncated").text = "false"
                count = 0
                for file in sorted(self.server.root.rglob("*")):
                    relative = file.relative_to(self.server.root).as_posix()
                    if file.is_file() and relative.startswith(prefix):
                        entry = ET.SubElement(root, "Contents")
                        for name, value in (
                            ("Key", relative),
                            ("Size", str(file.stat().st_size)),
                            ("LastModified", "2026-01-01T00:00:00.000Z"),
                            ("ETag", '"simulated"'),
                        ):
                            ET.SubElement(entry, name).text = value
                        count += 1
                ET.SubElement(root, "KeyCount").text = str(count)
                data = ET.tostring(root)
                size = length = len(data)
                status = 200
            elif path.is_file():
                size = path.stat().st_size
                end = size
                status = 200
                requested = self.headers.get("Range")
                if requested:
                    left, right = requested.removeprefix("bytes=").split("-", 1)
                    if left:
                        start = int(left)
                        end = min(size, int(right) + 1) if right else size
                    else:
                        start = max(0, size - int(right))
                    if start >= size or end <= start:
                        self.send_error(416)
                        return
                    status = 206
                length = end - start
                data = b""
                if not head:
                    data = self.server.file_range(path, start, end) if length else b""
            else:
                self.send_error(404)
                return
            self.trace_state.update(start=start, bytes=0 if head else length)
            self.send_response(status)
            self.send_header("Content-Length", str(length))
            self.send_header("ETag", '"simulated"')
            self.send_header("Last-Modified", "Thu, 01 Jan 2026 00:00:00 GMT")
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{start + length - 1}/{size}")
            self.trace_state["response_ready"] = time.monotonic()
            self.end_headers()
            if not head:
                self.server.transmit(self.wfile, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--latency-ms", type=float, default=20)
    parser.add_argument("--mib-per-second", type=float, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if args.latency_ms < 0 or args.mib_per_second <= 0 or args.concurrency <= 0:
        parser.error("latency must be nonnegative; bandwidth and concurrency must be positive")
    Server(args).serve_forever()
