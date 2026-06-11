# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""End-to-end demo of the Vortex Hugging Face integration, fully local.

This script never touches the network. It stands up a local HTTP server that mimics the
Hugging Face Hub's ``resolve`` endpoint (with HTTP Range support), publishes a Vortex
file to it, and then:

1. converts a Parquet shard to Vortex,
2. lazily reads it back over ``hf://`` URLs, showing how few bytes a projected and
   filtered scan downloads compared to the file size,
3. loads the same files with ``datasets.load_dataset`` via the registered ``vortex``
   builder (if ``datasets`` is installed), and
4. does shuffled random-row access through the torch-compatible map-style dataset.

Run it with::

    uv run python vortex-python/examples/huggingface_local_demo.py

Against the real Hub the same code works by dropping the ``HF_ENDPOINT`` override, e.g.
``vx.open("hf://datasets/my-org/my-dataset/data/train.vortex")``.
"""

import os
import random
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

NUM_ROWS = 200_000
REPO_FILE = "datasets/demo-org/demo-dataset/resolve/main/data/train-00000.vortex"
HF_URL = "hf://datasets/demo-org/demo-dataset/data/train-00000.vortex"


class HubHandler(BaseHTTPRequestHandler):
    """Serves files like the Hub's resolve endpoint, counting the bytes it sends."""

    protocol_version = "HTTP/1.1"
    root: ClassVar[Path] = Path()
    bytes_served = 0
    requests = 0
    lock = threading.Lock()

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):
        self._serve(send_body=True)

    def do_HEAD(self):
        self._serve(send_body=False)

    def _serve(self, send_body):
        target = self.root / unquote(urlsplit(self.path).path).lstrip("/")
        if not target.is_file():
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
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
            end, status = min(end, size - 1), 206
        body = data[start : end + 1]
        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if send_body:
            self.wfile.write(body)
            with HubHandler.lock:
                HubHandler.bytes_served += len(body)
                HubHandler.requests += 1

    @classmethod
    def reset(cls):
        with cls.lock:
            cls.bytes_served = 0
            cls.requests = 0


def make_parquet_shard(path: Path) -> pa.Table:
    """A synthetic 'web documents' shard, vaguely FineWeb-shaped."""
    rng = random.Random(42)
    words = ["vortex", "columnar", "compression", "lazy", "scan", "arrow", "hub", "data"]
    table = pa.table(
        {
            "id": list(range(NUM_ROWS)),
            "url": [f"https://example.com/page/{rng.randrange(1 << 32):08x}" for _ in range(NUM_ROWS)],
            "text": [" ".join(rng.choices(words, k=rng.randrange(5, 30))) for _ in range(NUM_ROWS)],
            "score": [rng.random() for _ in range(NUM_ROWS)],
        }
    )
    pq.write_table(table, path)
    return table


def main() -> None:
    from vortex.torch_ import VortexMapDataset

    import vortex as vx
    import vortex.expr as ve

    workdir = Path(tempfile.mkdtemp(prefix="vortex-hf-demo-"))
    hub_root = workdir / "hub"

    # -- 1. Convert a Parquet shard to Vortex and "publish" it to the local hub. --
    parquet_path = workdir / "train-00000.parquet"
    make_parquet_shard(parquet_path)
    vortex_path = hub_root / REPO_FILE
    vortex_path.parent.mkdir(parents=True)
    vx.io.write(pq.read_table(parquet_path), str(vortex_path))

    parquet_size = parquet_path.stat().st_size
    vortex_size = vortex_path.stat().st_size
    print(f"shard: {NUM_ROWS:,} rows")
    print(f"  parquet: {parquet_size / 1e6:8.2f} MB")
    print(f"  vortex:  {vortex_size / 1e6:8.2f} MB")

    # -- 2. Serve it like the Hub and read it lazily via hf:// URLs. --
    HubHandler.root = hub_root
    server = ThreadingHTTPServer(("127.0.0.1", 0), HubHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{server.server_address[1]}"

    HubHandler.reset()
    vxf = vx.open(HF_URL)
    print(f"\nopened {HF_URL}")
    print(f"  rows: {len(vxf):,}, schema fields: {[f.name for f in vxf.dtype.to_arrow_schema()]}")
    print(f"  metadata reads: {HubHandler.bytes_served / 1e3:.1f} kB in {HubHandler.requests} requests")

    HubHandler.reset()
    scores = vxf.scan(["score"]).read_all()
    print(f"\nprojected scan of 'score' column ({len(scores):,} rows)")
    print(
        f"  downloaded {HubHandler.bytes_served / 1e6:.2f} MB "
        f"of a {vortex_size / 1e6:.2f} MB file ({100 * HubHandler.bytes_served / vortex_size:.1f}%)"
    )

    HubHandler.reset()
    top = vx.open(HF_URL).scan(["id", "score"], expr=ve.column("score") > 0.999).read_all()
    print(f"\nfiltered scan (score > 0.999) on a freshly opened file matched {len(top):,} rows")
    print(f"  downloaded {HubHandler.bytes_served / 1e6:.2f} MB ({100 * HubHandler.bytes_served / vortex_size:.1f}%)")

    # -- 3. Load with the Hugging Face `datasets` library. --
    try:
        import datasets
    except ImportError:
        print("\n`datasets` not installed; skipping load_dataset demo")
    else:
        import vortex.hf

        vortex.hf.register_datasets()
        ds = datasets.load_dataset(
            "vortex",
            data_files={"train": str(vortex_path)},
            cache_dir=str(workdir / "hf-cache"),
        )["train"]
        print(f"\ndatasets.load_dataset('vortex', ...) -> {ds}")

    # -- 4. Shuffled random access, DataLoader-style. --
    HubHandler.reset()
    mapped = VortexMapDataset(HF_URL, projection=["id", "text"])
    rng = random.Random(7)
    batch = mapped.__getitems__([rng.randrange(NUM_ROWS) for _ in range(32)])
    print(f"\nrandom-access batch of 32 rows over {HF_URL}")
    print(f"  first row: {batch[0]}")
    print(f"  downloaded {HubHandler.bytes_served / 1e3:.1f} kB for the batch")

    server.shutdown()
    print(f"\nworking directory: {workdir}")


if __name__ == "__main__":
    main()
