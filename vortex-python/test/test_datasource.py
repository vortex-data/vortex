# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

import pyarrow as pa
import pytest
import ray
from pyarrow.fs import GcsFileSystem
from ray.data import read_datasource
from typing_extensions import override

import vortex as vx
from vortex.ray.datasource import VortexDatasource, partition

from .test_file import record


@pytest.fixture(scope="module")
def ray_init() -> Iterator[None]:
    # Ray's uv_runtime_env_hook would auto-upload the working directory to
    # workers, but vortex-python's compiled _lib extension exceeds Ray's
    # 512 MiB upload limit. Disable the hook for these local-mode tests.
    # (Ray 2.55 added a string-type validation that broke the previous
    # `working_dir: None` workaround from ray-project/ray#53848.)
    import ray._private.ray_constants as ray_constants

    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    _ = ray.init()
    yield None
    ray.shutdown()


def test_partition() -> None:
    assert partition(1, []) == [[]]
    assert partition(1, [1]) == [[1]]
    assert partition(1, [1, 2, 3]) == [[1, 2, 3]]

    assert partition(2, [1, 2, 3]) == [[1, 2], [3]]
    assert partition(3, [1, 2, 3]) == [[1], [2], [3]]

    assert partition(2, list(range(9))) == [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    assert partition(3, list(range(9))) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    assert partition(3, list(range(11))) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]


@pytest.mark.parametrize("scheme", ["", "file://", "fsspec+file://"])
def test_vortex_datasource(ray_init, tmpdir_factory, scheme: str) -> None:
    folder = tmpdir_factory.mktemp("data")

    arr1 = vx.array([record(x) for x in range(5)])
    vx.io.write(arr1, str(folder / "01.vortex"))

    arr2 = vx.array([record(x) for x in range(5, 10)])
    vx.io.write(arr2, str(folder / "02.vortex"))

    ds = read_datasource(VortexDatasource(url=scheme + str(folder)))

    # Without an explicit sort, Ray may reorder rows *even within a single record batch*.
    ds = ds.sort("index")

    tbl = pa.concat_tables(pa.Table.from_pydict(x) for x in ds.iter_batches())
    expected = pa.Table.from_pylist([record(x) for x in range(0, 10)], schema=tbl.schema)

    assert tbl == expected


class _S3Handler(BaseHTTPRequestHandler):
    """Serves the files under `root` as path-style S3 objects."""

    root: ClassVar[Path]

    def do_HEAD(self) -> None:
        self._send_object(body=False)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if "list-type" not in query:
            self._send_object(body=True)
            return
        bucket = self.root / unquote(url.path).strip("/")
        prefix = query.get("prefix", [""])[0]
        keys = [p.relative_to(bucket).as_posix() for p in sorted(bucket.rglob("*")) if p.is_file()]
        keys = [key for key in keys if key.startswith(prefix)]
        contents = "".join(
            f"<Contents><Key>{key}</Key><Size>{(bucket / key).stat().st_size}</Size>"
            '<LastModified>2026-01-01T00:00:00Z</LastModified><ETag>"0"</ETag></Contents>'
            for key in keys
        )
        body = (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<KeyCount>{len(keys)}</KeyCount>{contents}</ListBucketResult>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    def _send_object(self, body: bool) -> None:
        file = self.root / unquote(urlsplit(self.path).path).lstrip("/")
        if not file.is_file():
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = file.read_bytes()
        ranged = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
        start, end = (int(ranged[1]), min(int(ranged[2]), len(data) - 1)) if ranged else (0, len(data) - 1)
        self.send_response(206 if ranged else 200)
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", '"0"')
        self.send_header("Last-Modified", "Thu, 01 Jan 2026 00:00:00 GMT")
        self.end_headers()
        if body:
            _ = self.wfile.write(data[start : end + 1])

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def s3_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    handler = type("Handler", (_S3Handler,), {"root": tmp_path})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key, value in {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_REGION": "us-east-1",
        "AWS_ENDPOINT_URL": f"http://127.0.0.1:{server.server_address[1]}",
        "AWS_ALLOW_HTTP": "true",
        "AWS_EC2_METADATA_DISABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    yield tmp_path
    server.shutdown()


def test_vortex_datasource_reads_s3(s3_root: Path) -> None:
    folder = s3_root / "ray-datasource-test" / "data"
    folder.mkdir(parents=True)
    vx.io.write(vx.array([record(x) for x in range(5)]), str(folder / "01.vortex"))
    vx.io.write(vx.array([record(x) for x in range(5, 10)]), str(folder / "a#b%41.vortex"))

    datasource = VortexDatasource(url="s3://ray-datasource-test/data/")

    blocks = [block for task in datasource.get_read_tasks(2) for block in task()]
    assert sorted(i for block in blocks for i in block["index"]) == list(range(10))


def test_url_scheme_of_gcs_is_gs() -> None:
    from vortex.ray.datasource import _url_scheme

    assert _url_scheme(GcsFileSystem(anonymous=True)) == "gs"
