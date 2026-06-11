# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the Hugging Face integration in :mod:`vortex.hf`.

These tests run a small local HTTP server that mimics the Hub's ``resolve`` endpoint
(including HTTP Range support), so no network access or Hugging Face account is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import vortex.store
from vortex.hf import HFLocation, resolve_url, store_and_path
from vortex.hf import token as hf_token
from vortex.store import HTTPStore

import vortex as vx

from .hub_server import HubState, sample_table

# --- URL parsing and translation ---


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "hf://datasets/my-org/my-data/file.vortex",
            HFLocation(repo_id="my-org/my-data", path="file.vortex"),
        ),
        (
            "hf://datasets/my-org/my-data@v1.0/dir/file.vortex",
            HFLocation(repo_id="my-org/my-data", path="dir/file.vortex", revision="v1.0"),
        ),
        (
            "hf://datasets/my-org/my-data@refs%2Fconvert%2Fparquet/x.vortex",
            HFLocation(repo_id="my-org/my-data", path="x.vortex", revision="refs/convert/parquet"),
        ),
        (
            "hf://my-org/my-model/weights.vortex",
            HFLocation(repo_id="my-org/my-model", path="weights.vortex", repo_type="model"),
        ),
        (
            "hf://spaces/my-org/my-space/data.vortex",
            HFLocation(repo_id="my-org/my-space", path="data.vortex", repo_type="space"),
        ),
        ("hf://datasets/my-org/my-data", HFLocation(repo_id="my-org/my-data")),
    ],
)
def test_parse(url: str, expected: HFLocation) -> None:
    assert HFLocation.parse(url) == expected


@pytest.mark.parametrize(
    "url",
    ["s3://bucket/file", "hf://", "hf://datasets", "hf://datasets/only-namespace", "hf://datasets/org/name@/x"],
)
def test_parse_invalid(url: str) -> None:
    with pytest.raises(ValueError):
        HFLocation.parse(url)


def test_resolve_url() -> None:
    assert (
        resolve_url("hf://datasets/my-org/my-data@v2/dir/file.vortex")
        == "https://huggingface.co/datasets/my-org/my-data/resolve/v2/dir/file.vortex"
    )
    assert (
        resolve_url("hf://my-org/my-model/weights.vortex")
        == "https://huggingface.co/my-org/my-model/resolve/main/weights.vortex"
    )


def test_resolve_url_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.com/")
    assert (
        resolve_url("hf://datasets/my-org/my-data/file.vortex")
        == "https://hub.example.com/datasets/my-org/my-data/resolve/main/file.vortex"
    )


def test_token_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)

    assert hf_token() is None
    assert hf_token("hf_explicit") == "hf_explicit"

    (tmp_path / "token").write_text("hf_from_file\n")
    assert hf_token() == "hf_from_file"

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert hf_token() == "hf_from_env"
    assert hf_token("hf_explicit") == "hf_explicit"


def test_store_from_url() -> None:
    store = vortex.store.from_url("hf://datasets/my-org/my-data/dir")
    assert isinstance(store, HTTPStore)
    assert store.url == "https://huggingface.co/datasets/my-org/my-data/resolve/main/dir"


def test_store_from_url_rejects_store_config() -> None:
    with pytest.raises(ValueError, match="client_options"):
        vortex.store.from_url("hf://datasets/my-org/my-data", mkdir=True)


def test_store_and_path_requires_file_path() -> None:
    with pytest.raises(ValueError, match="file path"):
        store_and_path("hf://datasets/my-org/my-data")


def test_auth_header_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    store, path = store_and_path("hf://datasets/my-org/my-data/file.vortex")
    assert path == "file.vortex"
    headers = (store.client_options or {}).get("default_headers")
    assert headers is not None
    assert headers["authorization"] == b"Bearer hf_secret" or headers["authorization"] == "Bearer hf_secret"


# --- End-to-end reads against the local hub ---


def test_open_hf_url(local_hub: HubState) -> None:
    table = sample_table()
    local_hub.publish("datasets/test-org/test-repo/resolve/main/data/train.vortex", table)

    vxf = vx.open("hf://datasets/test-org/test-repo/data/train.vortex")
    assert len(vxf) == table.num_rows

    result = vxf.scan().read_all().to_arrow_table()
    assert result.column("x").to_pylist() == table.column("x").to_pylist()
    assert result.column("s").to_pylist() == table.column("s").to_pylist()


def test_open_hf_url_revision(local_hub: HubState) -> None:
    table = sample_table(num_rows=10)
    local_hub.publish("datasets/test-org/test-repo/resolve/v1.0/data.vortex", table)

    vxf = vx.open("hf://datasets/test-org/test-repo@v1.0/data.vortex")
    assert len(vxf) == 10


def test_open_hf_url_is_lazy(local_hub: HubState) -> None:
    table = sample_table(num_rows=20_000)
    target = local_hub.publish("datasets/test-org/test-repo/resolve/main/data.vortex", table)
    file_size = target.stat().st_size

    local_hub.reset()
    vxf = vx.open("hf://datasets/test-org/test-repo/data.vortex")
    projected = vxf.scan(["x"]).read_all()
    assert len(projected) == 20_000

    # A projected scan must not download the whole file, and every read is a ranged read.
    assert 0 < local_hub.bytes_served < file_size
    assert all(r.range is not None for r in local_hub.requests if r.method == "GET")


def test_open_hf_url_with_token(local_hub: HubState, monkeypatch: pytest.MonkeyPatch) -> None:
    local_hub.required_token = "hf_test_token"
    table = sample_table(num_rows=10)
    local_hub.publish("datasets/test-org/gated-repo/resolve/main/data.vortex", table)

    with pytest.raises(Exception, match="401|Unauthorized"):
        vx.open("hf://datasets/test-org/gated-repo/data.vortex")

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    vxf = vx.open("hf://datasets/test-org/gated-repo/data.vortex")
    assert len(vxf) == 10


def test_vortex_hf_open_explicit_token(local_hub: HubState) -> None:
    local_hub.required_token = "hf_explicit_token"
    table = sample_table(num_rows=10)
    local_hub.publish("datasets/test-org/gated-repo/resolve/main/data.vortex", table)

    vxf = vx.hf.open("hf://datasets/test-org/gated-repo/data.vortex", token="hf_explicit_token")
    assert len(vxf) == 10
