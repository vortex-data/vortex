# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from pathlib import Path

import pytest
from vortex.store import HTTPStore, LocalStore, S3Store

import vortex as vx


@pytest.mark.parametrize(
    "store_type, url, options",
    [
        (LocalStore, "file:///", {}),
        (HTTPStore, "https://example.com/data", {}),
        (S3Store, "s3://test-bucket/data", {"region": "us-east-1", "skip_signature": True}),
    ],
)
def test_store_from_url(store_type, url, options):
    store = store_type.from_url(url, **options)
    assert isinstance(store, store_type)


def test_store_roundtrip(tmp_path: Path) -> None:
    # create a local store to write into
    local = LocalStore(prefix=tmp_path)

    records = vx.array([dict(name="Alice", salary=10), dict(name="Bob", salary=20), dict(name="Carol", salary=30)])

    assert len(records) == 3

    # write to the local store
    vx.io.write(records, "people.vortex", store=local)

    # verify file got written to correct location
    assert (tmp_path / "people.vortex").exists()

    # test vx.read for eager full-scan
    people = vx.io.read_url("people.vortex", store=local)

    assert people.to_pylist() == records.to_pylist()


def test_read_url_integer_projection(tmp_path: Path) -> None:
    local = LocalStore(prefix=tmp_path)
    records = vx.array([dict(name="Alice", salary=10), dict(name="Bob", salary=20)])
    vx.io.write(records, "people.vortex", store=local)

    # Columns are name (0) and salary (1); select salary by position.
    by_index = vx.io.read_url("people.vortex", store=local, projection=[1])
    assert by_index.to_pylist() == [{"salary": 10}, {"salary": 20}]

    # Integer and name projection agree.
    by_name = vx.io.read_url("people.vortex", store=local, projection=["salary"])
    assert by_index.to_pylist() == by_name.to_pylist()


def test_read_url_integer_projection_out_of_range(tmp_path: Path) -> None:
    local = LocalStore(prefix=tmp_path)
    records = vx.array([dict(name="Alice", salary=10)])
    vx.io.write(records, "people.vortex", store=local)

    with pytest.raises(IndexError):
        vx.io.read_url("people.vortex", store=local, projection=[99])
