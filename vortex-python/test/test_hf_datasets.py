# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The Vortex loader of Hugging Face ``datasets`` (5.1 and later) reads through the Vortex API.

These tests keep the parts of the API that the loader calls working.
"""

from pathlib import Path

import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve

datasets = pytest.importorskip("datasets", minversion="5.1.0")


@pytest.fixture
def data_files(tmp_path: Path) -> list[str]:
    files = []
    for part in range(2):
        start = part * 5
        table = pa.table(
            {
                "index": pa.array(range(start, start + 5), pa.int64()),
                "text": [f"row {i}" for i in range(start, start + 5)],
            }
        )
        path = tmp_path / f"part-{part}.vortex"
        vx.io.write(table, str(path))
        files.append(str(path))

    return files


def test_load_dataset(data_files: list[str]):
    ds = datasets.load_dataset("vortex", data_files=data_files, split="train")

    assert ds.num_rows == 10
    assert sorted(ds["index"]) == list(range(10))
    assert list(ds.features) == ["index", "text"]
    assert ds[0]["text"] == "row 0"


def test_load_dataset_streaming_with_columns(data_files: list[str]):
    ds = datasets.load_dataset("vortex", data_files=data_files, split="train", streaming=True, columns=["index"])

    rows = list(ds)

    assert [row["index"] for row in rows] == list(range(10))
    assert all(set(row) == {"index"} for row in rows)


@pytest.mark.parametrize(
    "filters",
    [
        [("index", ">=", 7)],
        [[("index", "in", [7, 8, 9])]],
        ve.column("index") >= 7,
    ],
)
def test_load_dataset_pushes_filters_into_the_scan(data_files: list[str], filters: object):
    ds = datasets.load_dataset("vortex", data_files=data_files, split="train", streaming=True, filters=filters)

    assert [row["index"] for row in ds] == [7, 8, 9]


def test_load_dataset_reshards_on_file_splits(data_files: list[str]):
    ds = datasets.load_dataset("vortex", data_files=data_files, split="train", streaming=True)
    if not hasattr(ds, "reshard"):
        pytest.skip("IterableDataset.reshard is not available")

    resharded = ds.reshard()

    assert resharded.num_shards >= 2
    assert sorted(row["index"] for row in resharded) == list(range(10))
