# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the Hugging Face ``datasets`` integration: the Vortex builder registered by
:func:`vortex.hf.register_datasets` and :func:`vortex.hf.dataset_to_vortex`."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

import vortex as vx


def test_register_datasets_builder(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    vx.hf.register_datasets()

    shard1, shard2 = tmp_path / "part-0.vortex", tmp_path / "part-1.vortex"
    vx.io.write(pa.table({"x": [1, 2, 3], "s": ["a", "b", "c"]}), str(shard1))
    vx.io.write(pa.table({"x": [4, 5], "s": ["d", "e"]}), str(shard2))

    ds = datasets.load_dataset(
        "vortex",
        data_files={"train": [str(shard1), str(shard2)]},
        cache_dir=str(tmp_path / "cache"),
    )["train"]

    assert ds.num_rows == 5
    assert ds["x"] == [1, 2, 3, 4, 5]
    assert ds["s"] == ["a", "b", "c", "d", "e"]
    assert ds.features["x"].dtype == "int64"
    assert ds.features["s"].dtype == "string"


def test_register_datasets_builder_columns(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    vx.hf.register_datasets()

    shard = tmp_path / "part-0.vortex"
    vx.io.write(pa.table({"x": [1, 2, 3], "s": ["a", "b", "c"]}), str(shard))

    ds = datasets.load_dataset(
        "vortex",
        data_files=str(shard),
        columns=["x"],
        cache_dir=str(tmp_path / "cache"),
    )["train"]

    assert ds.column_names == ["x"]
    assert ds["x"] == [1, 2, 3]


def test_dataset_to_vortex_roundtrip(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")

    ds = datasets.Dataset.from_dict({"x": [1, 2, 3], "s": ["a", "b", "c"]})
    path = tmp_path / "converted.vortex"
    vx.hf.dataset_to_vortex(ds, str(path))

    result = vx.open(str(path)).scan().read_all().to_arrow_table()
    assert result.column("x").to_pylist() == [1, 2, 3]
    assert result.column("s").to_pylist() == ["a", "b", "c"]
