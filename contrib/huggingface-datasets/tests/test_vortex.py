# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# NOTE: strip the two SPDX lines above when copying this file into
# huggingface/datasets, and consider switching `pytest.importorskip` to a
# `require_vortex` decorator in `tests/utils.py` if the maintainers prefer
# that convention (see `require_*` helpers used by other format tests).

"""Tests for the Vortex packaged module.

Destined for ``tests/packaged_modules/test_vortex.py`` in the
``huggingface/datasets`` repository.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

vortex = pytest.importorskip("vortex")

import datasets  # noqa: E402
from datasets import load_dataset  # noqa: E402


@pytest.fixture
def vortex_shards(tmp_path: Path) -> list[str]:
    shard1, shard2 = tmp_path / "part-0.vortex", tmp_path / "part-1.vortex"
    vortex.io.write(pa.table({"x": [1, 2, 3], "s": ["a", "b", "c"]}), str(shard1))
    vortex.io.write(pa.table({"x": [4, 5], "s": ["d", "e"]}), str(shard2))
    return [str(shard1), str(shard2)]


def test_load_dataset(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files={"train": vortex_shards},
        cache_dir=str(tmp_path / "cache"),
    )["train"]

    assert ds.num_rows == 5
    assert ds["x"] == [1, 2, 3, 4, 5]
    assert ds["s"] == ["a", "b", "c", "d", "e"]
    assert ds.features["x"].dtype == "int64"
    assert ds.features["s"].dtype == "string"


def test_load_dataset_streaming(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files={"train": vortex_shards},
        cache_dir=str(tmp_path / "cache"),
        streaming=True,
    )["train"]

    rows = list(ds)
    assert [row["x"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["s"] for row in rows] == ["a", "b", "c", "d", "e"]


def test_load_dataset_columns(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards[0],
        columns=["x"],
        cache_dir=str(tmp_path / "cache"),
    )["train"]

    assert ds.column_names == ["x"]
    assert ds["x"] == [1, 2, 3]


def test_load_dataset_explicit_features(vortex_shards: list[str], tmp_path: Path) -> None:
    features = datasets.Features({"x": datasets.Value("int64"), "s": datasets.Value("string")})
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards[0],
        features=features,
        cache_dir=str(tmp_path / "cache"),
    )["train"]

    assert ds.features == features
    assert ds.num_rows == 3


def test_columns_features_mismatch_raises(vortex_shards: list[str], tmp_path: Path) -> None:
    features = datasets.Features({"x": datasets.Value("int64"), "s": datasets.Value("string")})
    with pytest.raises(ValueError, match="columns and features"):
        load_dataset(
            "vortex",
            data_files=vortex_shards[0],
            columns=["x"],
            features=features,
            cache_dir=str(tmp_path / "cache"),
        )


def test_extension_inference(vortex_shards: list[str], tmp_path: Path) -> None:
    """A directory of .vortex files resolves to the vortex builder without naming it."""
    ds = load_dataset(
        str(Path(vortex_shards[0]).parent),
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds.num_rows == 5
