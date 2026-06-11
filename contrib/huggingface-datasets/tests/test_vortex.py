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


def test_filters_and(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        filters=[("x", ">", 1), ("x", "<", 5)],
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds["x"] == [2, 3, 4]


def test_filters_or(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        filters=[[("x", "==", 1)], [("x", "==", 5)]],
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds["x"] == [1, 5]


def test_filters_in(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        filters=[("s", "in", ["a", "e"])],
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds["x"] == [1, 5]


def test_filters_vortex_expr(vortex_shards: list[str], tmp_path: Path) -> None:
    import vortex.expr as ve

    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        filters=ve.column("x") >= 4,
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds["x"] == [4, 5]


def test_filters_invalid_operator(vortex_shards: list[str], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filter operator"):
        load_dataset(
            "vortex",
            data_files=vortex_shards,
            filters=[("x", "~", 1)],
            cache_dir=str(tmp_path / "cache"),
        )


def test_filters_streaming(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        filters=[("x", ">", 1), ("x", "<", 5)],
        streaming=True,
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert [row["x"] for row in ds] == [2, 3, 4]


def test_limit_across_shards(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        limit=4,
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds["x"] == [1, 2, 3, 4]


def test_limit_with_filters(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        filters=[("x", ">", 1)],
        limit=2,
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds["x"] == [2, 3]


def test_indices_across_shards(vortex_shards: list[str], tmp_path: Path) -> None:
    # Unsorted on purpose: rows come back in ascending row order.
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        indices=[4, 0, 2],
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    assert ds["x"] == [1, 3, 5]


def test_indices_out_of_range(vortex_shards: list[str], tmp_path: Path) -> None:
    ds = load_dataset(
        "vortex",
        data_files=vortex_shards,
        indices=[0, 99],
        streaming=True,
        cache_dir=str(tmp_path / "cache"),
    )["train"]
    with pytest.raises(IndexError, match="total row count"):
        list(ds)


def test_indices_with_filters_raises(vortex_shards: list[str], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="indices cannot be combined"):
        load_dataset(
            "vortex",
            data_files=vortex_shards,
            indices=[0],
            filters=[("x", ">", 1)],
            cache_dir=str(tmp_path / "cache"),
        )


def test_on_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.vortex"
    bad.write_bytes(b"this is not a vortex file")
    good = tmp_path / "good.vortex"
    vortex.io.write(pa.table({"x": [1, 2, 3]}), str(good))
    data_files = [str(bad), str(good)]

    ds = load_dataset("vortex", data_files=data_files, on_bad_files="skip", cache_dir=str(tmp_path / "c1"))["train"]
    assert ds["x"] == [1, 2, 3]

    ds = load_dataset("vortex", data_files=data_files, on_bad_files="warn", cache_dir=str(tmp_path / "c2"))["train"]
    assert ds["x"] == [1, 2, 3]

    with pytest.raises(Exception):  # noqa: B017 - datasets wraps the underlying error
        load_dataset("vortex", data_files=data_files, cache_dir=str(tmp_path / "c3"))

    with pytest.raises(ValueError, match="on_bad_files"):
        load_dataset("vortex", data_files=data_files, on_bad_files="bogus", cache_dir=str(tmp_path / "c4"))


def test_generate_num_examples(vortex_shards: list[str], tmp_path: Path) -> None:
    builder = datasets.load_dataset_builder(
        "vortex", data_files=vortex_shards, cache_dir=str(tmp_path / "cache")
    )
    assert list(builder._generate_num_examples(files=[[vortex_shards[0]], [vortex_shards[1]]])) == [3, 2]

    limited = datasets.load_dataset_builder(
        "vortex", data_files=vortex_shards, limit=2, cache_dir=str(tmp_path / "cache2")
    )
    with pytest.raises(NotImplementedError):
        list(limited._generate_num_examples(files=[[vortex_shards[0]]]))
