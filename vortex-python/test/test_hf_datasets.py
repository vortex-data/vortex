# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import cast

import datasets as hf_datasets
import pyarrow as pa
import pytest
import vortex.datasets as vx_datasets

import vortex as vx
import vortex.expr as ve


def test_datasets_module_is_lazy_exported():
    assert vx.datasets is vx_datasets


def write_vortex(path: Path, rows: list[dict[str, object]]) -> None:
    vx.io.write(pa.Table.from_pylist(rows), str(path))


def test_load_dataset_streaming_local_splits(tmp_path: Path):
    write_vortex(
        tmp_path / "train-0000.vortex",
        [
            {"text": "zero", "label": 0, "tokens": 10},
            {"text": "one", "label": 1, "tokens": 11},
        ],
    )
    write_vortex(
        tmp_path / "train-0001.vortex",
        [
            {"text": "two", "label": 0, "tokens": 12},
            {"text": "three", "label": 1, "tokens": 13},
        ],
    )
    write_vortex(tmp_path / "validation.vortex", [{"text": "valid", "label": 1, "tokens": 20}])

    dataset = vx_datasets.load_dataset(
        tmp_path,
        data_files={"train": "train-*.vortex", "validation": "validation.vortex"},
    )

    assert isinstance(dataset, hf_datasets.IterableDatasetDict)
    assert list(dataset) == ["train", "validation"]
    assert list(dataset["train"].take(3)) == [
        {"label": 0, "text": "zero", "tokens": 10},
        {"label": 1, "text": "one", "tokens": 11},
        {"label": 0, "text": "two", "tokens": 12},
    ]
    assert list(dataset["validation"]) == [{"label": 1, "text": "valid", "tokens": 20}]


def test_streaming_select_columns_pushes_projection(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero", "label": 0}, {"text": "one", "label": 1}])

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    selected = dataset.select_columns(["text"])

    assert isinstance(selected, vx_datasets.VortexIterableDataset)
    assert selected._vortex_columns == ("text",)  # pyright: ignore[reportPrivateUsage]
    assert list(selected) == [{"text": "zero"}, {"text": "one"}]


def test_streaming_filter_accepts_vortex_expression(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [
            {"text": "zero", "label": 0},
            {"text": "one", "label": 1},
            {"text": "two", "label": 0},
        ],
    )

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    filtered = dataset.filter(ve.column("label") == 1)

    assert isinstance(filtered, vx_datasets.VortexIterableDataset)
    assert list(filtered) == [{"label": 1, "text": "one"}]


def test_load_dataset_materializes_to_hf_dataset(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero", "label": 0}, {"text": "one", "label": 1}])

    dataset = vx_datasets.load_dataset(
        tmp_path / "train.vortex",
        split="train",
        streaming=False,
        columns=["text", "label"],
        keep_in_memory=True,
    )

    assert isinstance(dataset, hf_datasets.Dataset)
    assert dataset.to_list() == [{"text": "zero", "label": 0}, {"text": "one", "label": 1}]


def test_streaming_resume_with_limit_reads_full_limit(tmp_path: Path):
    rows: list[dict[str, object]] = [{"idx": i} for i in range(10)]
    write_vortex(tmp_path / "train.vortex", rows)

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train", limit=8, batch_size=2),
    )
    examples = cast(
        vx_datasets._VortexExamplesIterable,  # pyright: ignore[reportPrivateUsage]
        dataset._ex_iterable,  # pyright: ignore[reportPrivateUsage]
    )
    # Simulate resuming after the first two rows of the file were already yielded.
    examples._state_dict = {  # pyright: ignore[reportPrivateUsage]
        "file_idx": 0,
        "file_row_idx": 2,
        "num_yielded": 2,
        "type": type(examples).__name__,
    }

    produced = [row for _key, table in examples._iter_arrow() for row in table.to_pylist()]  # pyright: ignore[reportPrivateUsage]

    # The limit of 8 must still be honored: six rows remain after the two already yielded.
    assert produced == rows[2:8]


def test_streaming_filter_after_take_is_rejected(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero", "label": 0}, {"text": "one", "label": 1}])

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    limited = dataset.take(1)
    assert isinstance(limited, vx_datasets.VortexIterableDataset)
    with pytest.raises(ValueError, match="after a row limit"):
        _ = limited.filter(ve.column("label") == 1)


def test_streaming_filter_then_take_pushes_down(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [{"text": "a", "label": 0}, {"text": "b", "label": 1}, {"text": "c", "label": 1}],
    )

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    result = dataset.filter(ve.column("label") == 1).take(1)

    assert isinstance(result, vx_datasets.VortexIterableDataset)
    assert list(result) == [{"text": "b", "label": 1}]


def test_streaming_filter_and_limit_combined(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [{"text": "a", "label": 0}, {"text": "b", "label": 1}, {"text": "c", "label": 1}],
    )

    # Vortex cannot scan with a filter and a limit at once; load_dataset must still honor both.
    dataset = vx_datasets.load_dataset(
        tmp_path / "train.vortex", split="train", filter=ve.column("label") == 1, limit=1
    )

    assert isinstance(dataset, vx_datasets.VortexIterableDataset)
    assert list(dataset) == [{"text": "b", "label": 1}]


def test_materialize_filter_and_limit_combined(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [{"text": "a", "label": 0}, {"text": "b", "label": 1}, {"text": "c", "label": 1}],
    )

    dataset = vx_datasets.load_dataset(
        tmp_path / "train.vortex",
        split="train",
        streaming=False,
        filter=ve.column("label") == 1,
        limit=1,
        keep_in_memory=True,
    )

    assert isinstance(dataset, hf_datasets.Dataset)
    assert dataset.to_list() == [{"text": "b", "label": 1}]


def test_load_dataset_multi_split_without_mapping_raises(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero"}])

    with pytest.raises(ValueError, match="data_files"):
        _ = vx_datasets.load_dataset(tmp_path, split=["train", "validation"])
