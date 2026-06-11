# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Regression tests exercise runtime validation of bad inputs, so relax strict typing rules.
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnusedCallResult=false
# pyright: reportUnusedParameter=false

"""Regression tests for bugs found by fuzzing the file IO API.

Each test corresponds to a previously panicking or corrupting case; the Rust-level
regression tests live next to the fixes (vortex-array, vortex-layout, vortex-file).
"""

import os

import pyarrow as pa
import pytest

import vortex as vx


def write_and_open(table: pa.Table, tmp_path) -> vx.VortexFile:
    path = os.path.join(str(tmp_path), "data.vortex")
    vx.io.write(table, path)
    return vx.open(path)


@pytest.mark.parametrize(
    "arrow_type",
    [pa.duration("us"), pa.duration("s"), pa.binary(3)],
    ids=["duration_us", "duration_s", "fixed_size_binary"],
)
def test_unsupported_arrow_type_raises_cleanly(arrow_type: pa.DataType):
    # Used to abort with a PanicException from `unimplemented!()` in DType::from_arrow.
    table = pa.table({"c0": pa.array([], type=arrow_type)})
    with pytest.raises(RuntimeError, match="not yet supported"):
        vx.array(table)


def test_write_empty_table_with_struct_column(tmp_path):
    # Used to panic with "must have visited at least one chunk" because empty chunks are
    # filtered out before reaching the struct validity writer.
    table = pa.table({"c0": pa.array([], type=pa.struct([("a", pa.int8())]))})
    vxf = write_and_open(table, tmp_path)
    assert len(vxf) == 0
    assert vxf.scan().read_all().to_arrow_table().to_pylist() == []


def test_struct_of_struct_null_roundtrip(tmp_path):
    # A struct field nested directly inside another struct used to lose its nullability on
    # the read path: the inner nulls were applied to the outer struct instead.
    arr = pa.array(
        [{"a": {"c": 1}}, {"a": None}],
        type=pa.struct([("a", pa.struct([("c", pa.int64())]))]),
    )
    table = pa.table({"c0": arr})
    vxf = write_and_open(table, tmp_path)
    assert vxf.scan().read_all().to_arrow_table().to_pylist() == [
        {"c0": {"a": {"c": 1}}},
        {"c0": {"a": None}},
    ]


def test_sliced_fixed_size_list_of_struct():
    # Converting a sliced fixed_size_list<struct> used to panic with "end <= self.len()"
    # inside arrow-rs; vortex now normalizes imported ArrayData first.
    t = pa.list_(pa.struct([("a", pa.int8())]), 2)
    arr = pa.array([[{"a": 1}, {"a": 2}], [{"a": 3}, {"a": 4}]], type=t)
    result = vx.array(pa.table({"c0": arr.slice(1)}))
    assert result.to_arrow_table().to_pylist() == [{"c0": [{"a": 3}, {"a": 4}]}]


def test_sliced_struct_of_struct():
    # Same arrow-rs offset bug, triggered through a sliced struct<struct> column.
    t = pa.struct([("a", pa.struct([("c", pa.int8())]))])
    arr = pa.array([{"a": {"c": 1}}, {"a": {"c": 2}}], type=t)
    result = vx.array(pa.table({"c0": arr.slice(1)}))
    assert result.to_arrow_table().to_pylist() == [{"c0": {"a": {"c": 2}}}]


def test_sliced_fixed_size_list_of_struct_roundtrip(tmp_path):
    t = pa.list_(pa.struct([("a", pa.int8())]), 2)
    arr = pa.array([[{"a": 1}, None], [{"a": 3}, {"a": 4}]], type=t)
    table = pa.table({"c0": arr.slice(1)})
    vxf = write_and_open(table, tmp_path)
    assert vxf.scan().read_all().to_arrow_table().to_pylist() == [{"c0": [{"a": 3}, {"a": 4}]}]


@pytest.fixture
def simple_file(tmp_path) -> vx.VortexFile:
    return write_and_open(pa.table({"a": pa.array([1, 2, 3], type=pa.int64())}), tmp_path)


def test_scan_batch_size_zero_raises(simple_file: vx.VortexFile):
    # Used to panic with a step_by(0) panic inside split planning.
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        simple_file.scan(batch_size=0).read_all()


def test_scan_batch_size_negative_raises(simple_file: vx.VortexFile):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        simple_file.scan(batch_size=-5).read_all()


def test_scan_limit_negative_raises(simple_file: vx.VortexFile):
    with pytest.raises(ValueError, match="limit must be non-negative"):
        simple_file.scan(limit=-1).read_all()


def test_scan_unsorted_indices_raise(simple_file: vx.VortexFile):
    # Used to panic in debug builds and silently return wrong rows in release builds.
    indices = vx.array(pa.array([2, 0], type=pa.uint64()))
    with pytest.raises(RuntimeError, match="sorted"):
        simple_file.scan(indices=indices).read_all()


def test_scan_sorted_indices_work(simple_file: vx.VortexFile):
    indices = vx.array(pa.array([0, 2], type=pa.uint64()))
    result = simple_file.scan(indices=indices).read_all().to_arrow_table()
    assert result.to_pylist() == [{"a": 1}, {"a": 3}]


def test_read_url_rejects_integer_projection(simple_file: vx.VortexFile, tmp_path):
    # The stub used to advertise list[int] projections that the implementation rejects.
    path = os.path.join(str(tmp_path), "data.vortex")
    vx.io.write(pa.table({"a": pa.array([1, 2, 3])}), path)
    with pytest.raises(TypeError, match="projection"):
        vx.io.read_url(f"file://{path}", projection=[1])  # pyright: ignore[reportArgumentType]


def test_dataset_take_preserves_order_and_duplicates(simple_file: vx.VortexFile):
    # Vortex scans require sorted unique indices, but pyarrow take semantics require the
    # result to follow the indices, duplicates included. Used to silently mis-handle both.
    ds = simple_file.to_dataset()
    assert ds.take(pa.array([2, 0])).to_pylist() == [{"a": 3}, {"a": 1}]
    assert ds.take(pa.array([1, 1, 0])).to_pylist() == [{"a": 2}, {"a": 2}, {"a": 1}]
    assert ds.take(pa.array([], type=pa.uint64())).to_pylist() == []
