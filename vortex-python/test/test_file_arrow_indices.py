# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from pathlib import Path

import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve


@pytest.fixture
def indexed_file(tmp_path: Path) -> tuple[vx.VortexFile, pa.Table]:
    table = pa.table(
        {
            "id": list(range(9)),
            "payload": ["first", None, "long string with 数据", "", "four", "five", None, "seven", "last"],
        }
    )
    path = tmp_path / "indexed.vortex"
    vx.io.write(vx.compress(vx.array(table)), str(path))
    return vx.open(str(path)), table


@pytest.mark.parametrize("indices", [[], [1], [0, 2, 5, 8]])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_arrow_indices_preserve_values_and_requested_schema(indexed_file, indices, batch_size):
    file, table = indexed_file
    indices = pa.array(indices, type=pa.uint64())
    schema = pa.schema([("payload", pa.string())])
    reader = file.to_arrow(["payload"], indices=vx.array(indices), batch_size=batch_size, schema=schema)
    batches = list(reader)
    scan_batches = list(file.scan(["payload"], indices=vx.array(indices), batch_size=batch_size))
    assert [batch.num_rows for batch in batches] == [len(batch) for batch in scan_batches]
    result = pa.Table.from_batches(batches, schema=reader.schema)
    assert result.equals(table.select(["payload"]).take(indices))


def test_arrow_indices_apply_filter(indexed_file):
    file, table = indexed_file
    reader = file.to_arrow(
        ["payload"],
        indices=vx.array([0, 2, 5, 8]),
        expr=ve.column("id") >= 4,
        schema=pa.schema([("payload", pa.string())]),
    )
    assert reader.read_all().equals(table.select(["payload"]).take(pa.array([5, 8])))


def test_arrow_indices_apply_limit(indexed_file):
    file, table = indexed_file
    reader = file.to_arrow(
        ["payload"],
        indices=vx.array([0, 2, 5, 8]),
        limit=1,
        schema=pa.schema([("payload", pa.string())]),
    )
    assert reader.read_all().equals(table.select(["payload"]).slice(0, 1))


def test_arrow_indices_apply_filter_then_limit(indexed_file):
    file, table = indexed_file
    reader = file.to_arrow(
        ["payload"],
        indices=vx.array([0, 2, 5, 8]),
        expr=ve.column("id") >= 4,
        limit=1,
        schema=pa.schema([("payload", pa.string())]),
    )
    assert reader.read_all().equals(table.select(["payload"]).take(pa.array([5])))


@pytest.mark.parametrize("indices", [[2, 1], [1, 1], [None, 1]])
def test_arrow_indices_use_scan_validation(indexed_file, indices):
    file, _ = indexed_file
    indices = vx.array(pa.array(indices, type=pa.uint64()))
    with pytest.raises(ValueError):
        file.to_arrow(indices=indices)


@pytest.mark.parametrize("indices", [None, [], [1], [0, 2, 5, 8]])
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("combine_chunks", [False, True])
def test_iterator_to_arrow_array_matches_read_all(
    indexed_file: tuple[vx.VortexFile, pa.Table], indices, batch_size, combine_chunks
):
    file, _ = indexed_file
    selection = None if indices is None else vx.array(pa.array(indices, type=pa.uint64()))
    arrow_type = pa.struct([("id", pa.int64()), ("payload", pa.string_view())])

    def scan() -> vx.ArrayIterator:
        return file.scan(indices=selection, batch_size=batch_size)

    expected = scan().read_all().to_arrow_array(arrow_type=arrow_type, combine_chunks=combine_chunks)
    result = scan().to_arrow_array(arrow_type=arrow_type, combine_chunks=combine_chunks)
    assert type(result) is type(expected)
    assert result == expected

    inferred = scan().read_all().to_arrow_array(combine_chunks=combine_chunks)
    assert scan().to_arrow_array(combine_chunks=combine_chunks) == inferred


def test_iterator_to_arrow_array_after_read_is_empty(indexed_file):
    file, _ = indexed_file
    iterator = file.scan()
    assert len(iterator.to_arrow_array()) == 9
    assert len(iterator.to_arrow_array()) == 0


def test_long_row_indices(tmp_path: Path):
    # Long index lists take the path that releases the GIL while they are decoded.
    rows = (1 << 16) + 5
    path = tmp_path / "long.vortex"
    vx.io.write(vx.array(pa.table({"x": list(range(rows))})), str(path))
    indices = list(range(rows))
    result = vx.open(str(path)).scan(indices=vx.array(indices)).to_arrow_array(combine_chunks=True)
    assert isinstance(result, pa.StructArray)
    assert result.field("x").to_pylist() == indices
