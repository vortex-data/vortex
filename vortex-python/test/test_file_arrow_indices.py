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


def test_arrow_indices_use_scan_filter_limit_validation(indexed_file):
    file, _ = indexed_file
    reader = file.to_arrow(indices=vx.array([0, 2, 5, 8]), expr=ve.column("id") >= 4, limit=1)
    with pytest.raises(pa.ArrowInvalid, match="doesn't support scans with both a filter and a limit"):
        reader.read_all()


@pytest.mark.parametrize("indices", [[2, 1], [1, 1], [None, 1]])
def test_arrow_indices_use_scan_validation(indexed_file, indices):
    file, _ = indexed_file
    indices = vx.array(pa.array(indices, type=pa.uint64()))
    with pytest.raises(RuntimeError):
        file.to_arrow(indices=indices)
