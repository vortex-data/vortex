# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Test for timestamp filtering with DuckDB via vortex extension and PyArrow dataset interface."""

from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa

import vortex as vx


def test_duckdb_via_substrait(tmp_path: Path) -> None:
    con = duckdb.connect()

    arr = pa.array([datetime(2024, 1, 1), datetime(2024, 6, 15), datetime(2024, 12, 31)])
    table = pa.table({"ts": arr})
    path = str(tmp_path / "test_timestamp.vortex")
    vx.io.write(table, path)

    ds = vx.open(path).to_dataset()  # noqa: F841  # pyright: ignore[reportUnusedVariable] - used by duckdb via SQL
    result = con.execute("SELECT * FROM ds WHERE ts > '2024-06-01'").fetchall()
    assert len(result) == 2
    print(result)
