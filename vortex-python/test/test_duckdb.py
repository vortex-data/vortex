# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Test for timestamp filtering with DuckDB via vortex extension and PyArrow dataset interface."""

from datetime import datetime

import duckdb
import pyarrow as pa
import pytest

import vortex as vx


@pytest.mark.xfail(reason="is_not_null not yet implemented in substrait conversion")
def test_duckdb_via_substrait(tmp_path):
    con = duckdb.connect()

    arr = pa.array([datetime(2024, 1, 1), datetime(2024, 6, 15), datetime(2024, 12, 31)])
    table = pa.table({"ts": arr})
    vx.io.write(table, '/tmp/test_timestamp.vortex')

    ds = vx.open("/tmp/test_timestamp.vortex").to_dataset()  # noqa: F841 - used by duckdb
    result = con.execute("SELECT * FROM ds WHERE ts > '2024-06-01'").fetchall()
    print(result)
