# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Reads of a Vortex file through pyarrow, Polars and DuckDB give the same rows as the source data."""

import hashlib
from typing import Any, Literal

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import vortex as vx

LENGTH = 10_000
ROW = 5_000

Engine = Literal["auto", "streaming"]
ENGINES: list[Engine] = ["auto", "streaming"]


@pytest.fixture(
    scope="module",
    params=[("x",), ("x", "y"), ("x", "z"), ("x", "y", "z")],
    ids=["int", "int_str", "int_float", "int_str_float"],
)
def source(request: pytest.FixtureRequest) -> pa.Table:
    x = pa.array(range(LENGTH), type=pa.int64())
    arrays: list[pa.Array[Any] | pa.ChunkedArray[Any]] = [x]

    if "y" in request.param:
        arrays.append(
            pa.array([hashlib.md5(i.to_bytes(length=4), usedforsecurity=False).hexdigest() for i in range(LENGTH)])
        )
    if "z" in request.param:
        arrays.append(pc.sqrt(x))

    return pa.table(arrays, names=list(request.param))


@pytest.fixture(scope="module")
def vxf(source: pa.Table, tmp_path_factory: pytest.TempPathFactory) -> vx.VortexFile:
    path = str(tmp_path_factory.mktemp("engine_reads") / "data.vortex")
    vx.io.write(source, path)

    return vx.open(path)


@pytest.fixture
def conn(vxf: vx.VortexFile) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=":memory:")
    _ = conn.register("ds", vxf.to_dataset())
    return conn


def numeric_columns(source: pa.Table) -> list[str]:
    return [
        field.name for field in source.schema if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
    ]


def assert_rows_equal(actual: pa.Table, expected: pa.Table) -> None:
    # Vortex reads strings back as string_view, so compare values, not types.
    assert actual.column_names == expected.column_names
    assert actual.to_pylist() == expected.to_pylist()


def test_arrow_table_aggregation(source: pa.Table, vxf: vx.VortexFile) -> None:
    aggregations: list[tuple[str, Literal["mean"]]] = [(name, "mean") for name in numeric_columns(source)]
    actual = pa.concat_tables(x.to_arrow_table() for x in vxf.scan()).group_by([]).aggregate(aggregations)

    assert actual.to_pylist() == source.group_by([]).aggregate(aggregations).to_pylist()


@pytest.mark.parametrize("engine", ENGINES)
def test_polars_scan(source: pa.Table, vxf: vx.VortexFile, engine: Engine) -> None:
    actual = vxf.to_polars().collect(engine=engine).to_arrow()

    assert_rows_equal(actual, source)


@pytest.mark.parametrize("engine", ENGINES)
def test_polars_filter(source: pa.Table, vxf: vx.VortexFile, engine: Engine) -> None:
    lf = vxf.to_polars().filter(pl.col("x") >= pl.lit(ROW).cast(pl.Int64))
    actual = lf.collect(engine=engine).to_arrow()

    assert_rows_equal(actual, source.filter(pc.field("x") >= ROW))


@pytest.mark.parametrize("engine", ENGINES)
def test_polars_slice(source: pa.Table, vxf: vx.VortexFile, engine: Engine) -> None:
    actual = vxf.to_polars().slice(ROW, 1).collect(engine=engine).to_arrow()

    assert_rows_equal(actual, source.slice(ROW, 1))


@pytest.mark.parametrize("engine", ENGINES)
def test_polars_aggregation(source: pa.Table, vxf: vx.VortexFile, engine: Engine) -> None:
    actual = vxf.to_polars().mean().collect(engine=engine)

    for name in numeric_columns(source):
        assert actual[name][0] == pytest.approx(pc.mean(source[name]).as_py())


def test_duckdb_scan(source: pa.Table, conn: duckdb.DuckDBPyConnection) -> None:
    actual = conn.sql("select ds.x from ds").to_arrow_table()

    assert_rows_equal(actual, source.select(["x"]))


def test_duckdb_filter(source: pa.Table, conn: duckdb.DuckDBPyConnection) -> None:
    actual = conn.sql(f"select ds.x from ds where x >= {ROW}").to_arrow_table()

    assert_rows_equal(actual, source.select(["x"]).filter(pc.field("x") >= ROW))


def test_duckdb_offset(source: pa.Table, conn: duckdb.DuckDBPyConnection) -> None:
    actual = conn.sql(f"select ds.x from ds offset {ROW} limit 1").to_arrow_table()

    assert_rows_equal(actual, source.select(["x"]).slice(ROW, 1))


def test_duckdb_aggregation(source: pa.Table, conn: duckdb.DuckDBPyConnection) -> None:
    names = numeric_columns(source)
    aggregations = ",".join(f"avg(ds.{name}) as {name}" for name in names)
    (actual,) = conn.sql(f"select {aggregations} from ds").fetchall()

    for name, value in zip(names, actual, strict=True):
        assert value == pytest.approx(pc.mean(source[name]).as_py())
