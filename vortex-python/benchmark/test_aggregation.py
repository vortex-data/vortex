# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import Literal

import duckdb
import pyarrow as pa
import pytest
from pyarrow.types import is_floating, is_integer
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

import vortex as vx


def _has_mean(t: pa.DataType) -> bool:
    return is_integer(t) or is_floating(t)


@pytest.mark.benchmark(group="aggregation", disable_gc=True)
def test_arrow_table_aggregation(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    aggregations: list[tuple[str, Literal["mean"]]] = [
        (field.name, "mean")
        for field in vxf.dtype.to_arrow_schema()  # pyright: ignore[reportUnknownVariableType]
        if _has_mean(field.type)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    ]
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan()).group_by([]).aggregate(aggregations))


@pytest.mark.benchmark(group="aggregation", disable_gc=True)
def test_polars_aggregation(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.mean().collect().to_arrow())


@pytest.mark.benchmark(group="aggregation", disable_gc=True)
def test_polars_streaming_aggregation(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.mean().collect(engine="streaming").to_arrow())


@pytest.mark.benchmark(group="aggregation", disable_gc=True)
def test_duckdb_aggregation(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    conn = duckdb.connect(database=":memory:")
    ds = vxf.to_dataset()
    _ = conn.register("ds", ds)
    aggregations = ",".join(
        [f"avg(ds.{field.name}) as {field.name}" for field in vxf.dtype.to_arrow_schema() if _has_mean(field.type)]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]
    )
    print(aggregations)
    query = f"select {aggregations} from ds"
    benchmark(lambda: conn.sql(query).to_arrow_table())
