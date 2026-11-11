# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Float filters pushed into Vortex keep the rows that the engine itself keeps.

PyArrow compares floats with IEEE 754 semantics, and Polars makes all NaNs equal and greater than
all other values. Both make ``-0.0`` equal to ``+0.0``. Vortex orders floats by their bits, so each
pushed comparison must carry the semantics of its engine.
"""

import math
from collections.abc import Callable
from typing import Any, cast

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest

import vortex as vx

NAN = math.nan
NEG_NAN = -math.nan

VALUES = [-0.0, 0.0, NAN, NEG_NAN, 1.0, -1.0, math.inf, -math.inf, None]


@pytest.fixture(scope="module")
def source() -> pa.Table:
    return pa.table({"x": pa.array(VALUES, type=pa.float64()), "i": pa.array(range(len(VALUES)), type=pa.int64())})


@pytest.fixture(scope="module")
def vxf(source: pa.Table, tmp_path_factory: pytest.TempPathFactory) -> vx.VortexFile:
    path = str(tmp_path_factory.mktemp("float_filters") / "data.vortex")
    vx.io.write(source, path)

    return vx.open(path)


def rows(table: pa.Table) -> list[int]:
    return sorted(cast(list[int], table.column("i").to_pylist()))


ARROW_FILTERS: list[pc.Expression] = [
    pc.field("x") == 0.0,
    pc.field("x") == -0.0,
    pc.field("x") != 0.0,
    pc.field("x") == NAN,
    pc.field("x") != NAN,
    pc.field("x") > 0.0,
    pc.field("x") >= -0.0,
    pc.field("x") < 0.0,
    pc.field("x") <= 0.0,
    pc.field("x") > NAN,
    pc.field("x") < NAN,
    pc.field("x") > 1.0,
    pc.field("x").isin([0.0, NAN]),
]


@pytest.mark.parametrize("predicate", ARROW_FILTERS, ids=str)
def test_arrow_float_filters(source: pa.Table, vxf: vx.VortexFile, predicate: pc.Expression) -> None:
    expected = rows(ds.dataset(source).to_table(filter=predicate))
    actual = rows(vxf.to_dataset().to_table(filter=predicate))

    assert actual == expected


POLARS_FILTERS: list[Callable[[pl.Expr], pl.Expr]] = [
    lambda x: x == 0.0,
    lambda x: x == -0.0,
    lambda x: x != 0.0,
    lambda x: x == NAN,
    lambda x: x != NAN,
    lambda x: x > 0.0,
    lambda x: x >= -0.0,
    lambda x: x < 0.0,
    lambda x: x <= 0.0,
    lambda x: x >= NAN,
    lambda x: x < NAN,
    lambda x: x > math.inf,
    lambda x: x.is_between(-0.0, NAN),
    lambda x: x.is_in([-0.0, NAN]),
    lambda x: x.is_in([0.0, None], nulls_equal=True),
    lambda x: x.is_between(NAN, math.inf),
    lambda x: x.is_between(-math.inf, NAN, closed="none"),
    lambda x: (x > -math.inf) & (x < NAN),
    lambda x: (x < NAN) & (x.abs() > 0.5),
    lambda x: (x.cast(pl.Float32) == pl.lit(math.inf, pl.Float32)) | (x == NAN),
]


@pytest.mark.parametrize("predicate", POLARS_FILTERS, ids=lambda f: str(f(pl.col("x"))))
def test_polars_float_filters(source: pa.Table, vxf: vx.VortexFile, predicate: Callable[[pl.Expr], Any]) -> None:
    expected = rows(pl.DataFrame(source).lazy().filter(predicate(pl.col("x"))).collect().to_arrow())
    actual = rows(vxf.to_polars().filter(predicate(pl.col("x"))).collect().to_arrow())

    assert actual == expected
