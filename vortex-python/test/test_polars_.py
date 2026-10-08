# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from datetime import date, time
from decimal import Decimal

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

import vortex as vx
import vortex.expr as ve
from vortex.polars_ import polars_to_vortex


@pytest.mark.parametrize(
    "polars, vortex",
    [
        (pl.col("AdvEngineID") != 0, ve.column("AdvEngineID") != 0),
        (pl.col("MobilePhoneModel") != "", ve.column("MobilePhoneModel") != ""),
        (pl.col("UserID") == 435090932899640449, ve.column("UserID") == 435090932899640449),
        # (pl.col("URL").str.contains("google"), ve.column("URL").str.contains("google")),
        # (
        #     (
        #         (pl.col("Title").str.contains("Google"))
        #         & (~pl.col("URL").str.contains(".google."))
        #         & (pl.col("SearchPhrase") != "")
        #     ),
        #     (
        #         (ve.column("Title").str.contains("Google"))
        #         & (~ve.column("URL").str.contains(".google."))
        #         & (ve.column("SearchPhrase") != "")
        #     ),
        # ),
        (pl.col("c") > 10000, ve.column("c") > 10000),
        #        (pl.col("EventDate") >= date(2013, 7, 1), ve.column("EventDate") >= date(2013, 7, 1)),
    ],
)
def test_exprs(polars: pl.Expr, vortex: ve.Expr) -> None:
    assert polars_to_vortex(polars).serialize() == vortex.serialize()


@pytest.fixture(scope="module")
def vxf(tmpdir_factory) -> vx.VortexFile:
    fname = tmpdir_factory.mktemp("data") / "polars_test.vortex"

    if not os.path.exists(fname):
        a = pa.array([{"index": x, "value": math.sqrt(x)} for x in range(1_000_000)])
        vx.io.write(vx.compress(vx.array(a)), str(fname))
    return vx.open(str(fname), without_segment_cache=True)


def test_to_polars_with_limit(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().limit(100).collect()
    assert len(df) == 100


def test_to_polars_with_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().filter(pl.col("index") < 500).collect()
    assert len(df) == 500
    assert df["index"].to_list() == list(range(500))


def test_to_polars_with_projection(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().select("index").limit(10).collect()
    assert df.columns == ["index"]
    assert len(df) == 10


def test_to_polars_with_projection_and_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().select("index", "value").filter(pl.col("index") < 100).collect()
    assert df.columns == ["index", "value"]
    assert len(df) == 100


def test_polars_time_literals(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [time(11), time(12), None, time(13)]})
    expr = pl.col("x") >= time(12)
    path = tmp_path / "time_literals.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1, 3]


def test_polars_conditional(tmp_path):
    frame = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "p": [True, False, None, False],
            "x": [1, 2, 3, 4],
            "y": [4, 5, 6, None],
        }
    )
    expr = pl.when(pl.col("p")).then(pl.col("x")).otherwise(pl.col("y")) >= 5
    path = tmp_path / "conditional.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1, 2]


def test_polars_decimal_literals(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [Decimal("1.24"), Decimal("1.25"), None, Decimal("1.26")]})
    expr = pl.col("x") >= Decimal("1.25")
    path = tmp_path / "decimal_literals.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1, 3]


@pytest.mark.parametrize(
    "closed, expected",
    [("both", [0, 1, 2]), ("left", [0, 1]), ("right", [1, 2]), ("none", [1])],
)
def test_polars_is_between(tmp_path, closed, expected):
    frame = pl.DataFrame({"id": list(range(6)), "x": [1, 2, 3, 4, None, 2], "l": [1, 1, 1, 1, 1, None], "u": [3] * 6})
    expr = pl.col("x").is_between(pl.col("l"), pl.col("u"), closed=closed)
    path = tmp_path / "between.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == expected


@pytest.mark.parametrize(
    "arrow_type, threshold",
    [(pa.uint8(), 50), (pa.uint16(), 500), (pa.uint32(), 500), (pa.uint64(), 500)],
)
def test_unsigned_predicate_pushdown(tmp_path, arrow_type, threshold):
    table = pa.table(
        {
            "id": [0, 1, 2],
            "value": pa.array([threshold - 1, threshold, threshold + 1], type=arrow_type),
        }
    )
    path = tmp_path / "unsigned.vortex"
    vx.io.write(vx.array(table), str(path))
    expr = pl.col("value") >= threshold
    expected = pl.DataFrame(table).lazy().filter(expr).collect()
    result = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(result, expected)
    assert result["id"].to_list() == [1, 2]


def test_is_not_null_predicate_pushdown(tmp_path):
    table = pa.table({"id": [0, 1, 2], "value": ["first", None, "last"]})
    path = tmp_path / "non_null.vortex"
    vx.io.write(vx.array(table), str(path))
    expr = pl.col("value").is_not_null()
    expected = pl.DataFrame(table).lazy().filter(expr).collect()
    result = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(result, expected)
    assert result["id"].to_list() == [0, 2]


def test_polars_date_literals(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [date(2024, 1, 1), date(2024, 1, 2), None, date(2024, 1, 3)]})
    expr = pl.col("x") >= date(2024, 1, 2)
    path = tmp_path / "date_literals.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1, 3]
