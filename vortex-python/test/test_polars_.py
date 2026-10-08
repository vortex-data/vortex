# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from datetime import time
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


def test_polars_struct_field(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [{"a": 1}, None, {"a": 3}, {"a": None}]})
    expr = pl.col("x").struct.field("a") >= 2
    path = tmp_path / "struct_field.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [2]


def _assert_struct_field_scan(tmp_path, array, predicate, expected):
    table = pa.table({"id": range(len(array)), "x": array})
    polars_result = pl.from_arrow(table).lazy().filter(predicate).collect()
    path = tmp_path / "struct_parent_nulls.vortex"
    vx.io.write(vx.array(table), str(path))
    vortex_result = vx.open(str(path)).to_polars().filter(predicate).collect()
    assert_frame_equal(polars_result, expected)
    assert_frame_equal(vortex_result, expected)


def test_polars_struct_field_null_parent(tmp_path):
    leaf = pa.array([10, 20])
    parent = pa.StructArray.from_arrays([leaf], names=["value"], mask=pa.array([True, False]))
    assert parent.field("value").to_pylist() == [10, 20]
    predicate = pl.col("x").struct.field("value") >= 0
    expected = pl.DataFrame({"id": [1], "x": [{"value": 20}]})
    _assert_struct_field_scan(tmp_path, parent, predicate, expected)


def test_polars_struct_field_null_inner_parent(tmp_path):
    leaf = pa.array([10, 20])
    inner = pa.StructArray.from_arrays([leaf], names=["value"], mask=pa.array([True, False]))
    outer = pa.StructArray.from_arrays([inner], names=["child"])
    assert outer.field("child").field("value").to_pylist() == [10, 20]
    predicate = pl.col("x").struct.field("child").struct.field("value") >= 0
    expected = pl.DataFrame({"id": [1], "x": [{"child": {"value": 20}}]})
    _assert_struct_field_scan(tmp_path, outer, predicate, expected)


def test_polars_struct_field_null_outer_parent(tmp_path):
    leaf = pa.array([10, 20])
    inner = pa.StructArray.from_arrays([leaf], names=["value"])
    outer = pa.StructArray.from_arrays([inner], names=["child"], mask=pa.array([True, False]))
    assert outer.field("child").field("value").to_pylist() == [10, 20]
    predicate = pl.col("x").struct.field("child").struct.field("value") >= 0
    expected = pl.DataFrame({"id": [1], "x": [{"child": {"value": 20}}]})
    _assert_struct_field_scan(tmp_path, outer, predicate, expected)


def test_polars_struct_field_null_middle_parent(tmp_path):
    leaf = pa.array([10, 20])
    inner = pa.StructArray.from_arrays([leaf], names=["value"])
    middle = pa.StructArray.from_arrays([inner], names=["child"], mask=pa.array([True, False]))
    outer = pa.StructArray.from_arrays([middle], names=["child"])
    assert outer.field("child").field("child").field("value").to_pylist() == [10, 20]
    predicate = pl.col("x").struct.field("child").struct.field("child").struct.field("value") >= 0
    expected = pl.DataFrame({"id": [1], "x": [{"child": {"child": {"value": 20}}}]})
    _assert_struct_field_scan(tmp_path, outer, predicate, expected)


def test_polars_struct_field_null_parents_and_leaf(tmp_path):
    leaf = pa.array([10, 20, 30, None, 50])
    inner = pa.StructArray.from_arrays([leaf], names=["value"], mask=pa.array([True, False, True, False, False]))
    outer = pa.StructArray.from_arrays([inner], names=["child"], mask=pa.array([False, True, True, False, False]))
    assert outer.field("child").field("value").to_pylist() == [10, 20, 30, None, 50]
    predicate = pl.col("x").struct.field("child").struct.field("value") >= 0
    expected = pl.DataFrame({"id": [4], "x": [{"child": {"value": 50}}]})
    _assert_struct_field_scan(tmp_path, outer, predicate, expected)
