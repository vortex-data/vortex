# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

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


def test_polars_fill_null(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [1, None, None, 5], "y": [4, 5, None, 6]})
    expr = pl.col("x").fill_null(pl.col("y")) == 5
    path = tmp_path / "fill_null.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1, 3]


@pytest.mark.parametrize("fill_value, expected_ids", [(5, [1, 2, 3]), (0, [1, 2]), (-1, [1, 2])])
def test_polars_fill_null_literal(tmp_path, fill_value, expected_ids):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [1, None, None, 5]})
    filled = pl.col("x").fill_null(fill_value)
    assert polars_to_vortex(filled).serialize() == ve.fill_null(ve.column("x"), fill_value).serialize()
    predicate = filled == fill_value
    path = tmp_path / "fill_null_literal.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(predicate).collect()
    actual = vx.open(str(path)).to_polars().filter(predicate).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == expected_ids
