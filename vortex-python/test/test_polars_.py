# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

import polars as pl
import pyarrow as pa
import pytest

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
def test_exprs(polars: pl.Expr, vortex: ve.Expr):
    # Dump the clickbench filters
    assert polars_to_vortex(polars) == vortex


@pytest.fixture(scope="module")
def vxf(tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "polars_test.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    if not os.path.exists(fname):  # pyright: ignore[reportUnknownArgumentType]
        a = pa.array([{"index": x, "value": math.sqrt(x)} for x in range(1_000_000)])
        vx.io.write(vx.compress(vx.array(a)), str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname), without_segment_cache=True)  # pyright: ignore[reportUnknownArgumentType]


def test_to_polars_with_limit(vxf: vx.VortexFile):
    df = vxf.to_polars().limit(100).collect()
    assert len(df) == 100


def test_to_polars_with_filter(vxf: vx.VortexFile):
    df = vxf.to_polars().filter(pl.col("index") < 500).collect()
    assert len(df) == 500
    assert df["index"].to_list() == list(range(500))


def test_to_polars_with_projection(vxf: vx.VortexFile):
    df = vxf.to_polars().select("index").limit(10).collect()
    assert df.columns == ["index"]
    assert len(df) == 10


def test_to_polars_with_projection_and_filter(vxf: vx.VortexFile):
    df = vxf.to_polars().select("index", "value").filter(pl.col("index") < 100).collect()
    assert df.columns == ["index", "value"]
    assert len(df) == 100


@pytest.fixture(scope="session")
def small_vxf(tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "polars_small.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    table = pa.table({"a": pa.array([1, 2, 3]), "s": pa.array(["xa", "yb", "zc"])})
    vx.io.write(table, str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname))  # pyright: ignore[reportUnknownArgumentType]


def test_to_polars_with_unsupported_predicate(small_vxf: vx.VortexFile):
    # Regression test: predicates that cannot be converted to Vortex expressions (e.g.
    # str.contains) used to fail the whole query instead of being applied by Polars.
    df = small_vxf.to_polars().filter(pl.col("s").str.contains("a")).collect()
    assert df.to_dicts() == [{"a": 1, "s": "xa"}]


def test_to_polars_with_unsupported_predicate_and_limit(small_vxf: vx.VortexFile):
    # The row limit must apply after a fallback filter, not before.
    df = small_vxf.to_polars().filter(pl.col("s").str.contains("a") | (pl.col("a") > 1)).head(2).collect()
    assert len(df) == 2
