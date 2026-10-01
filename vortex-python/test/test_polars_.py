# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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


def test_datetime_predicate_pushdown(tmp_path):
    table = pa.table(
        {
            "id": [0, 1, 2],
            "value": pa.array(
                [datetime(2026, 9, day, tzinfo=timezone.utc) for day in [16, 17, 18]],
                type=pa.timestamp("us", tz="UTC"),
            ),
        }
    )
    path = tmp_path / "datetimes.vortex"
    vx.io.write(vx.array(table), str(path))
    predicate = pl.col("value") >= datetime(2026, 9, 17, tzinfo=timezone.utc)
    expected = pl.from_arrow(table).lazy().filter(predicate).collect()
    result = vx.open(str(path)).to_polars().filter(predicate).collect()
    assert_frame_equal(result, expected)
    assert result["id"].to_list() == [1, 2]


@pytest.mark.parametrize("unit, scale", [("ms", 1_000), ("us", 1_000_000), ("ns", 1_000_000_000)])
@pytest.mark.parametrize("source_zone", [None, "UTC", "Europe/London"])
@pytest.mark.parametrize("target_zone", [None, "UTC", "America/New_York"])
def test_replace_time_zone_columns(tmp_path, unit, scale, source_zone, target_zone):
    seconds = [1_705_320_000, 1_721_041_200 if source_zone == "Europe/London" else 1_721_044_800, None]
    values = pa.array(
        [None if value is None else value * scale for value in seconds],
        type=pa.timestamp(unit, tz=source_zone),
    )
    reference, frame = _time_zone_scan(tmp_path, values)
    expression = pl.col("dt").dt.replace_time_zone(target_zone)
    threshold = pl.lit(
        datetime(2024, 7, 15, 12, tzinfo=None if target_zone is None else ZoneInfo(target_zone)),
        dtype=pl.Datetime(unit, target_zone),
    )
    expected_frame = reference.filter(expression >= threshold).collect()
    actual = frame.filter(expression >= threshold).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1]


@pytest.mark.parametrize(
    "ambiguous, fold, expected",
    [
        ("earliest", 0, [0]),
        ("earliest", 1, []),
        ("latest", 0, []),
        ("latest", 1, [0]),
        ("null", 0, []),
        ("null", 1, []),
    ],
)
def test_replace_time_zone_ambiguous(tmp_path, ambiguous, fold, expected):
    values = pa.array([datetime(2024, 11, 3, 1, 30), None], type=pa.timestamp("us"))
    reference, frame = _time_zone_scan(tmp_path, values)
    expression = pl.col("dt").dt.replace_time_zone("America/New_York", ambiguous=ambiguous)
    threshold = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=fold)
    expected_frame = reference.filter(expression == threshold).collect()
    actual = frame.filter(expression == threshold).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == expected


@pytest.mark.parametrize("fold, expected", [(0, [0]), (1, [1])])
def test_replace_time_zone_policy_column(tmp_path, fold, expected):
    values = pa.array([datetime(2024, 11, 3, 1, 30)] * 4, type=pa.timestamp("us"))
    reference, frame = _time_zone_scan(tmp_path, values, ["earliest", "latest", "null", None])
    expression = pl.col("dt").dt.replace_time_zone("America/New_York", ambiguous=pl.col("policy"))
    threshold = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=fold)
    expected_frame = reference.filter(expression == threshold).collect()
    actual = frame.filter(expression == threshold).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == expected


def test_replace_time_zone_non_existent_null(tmp_path):
    values = pa.array(
        [datetime(2024, 3, 10, 2, 30), datetime(2024, 3, 10, 3, 30)], type=pa.timestamp("us")
    )
    reference, frame = _time_zone_scan(tmp_path, values)
    expression = pl.col("dt").dt.replace_time_zone("America/New_York", non_existent="null")
    threshold = datetime(2024, 3, 10, 3, 30, tzinfo=ZoneInfo("America/New_York"))
    expected_frame = reference.filter(expression == threshold).collect()
    actual = frame.filter(expression == threshold).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1]


@pytest.mark.parametrize("value", [datetime(2024, 11, 3, 1, 30), datetime(2024, 3, 10, 2, 30)])
def test_replace_time_zone_raises(tmp_path, value):
    reference, frame = _time_zone_scan(tmp_path, pa.array([value], type=pa.timestamp("us")))
    expression = pl.col("dt").dt.replace_time_zone("America/New_York")
    threshold = datetime(2024, 1, 1, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(pl.exceptions.ComputeError):
        reference.filter(expression >= threshold).collect()
    with pytest.raises((RuntimeError, pl.exceptions.ComputeError), match="ambiguous|gap|fold"):
        frame.filter(expression >= threshold).collect()


def test_replace_time_zone_maps_to_native_expression():
    expression = pl.col("dt").dt.replace_time_zone(
        "America/New_York", ambiguous=pl.col("policy"), non_existent="null"
    )
    expected = ve.replace_time_zone(
        ve.column("dt"), "America/New_York", ambiguous=ve.column("policy"), non_existent="null"
    )
    assert polars_to_vortex(expression).serialize() == expected.serialize()


def test_replace_time_zone_same_zone_during_fold(tmp_path):
    # 2024-11-03 06:30 UTC is the second 01:30 in New York.
    values = pa.array([1_730_615_400_000_000], type=pa.timestamp("us", tz="America/New_York"))
    reference, frame = _time_zone_scan(tmp_path, values)
    expression = pl.col("dt").dt.replace_time_zone("America/New_York")
    threshold = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1)
    expected_frame = reference.filter(expression == threshold).collect()
    actual = frame.filter(expression == threshold).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [0]


def _time_zone_scan(tmp_path, values, policy=None):
    columns = {"id": pa.array(range(len(values))), "dt": values}
    if policy is not None:
        columns["policy"] = pa.array(policy)
    path = tmp_path / "timezones.vortex"
    table = pa.table(columns)
    vx.io.write(vx.array(table), str(path))
    return pl.from_arrow(table).lazy(), vx.open(str(path)).to_polars()
