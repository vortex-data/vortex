# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
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
        (pl.col("URL").str.contains("google", literal=True), ve.like(ve.column("URL"), "%google%")),
        (
            (
                (pl.col("Title").str.contains("Google", literal=True))
                & (~pl.col("URL").str.contains(".google.", literal=True))
                & (pl.col("SearchPhrase") != "")
            ),
            (
                ve.like(ve.column("Title"), "%Google%")
                & ve.not_(ve.like(ve.column("URL"), "%.google.%"))
                & (ve.column("SearchPhrase") != "")
            ),
        ),
        (pl.col("c") > 10000, ve.column("c") > 10000),
        (
            pl.col("EventDate") >= date(2013, 7, 1),
            ve.column("EventDate") >= ve.literal(vx.date("days"), (date(2013, 7, 1) - date(1970, 1, 1)).days),
        ),
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


def test_polars_binary_literals(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2], "x": [b"a", None, b"b"]})
    expr = pl.col("x") == b"a"
    path = tmp_path / "binary_literals.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [0]


def test_polars_is_null(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2], "x": [1, None, 3]})
    expr = pl.col("x").is_null()
    path = tmp_path / "is_null.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1]


def test_polars_boolean_not(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2], "x": [True, None, False]})
    expr = ~pl.col("x")
    path = tmp_path / "boolean_not.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [2]


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


def test_polars_fill_null_string_literal(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2], "x": ["a", None, "c"]})
    filled = pl.col("x").fill_null("z")
    assert polars_to_vortex(filled).serialize() == ve.fill_null(ve.column("x"), "z").serialize()
    expr = filled == "z"
    path = tmp_path / "fill_null_string.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1]


def test_datetime_predicate_pushdown(tmp_path):
    table = pa.table(
        {
            "id": [0, 1, 2],
            "value": pa.array(
                [datetime(2026, 9, day, tzinfo=UTC) for day in [16, 17, 18]],
                type=pa.timestamp("us", tz="UTC"),
            ),
        }
    )
    path = tmp_path / "datetimes.vortex"
    vx.io.write(vx.array(table), str(path))
    predicate = pl.col("value") >= datetime(2026, 9, 17, tzinfo=UTC)
    expected = pl.DataFrame(table).lazy().filter(predicate).collect()
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
    values = pa.array([datetime(2024, 3, 10, 2, 30), datetime(2024, 3, 10, 3, 30)], type=pa.timestamp("us"))
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
    expression = pl.col("dt").dt.replace_time_zone("America/New_York", ambiguous=pl.col("policy"), non_existent="null")
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


def _time_zone_scan(tmp_path, values, policy=None) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    columns = {"id": pa.array(range(len(values))), "dt": values}
    if policy is not None:
        columns["policy"] = pa.array(policy)
    path = tmp_path / "timezones.vortex"
    table = pa.table(columns)
    vx.io.write(vx.array(table), str(path))
    return pl.DataFrame(table).lazy(), vx.open(str(path)).to_polars()


# To unify operand types, Polars' optimizer adds widening casts with `NonStrict` or `Overflowing`
# options. Telling those apart from a lossy user `cast(strict=False)` needs the column types,
# which `polars_to_vortex` does not have, so such casts are not translated yet.
_COERCION_CAST_GAP = pytest.mark.xfail(strict=True, reason="coercion casts need the file schema")


def _assert_pushdown(tmp_path, frame: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    """Filter `frame` through a Vortex file and check the result matches Polars' own filter."""
    path = tmp_path / "pushdown.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected)
    return actual


@pytest.mark.parametrize(
    "expr, expected",
    [
        (pl.col("x") + 1 > 3, [2, 3]),
        (pl.col("x") - 1 > 1, [2, 3]),
        (pl.col("x") * 2 > 4, [2, 3]),
        # True division: integer division would make 3 / 2 == 1 and drop id 2.
        (pl.col("x") / 2 > 1, [2, 3]),
        (pl.col("x32") + 1 > 3, [2, 3]),
        pytest.param((pl.col("x") + pl.col("x32")) * 2 > 8, [2, 3], marks=_COERCION_CAST_GAP),
    ],
)
def test_polars_arithmetic(tmp_path, expr, expected):
    frame = pl.DataFrame(
        {"id": [0, 1, 2, 3, 4], "x": [1, 2, 3, 4, None], "x32": pl.Series([1, 2, 3, 4, None], dtype=pl.Int32)}
    )
    assert _assert_pushdown(tmp_path, frame, expr)["id"].to_list() == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        (pl.col("s").str.contains("bob", literal=True), [1, 6]),
        (pl.col("s").str.starts_with("bob"), [1, 6]),
        (pl.col("s").str.ends_with("e"), [0, 3, 4]),
        # `%` and `_` in the needle must match literally, not as LIKE wildcards.
        (pl.col("s").str.contains("%_", literal=True), [7]),
        (pl.col("s").str.starts_with("a_"), [8]),
        (~pl.col("s").str.contains("o", literal=True), [0, 3, 4, 7, 8, 9]),
    ],
)
def test_polars_string_matching(tmp_path, expr, expected):
    names = ["alice", "bob", "carol", "dave", "eve", None, "bobby", "100%_legit", "a_b", "axb"]
    frame = pl.DataFrame({"id": list(range(len(names))), "s": names})
    # "axb" must not match `starts_with("a_")`, which it would if `_` were a wildcard.
    assert _assert_pushdown(tmp_path, frame, expr)["id"].to_list() == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        (pl.col("x").is_in([2, 4, 99]), [1, 3]),
        pytest.param(pl.col("x32").is_in([2, 4]), [1, 3], marks=_COERCION_CAST_GAP),
        (pl.col("s").is_in(["a", "c"]), [0, 2]),
        (pl.col("x").is_in([]), []),
    ],
)
def test_polars_is_in(tmp_path, expr, expected):
    frame = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "x": [1, 2, 3, 4, None],
            "x32": pl.Series([1, 2, 3, 4, None], dtype=pl.Int32),
            "s": ["a", "b", "c", None, "d"],
        }
    )
    assert _assert_pushdown(tmp_path, frame, expr)["id"].to_list() == expected


def test_polars_naive_datetime_literal(tmp_path):
    # A naive datetime literal is serialized as a UTC literal cast to a naive datetime.
    frame = pl.DataFrame({"id": [0, 1, 2], "ts": [datetime(2020, 1, 1), datetime(2020, 1, 2), None]})
    assert _assert_pushdown(tmp_path, frame, pl.col("ts") > datetime(2020, 1, 1))["id"].to_list() == [1]


def test_polars_cast(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [1, 2, None, 3]})
    expr = pl.col("x").cast(pl.Float64) > 1.5
    assert _assert_pushdown(tmp_path, frame, expr)["id"].to_list() == [1, 3]


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(pl.col("s").str.contains("goo.*"), id="regex-contains"),
        pytest.param(pl.col("d") > timedelta(days=1), id="duration-literal"),
        pytest.param(pl.col("x").is_in([1, None]), id="null-in-is-in"),
        pytest.param(pl.col("x").cast(pl.Int8, strict=False) > 1, id="non-strict-cast"),
    ],
)
def test_unsupported_exprs(expr):
    with pytest.raises(NotImplementedError):
        polars_to_vortex(expr)
