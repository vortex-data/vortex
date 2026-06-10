# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Fuzz strategies are inherently dynamically typed, so relax the strict pyright rules here.
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnusedCallResult=false, reportUnusedFunction=false, reportUnreachable=false

"""Property-based fuzz tests for the Vortex Python file IO API.

These tests generate random (but reproducible) Arrow tables, write them to disk as
Vortex files, read them back, and compare the round-tripped values via Arrow.

Two Hypothesis profiles are registered:

- ``default``: a small, fully deterministic smoke test that runs in the per-PR CI gate
  and local test runs.
- ``fuzz``: a randomized exploration used by the scheduled fuzz workflow
  (``.github/workflows/fuzz.yml``). Run it locally via::

      VORTEX_FUZZ_PROFILE=fuzz VORTEX_FUZZ_MAX_EXAMPLES=2000 pytest test/test_fuzz_file_roundtrip.py
"""

import datetime
import decimal as decimal_module
import math
import os
import tempfile
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import vortex as vx

_HEALTH_CHECKS = [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much]

settings.register_profile(
    "default",
    max_examples=25,
    derandomize=True,
    deadline=None,
    suppress_health_check=_HEALTH_CHECKS,
)
settings.register_profile(
    "fuzz",
    max_examples=int(os.environ.get("VORTEX_FUZZ_MAX_EXAMPLES", "1000")),
    deadline=None,
    suppress_health_check=_HEALTH_CHECKS,
    print_blob=True,
)
settings.load_profile(os.environ.get("VORTEX_FUZZ_PROFILE", "default"))

# ---------------------------------------------------------------------------
# Arrow type strategies
# ---------------------------------------------------------------------------

INT_TYPES: list[pa.DataType] = [
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
]

FLAT_TYPES: list[pa.DataType] = INT_TYPES + [
    pa.bool_(),
    pa.null(),
    pa.float32(),
    pa.float64(),
    pa.string(),
    pa.large_string(),
    pa.string_view(),
    pa.binary(),
    pa.large_binary(),
    pa.binary_view(),
    # pa.binary(3) (fixed-size binary) excluded: DType::from_arrow panics with
    # `unimplemented!()` (vortex-array/src/dtype/arrow.rs) instead of raising a clean error.
    # See https://github.com/vortex-data/vortex/issues/8346.
    pa.decimal128(38, 9),
    pa.decimal128(5, 2),
    pa.date32(),
    pa.date64(),
    pa.time32("s"),
    pa.time32("ms"),
    pa.time64("us"),
    pa.time64("ns"),
    pa.timestamp("s"),
    pa.timestamp("ms"),
    pa.timestamp("us"),
    pa.timestamp("ns"),
    pa.timestamp("us", tz="UTC"),
    # pa.duration(..) excluded: DType::from_arrow panics with `unimplemented!()`
    # (vortex-array/src/dtype/arrow.rs) instead of raising a clean error.
    # See https://github.com/vortex-data/vortex/issues/8346.
    pa.dictionary(pa.int32(), pa.string()),
]


def _nested_types(inner: st.SearchStrategy[pa.DataType]) -> st.SearchStrategy[pa.DataType]:
    return st.one_of(
        inner.map(pa.list_),
        inner.map(pa.large_list),
        st.tuples(inner, st.integers(1, 4)).map(lambda t: pa.list_(t[0], t[1])),  # fixed-size list
        st.lists(
            st.tuples(st.sampled_from("abcdefg"), inner),
            min_size=1,
            max_size=3,
            unique_by=lambda kv: kv[0],
        ).map(lambda fields: pa.struct([pa.field(n, t) for n, t in fields])),
    )


flat_type_st = st.sampled_from(FLAT_TYPES)
arrow_type_st = st.one_of(flat_type_st, _nested_types(flat_type_st), _nested_types(_nested_types(flat_type_st)))


# ---------------------------------------------------------------------------
# Value strategies for a given Arrow type
# ---------------------------------------------------------------------------


def _int_bounds(t: pa.DataType) -> tuple[int, int]:
    bits = t.bit_width
    if pa.types.is_signed_integer(t):
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def value_strategy(t: pa.DataType) -> st.SearchStrategy[Any]:
    """A strategy producing python values valid for the Arrow type ``t`` (excluding nulls)."""
    if pa.types.is_null(t):
        return st.none()
    if pa.types.is_boolean(t):
        return st.booleans()
    if pa.types.is_integer(t):
        lo, hi = _int_bounds(t)
        return st.integers(lo, hi)
    if pa.types.is_float32(t):
        return st.floats(width=32, allow_nan=True, allow_infinity=True)
    if pa.types.is_float64(t):
        return st.floats(allow_nan=True, allow_infinity=True)
    if pa.types.is_string(t) or pa.types.is_large_string(t) or t == pa.string_view():
        return st.text(max_size=12)
    if t == pa.binary_view() or pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return st.binary(max_size=12)
    if pa.types.is_fixed_size_binary(t):
        return st.binary(min_size=t.byte_width, max_size=t.byte_width)
    if pa.types.is_decimal(t):
        # An explicit context, since scaleb rounds through the default 28-digit context.
        ctx = decimal_module.Context(prec=t.precision + 1)
        max_unscaled = 10**t.precision - 1
        return st.integers(-max_unscaled, max_unscaled).map(lambda u: Decimal(u).scaleb(-t.scale, context=ctx))
    if pa.types.is_date(t):
        return st.dates()
    if pa.types.is_time(t):
        resolutions = {"s": "seconds", "ms": "milliseconds", "us": "microseconds", "ns": "microseconds"}
        return st.times().map(lambda v: _truncate_time(v, resolutions[t.unit]))
    if pa.types.is_timestamp(t):
        tz = datetime.UTC if t.tz is not None else None
        # timestamp[ns] only covers 1677-09-21..2262-04-11.
        dt = st.datetimes(
            min_value=datetime.datetime(1700, 1, 1) if t.unit == "ns" else datetime.datetime(1800, 1, 1),
            max_value=datetime.datetime(2262, 1, 1) if t.unit == "ns" else datetime.datetime(2300, 1, 1),
        )
        if t.unit in ("s", "ms"):
            dt = dt.map(lambda v: _truncate_datetime(v, t.unit))
        if tz is not None:
            dt = dt.map(lambda v: v.replace(tzinfo=tz))
        return dt
    if pa.types.is_duration(t):
        return st.integers(-(2**40), 2**40).map(lambda us: datetime.timedelta(microseconds=us))
    if pa.types.is_dictionary(t):
        return value_strategy(t.value_type)
    if pa.types.is_fixed_size_list(t):
        return st.lists(nullable_value_strategy(t.value_type), min_size=t.list_size, max_size=t.list_size)
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return st.lists(nullable_value_strategy(t.value_type), max_size=4)
    if pa.types.is_struct(t):
        return st.fixed_dictionaries({f.name: nullable_value_strategy(f.type) for f in t})
    raise NotImplementedError(f"no value strategy for {t}")


def nullable_value_strategy(t: pa.DataType) -> st.SearchStrategy[Any]:
    if pa.types.is_null(t):
        return st.none()
    return st.none() | value_strategy(t)


def _truncate_time(v: datetime.time, resolution: str) -> datetime.time:
    if resolution == "seconds":
        return v.replace(microsecond=0)
    if resolution == "milliseconds":
        return v.replace(microsecond=(v.microsecond // 1000) * 1000)
    return v


def _truncate_datetime(v: datetime.datetime, unit: str) -> datetime.datetime:
    if unit == "s":
        return v.replace(microsecond=0)
    return v.replace(microsecond=(v.microsecond // 1000) * 1000)


# ---------------------------------------------------------------------------
# Table strategy
# ---------------------------------------------------------------------------


@st.composite
def arrow_tables(draw: st.DrawFn) -> pa.Table:
    ncols = draw(st.integers(1, 3))
    names = [f"c{i}" for i in range(ncols)]
    types = [draw(arrow_type_st) for _ in range(ncols)]
    nrows = draw(st.integers(0, 40))

    columns = []
    for t in types:
        values = draw(st.lists(nullable_value_strategy(t), min_size=nrows, max_size=nrows))
        columns.append(pa.array(values, type=t))

    table = pa.table(dict(zip(names, columns)))

    # Sometimes re-chunk so the writer sees multiple batches.
    # TODO(https://github.com/vortex-data/vortex/issues/8349): sliced fixed_size_list<struct>
    # arrays panic in from_arrow with "end <= self.len()", so don't slice those.
    if nrows > 1 and not any(_has_fsl_of_struct(t) for t in types) and draw(st.booleans()):
        split = draw(st.integers(1, nrows - 1))
        table = pa.concat_tables([table.slice(0, split), table.slice(split)])
    return table


def _has_fsl_of_struct(t: pa.DataType) -> bool:
    if pa.types.is_fixed_size_list(t) and pa.types.is_struct(t.value_type):
        return True
    return any(_has_fsl_of_struct(t.field(i).type) for i in range(t.num_fields))


@st.composite
def patterned_tables(draw: st.DrawFn) -> pa.Table:
    """Tables with value patterns (constant/sorted/low-cardinality runs) that trigger the
    specialized compression encodings random data rarely reaches."""
    ncols = draw(st.integers(1, 3))
    nrows = draw(st.integers(1, 1000))

    columns = {}
    for i in range(ncols):
        t = draw(flat_type_st)
        pattern = draw(st.sampled_from(["random", "constant", "sorted", "low_cardinality"]))
        base = nullable_value_strategy(t)
        if pattern == "constant":
            values = [draw(base)] * nrows
        elif pattern == "sorted" and pa.types.is_integer(t):
            values = sorted(draw(st.lists(value_strategy(t), min_size=nrows, max_size=nrows)))
        elif pattern == "low_cardinality":
            pool = draw(st.lists(base, min_size=1, max_size=4))
            values = draw(st.lists(st.sampled_from(pool), min_size=nrows, max_size=nrows))
        else:
            values = draw(st.lists(base, min_size=nrows, max_size=nrows))
        columns[f"c{i}"] = pa.array(values, type=t)

    return pa.table(columns)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def values_equal(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isnan(expected) or math.isnan(actual):
            return math.isnan(expected) and math.isnan(actual)
        return expected == actual
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(values_equal(e, a) for e, a in zip(expected, actual))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(values_equal(v, actual[k]) for k, v in expected.items())
        )
    return bool(expected == actual)


def assert_rows_equal(expected_rows: list[dict[str, Any]], actual: pa.Table) -> None:
    actual_rows = actual.to_pylist()
    assert len(actual_rows) == len(expected_rows), f"row count {len(actual_rows)} != {len(expected_rows)}"
    for i, (e, a) in enumerate(zip(expected_rows, actual_rows)):
        assert values_equal(e, a), f"row {i}: expected {e!r}, got {a!r}"


def assert_tables_equal(expected: pa.Table, actual: pa.Table) -> None:
    assert actual.num_rows == expected.num_rows, f"row count {actual.num_rows} != {expected.num_rows}"
    assert set(actual.column_names) == set(expected.column_names)
    for name in expected.column_names:
        evals = expected.column(name).to_pylist()
        avals = actual.column(name).to_pylist()
        for i, (e, a) in enumerate(zip(evals, avals)):
            assert values_equal(e, a), (
                f"column {name!r} row {i}: expected {e!r}, got {a!r} "
                f"(expected type {expected.column(name).type}, actual type {actual.column(name).type})"
            )


# Types that Vortex cleanly reports as unsupported are recorded here (and the example is
# skipped) so the fuzzer keeps exploring instead of failing on a known clean error.
UNSUPPORTED_MARKERS = ("not implemented", "unsupported", "not supported", "unimplemented")
SKIPPED_UNSUPPORTED: set[str] = set()


def _has_struct_of_struct(t: pa.DataType) -> bool:
    if pa.types.is_struct(t) and any(pa.types.is_struct(f.type) for f in t):
        return True
    return any(_has_struct_of_struct(t.field(i).type) for i in range(t.num_fields))


def vortex_array_or_skip(table: pa.Table) -> vx.Array:
    """Convert to a Vortex array, skipping the example on a clean "unsupported type" error."""
    # TODO(https://github.com/vortex-data/vortex/issues/8347): writing an empty table with a
    # top-level struct column panics with "must have visited at least one chunk"
    # (vortex-layout/src/layouts/collect.rs). Remove this guard once fixed.
    assume(not (table.num_rows == 0 and any(pa.types.is_struct(f.type) for f in table.schema)))
    # TODO(https://github.com/vortex-data/vortex/issues/8348): a struct field directly nested in
    # another struct loses its nullability on file roundtrip when it contains nulls
    # (debug_assert in vortex-array/src/stream/adapter.rs). Remove this guard once fixed.
    assume(not any(_has_struct_of_struct(f.type) for f in table.schema))
    try:
        return vx.array(table)
    except ValueError as e:
        msg = str(e).lower()
        if any(marker in msg for marker in UNSUPPORTED_MARKERS):
            SKIPPED_UNSUPPORTED.add(str(table.schema))
            assume(False)
        raise
    raise AssertionError("unreachable")


@pytest.fixture(scope="module", autouse=True)
def _report_unsupported():
    yield
    if SKIPPED_UNSUPPORTED:
        print("\nSchemas skipped because Vortex cleanly reported them as unsupported:")
        for schema in sorted(SKIPPED_UNSUPPORTED):
            print(f"  - {schema}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@given(table=arrow_tables())
def test_write_scan_roundtrip(table: pa.Table) -> None:
    if os.environ.get("VORTEX_FUZZ_DEBUG"):
        print("SCHEMA:", repr(table.schema))
        print("CHUNK LENGTHS:", [[len(c) for c in col.chunks] for col in table.columns])
        print("DATA:", table.to_pylist())
    vortex_array_or_skip(table)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fuzz.vortex")
        vx.io.write(table, path)
        vxf = vx.open(path)
        assert len(vxf) == table.num_rows

        result = vxf.scan().read_all().to_arrow_table()
        assert_tables_equal(table, result)


@given(table=arrow_tables())
def test_write_to_arrow_reader_roundtrip(table: pa.Table) -> None:
    vortex_array_or_skip(table)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fuzz.vortex")
        vx.io.write(table, path)
        vxf = vx.open(path)

        result = vxf.to_arrow().read_all()
        assert_tables_equal(table, result)


@given(table=arrow_tables())
def test_compressed_write_roundtrip(table: pa.Table) -> None:
    arr = vortex_array_or_skip(table)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fuzz.vortex")
        compressed = vx.compress(arr)
        vx.io.write(compressed, path)
        vxf = vx.open(path)

        result = vxf.scan().read_all().to_arrow_table()
        assert_tables_equal(table, result)


@given(table=patterned_tables())
def test_compressed_patterned_roundtrip(table: pa.Table) -> None:
    arr = vortex_array_or_skip(table)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fuzz.vortex")
        vx.io.write(vx.compress(arr), path)
        vxf = vx.open(path)

        result = vxf.scan().read_all().to_arrow_table()
        assert_tables_equal(table, result)


@given(table=arrow_tables(), data=st.data())
def test_scan_parameters(table: pa.Table, data: st.DataObject) -> None:
    vortex_array_or_skip(table)
    nrows = table.num_rows
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fuzz.vortex")
        vx.io.write(table, path)
        vxf = vx.open(path)

        # limit
        limit = data.draw(st.integers(0, nrows + 2), label="limit")
        result = vxf.scan(limit=limit).read_all().to_arrow_table()
        assert_tables_equal(table.slice(0, limit), result)

        # batch_size
        batch_size = data.draw(st.integers(1, nrows + 1), label="batch_size")
        result = vxf.scan(batch_size=batch_size).read_all().to_arrow_table()
        assert_tables_equal(table, result)

        # projection (column subset)
        projection = data.draw(
            st.lists(st.sampled_from(table.column_names), unique=True, min_size=1), label="projection"
        )
        result = vxf.scan(projection).read_all().to_arrow_table()
        assert_tables_equal(table.select(projection), result)

        # sorted, unique row indices
        indices = data.draw(
            st.lists(st.integers(0, nrows - 1), unique=True, max_size=nrows).map(sorted) if nrows else st.just([]),
            label="indices",
        )
        result = vxf.scan(indices=vx.array(pa.array(indices, type=pa.uint64()))).read_all().to_arrow_table()
        rows = table.to_pylist()
        assert_rows_equal([rows[i] for i in indices], result)

        # all parameters combined
        result = (
            vxf.scan(
                projection,
                limit=limit,
                indices=vx.array(pa.array(indices, type=pa.uint64())),
                batch_size=batch_size,
            )
            .read_all()
            .to_arrow_table()
        )
        expected = [{name: rows[i][name] for name in projection} for i in indices[:limit]]
        assert_rows_equal(expected, result)


@given(table=arrow_tables(), data=st.data())
def test_read_url_row_range(table: pa.Table, data: st.DataObject) -> None:
    vortex_array_or_skip(table)
    nrows = table.num_rows
    start = data.draw(st.integers(0, nrows), label="start")
    end = data.draw(st.integers(start, nrows), label="end")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fuzz.vortex")
        vx.io.write(table, path)

        result = vx.io.read_url(f"file://{path}", row_range=(start, end)).to_arrow_table()
        assert_tables_equal(table.slice(start, end - start), result)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
