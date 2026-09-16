# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import io
import json
import operator
from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

import polars as pl
import pyarrow as pa

import vortex.expr as ve

from ._lib import dtype as _dtype


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    data = json.loads(expr.meta.serialize(format="json"))
    assert isinstance(data, dict)
    return _polars_to_vortex(data)


def decompose_predicate(predicate: pl.Expr) -> tuple[ve.Expr | None, pl.Expr | None]:
    """Split a Polars predicate into a pushable Vortex expression and a Polars residual.

    The predicate is decomposed into its top-level AND conjuncts. Conjuncts that convert to
    Vortex are ANDed into a single Vortex expression; the rest are ANDed into a residual
    Polars expression that the caller must apply itself, e.g. per batch with
    ``DataFrame.filter``. A row passes an AND conjunction exactly when every conjunct is
    true, so filtering by the two parts in sequence is equivalent to filtering by the
    original predicate.

    Returns ``(vortex_expr | None, residual_polars_expr | None)``. At least one side is
    always non-None for a non-trivial predicate; both parts must be applied when present.
    """
    data = json.loads(predicate.meta.serialize(format="json"))
    assert isinstance(data, dict)

    pushed: ve.Expr | None = None
    residual: pl.Expr | None = None
    for conjunct in _conjuncts(data):
        try:
            converted = _polars_to_vortex(conjunct)
        except Exception:
            # Unsupported conjuncts fall back to evaluation by Polars rather than failing
            # the whole scan.
            expr = pl.Expr.deserialize(io.BytesIO(json.dumps(conjunct).encode()), format="json")
            residual = expr if residual is None else residual & expr
        else:
            pushed = converted if pushed is None else pushed & converted
    return pushed, residual


def _conjuncts(expr: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Iterate the top-level AND conjuncts of a serialized Polars expression."""
    if "BinaryExpr" in expr and expr["BinaryExpr"].get("op") in ("And", "LogicalAnd"):
        yield from _conjuncts(expr["BinaryExpr"]["left"])
        yield from _conjuncts(expr["BinaryExpr"]["right"])
    else:
        yield expr


_OPS = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Lt": operator.lt,
    "LtEq": operator.le,
    "Gt": operator.gt,
    "GtEq": operator.ge,
    "And": operator.and_,
    "Or": operator.or_,
    "LogicalAnd": operator.and_,
    "LogicalOr": operator.or_,
}


_LITERAL_TYPES: dict[str, Callable[[Any | None], _dtype.DType]] = {
    "Boolean": lambda v: _dtype.bool_(nullable=v is None),
    "Int": lambda v: _dtype.int_(64, nullable=v is None),
    "Int8": lambda v: _dtype.int_(8, nullable=v is None),
    "Int16": lambda v: _dtype.int_(16, nullable=v is None),
    "Int32": lambda v: _dtype.int_(32, nullable=v is None),
    "Int64": lambda v: _dtype.int_(64, nullable=v is None),
    "UInt8": lambda v: _dtype.uint(8, nullable=v is None),
    "UInt16": lambda v: _dtype.uint(16, nullable=v is None),
    "UInt32": lambda v: _dtype.uint(32, nullable=v is None),
    "UInt64": lambda v: _dtype.uint(64, nullable=v is None),
    "Float32": lambda v: _dtype.float_(32, nullable=v is None),
    "Float64": lambda v: _dtype.float_(64, nullable=v is None),
    "Null": lambda v: _dtype.null(),
    "String": lambda v: _dtype.utf8(nullable=v is None),
    "Binary": lambda v: _dtype.binary(nullable=v is None),
}


def _polars_to_vortex(expr: dict[str, Any]) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    if "BinaryExpr" in expr:
        expr = expr["BinaryExpr"]
        lhs = _polars_to_vortex(expr["left"])
        rhs = _polars_to_vortex(expr["right"])
        op = expr["op"]

        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return cast(ve.Expr, _OPS[op](lhs, rhs))

    if "Ternary" in expr:
        node = expr["Ternary"]
        return ve.zip_(*[_polars_to_vortex(node[k]) for k in ("predicate", "truthy", "falsy")])
    if "Column" in expr:
        return ve.column(expr["Column"])

    # See https://github.com/pola-rs/polars/pull/21849
    if "Scalar" in expr:
        scalar = expr["Scalar"]

        if "Null" in scalar:
            value = None
            dtype = "Null"
        elif "String" in scalar:
            value = scalar["String"]
            dtype = "String"
        elif "Int" in scalar:
            value = scalar["Int"]
            dtype = "Int64"
        elif "Float" in scalar:
            value = scalar["Float"]
            dtype = "Float64"
        elif "Float32" in scalar:
            value = scalar["Float32"]
            dtype = "Float32"
        elif "Float64" in scalar:
            value = scalar["Float64"]
            dtype = "Float64"
        elif "Int32" in scalar:
            value = scalar["Int32"]
            dtype = "Int32"
        elif "Int64" in scalar:
            value = scalar["Int64"]
            dtype = "Int64"
        elif "Time" in scalar:
            return ve.literal(_dtype.time("ns"), scalar["Time"])
        elif "Decimal" in scalar:
            # Polars <1.35 omits the precision: [value, scale].
            if len(scalar["Decimal"]) == 2:
                value, scale = scalar["Decimal"]
                precision = 38
            else:
                value, precision, scale = scalar["Decimal"]
            return ve.literal(_dtype.decimal(precision=precision, scale=scale), value)
        elif "Date" in scalar:
            return ve.literal(_dtype.date("days"), scalar["Date"])
        elif "Binary" in scalar:
            return ve.literal(_dtype.binary(), bytes(scalar["Binary"]))
        elif "Datetime" in scalar:
            return _datetime_literal(scalar["Datetime"])
        elif len(scalar) == 1 and next(iter(scalar)) in _LITERAL_TYPES:
            dtype, value = next(iter(scalar.items()))
        else:
            raise ValueError(f"Cannot convert to Vortex: unsupported Polars scalar value type {scalar}")

        return ve.literal(_LITERAL_TYPES[dtype](value), value)

    if "Literal" in expr:
        expr = expr["Literal"]

        literal_type = next(iter(expr.keys()), None)

        if literal_type == "Scalar":
            return _polars_to_vortex(expr)

        # Special-case Series
        if literal_type == "Series":
            raise ValueError

        # Special-case date-times
        if literal_type == "DateTime":
            return _datetime_literal(expr[literal_type])

        # Unwrap 'Dyn' scalars, whose type hasn't been established yet.
        # (post https://github.com/pola-rs/polars/pull/21849)
        if literal_type == "Dyn":
            expr = expr["Dyn"]
            literal_type = next(iter(expr.keys()), None)

        if literal_type not in _LITERAL_TYPES:
            raise NotImplementedError(f"Unsupported Polars literal type: {literal_type}")
        value = expr[literal_type]
        return ve.literal(_LITERAL_TYPES[literal_type](value), value)

    if "Function" in expr:
        expr = expr["Function"]

        # The members of `is_in` are a list literal, which is not a scalar of its own.
        is_in = expr["function"].get("Boolean", {}) if isinstance(expr["function"], dict) else {}
        if isinstance(is_in, dict) and "IsIn" in is_in:
            needle, members = expr["input"]
            return _is_in(_polars_to_vortex(needle), members, nulls_equal=is_in["IsIn"]["nulls_equal"])

        _inputs = [_polars_to_vortex(e) for e in expr["input"]]

        if expr["function"] == "FillNull":
            if "Literal" in expr["input"][1]:
                return ve.fill_null(_inputs[0], _inputs[1])
            return ve.zip_(ve.is_null(_inputs[0]), _inputs[1], _inputs[0])
        fn = expr["function"]
        if isinstance(fn, dict) and "FieldByName" in fn.get("StructExpr", {}):
            return ve.get_item(fn["StructExpr"]["FieldByName"], _inputs[0])
        if isinstance(fn, dict) and "ReplaceTimeZone" in fn.get("TemporalExpr", {}):
            time_zone, non_existent = fn["TemporalExpr"]["ReplaceTimeZone"]
            if isinstance(time_zone, dict):
                time_zone = time_zone["inner"]
            if non_existent not in ("Raise", "Null"):
                raise NotImplementedError(f"Unsupported Polars nonexistent-time policy: {non_existent}")
            return ve.replace_time_zone(
                _inputs[0],
                time_zone,
                ambiguous=_inputs[1],
                non_existent="raise" if non_existent == "Raise" else "null",
            )

        if "Boolean" in fn:
            fn = fn["Boolean"]

            if "IsBetween" in fn:
                closed = fn["IsBetween"]["closed"]
                lower = operator.ge if closed in ("Both", "Left") else operator.gt
                upper = operator.le if closed in ("Both", "Right") else operator.lt
                return cast(ve.Expr, lower(_inputs[0], _inputs[1]) & upper(_inputs[0], _inputs[2]))

            if fn == "IsNotNull":
                return ve.is_not_null(_inputs[0])

            if fn == "IsNull":
                return ve.is_null(_inputs[0])

            if fn == "Not":
                return ve.not_(_inputs[0])

        if "StringExpr" in fn:
            fn = fn["StringExpr"]
            if "Contains" in fn:
                raise ValueError("Unsupported Polars StringExpr.Contains")

        raise NotImplementedError(f"Unsupported Polars function: {fn}")

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")


def _is_in(needle: ve.Expr, members: dict[str, Any], *, nulls_equal: bool) -> ve.Expr:
    """``needle.is_in(members)`` for a literal list of members, with the null semantics of Polars.

    Polars ignores a null member unless nulls are equal, and a null needle gives null, even against
    an empty list. Vortex's SQL null semantics agree with that for a list without a null, and its
    default semantics for a list with one, which cannot be empty. With nulls equal, a null needle is
    a member exactly when the list holds a null.
    """
    values = _literal_list(members)
    has_null = values.null_count > 0
    dtype = _dtype.list_(_dtype.DType.from_arrow(values.type))
    members_expr = ve.literal(dtype, values.to_pylist())

    if nulls_equal:
        return ve.fill_null(ve.list_contains(members_expr, needle), has_null)
    return ve.list_contains(members_expr, needle, sql_null_semantics=not has_null)


def _literal_list(expr: dict[str, Any]) -> pa.Array[pa.Scalar[Any]]:
    """The members of ``is_in``: a Polars list scalar, or a series of one list, as an Arrow array.

    Polars writes both as an Arrow IPC stream. A list scalar holds the members as its values, and
    an imploded series holds them as the list of its one row.
    """
    while "Alias" in expr:
        expr = expr["Alias"][0]
    literal = expr.get("Literal", {})

    if "Series" in literal:
        series = _ipc_values(literal["Series"])
        if not isinstance(series, pa.ListArray | pa.LargeListArray) or len(series) != 1:
            raise NotImplementedError(f"Unsupported Polars is_in series: {series.type} of {len(series)} rows")
        return series.flatten()

    if "List" in literal.get("Scalar", {}):
        return _ipc_values(literal["Scalar"]["List"])

    raise NotImplementedError(f"Unsupported Polars is_in members: {expr}")


def _ipc_values(raw: list[int]) -> pa.Array[pa.Scalar[Any]]:
    """The one column of an Arrow IPC stream that Polars writes for a literal."""
    return pa.ipc.open_stream(bytes(raw)).read_all().column(0).combine_chunks()


def _datetime_literal(data: list[Any]) -> ve.Expr:
    value, unit, tz = data
    units: dict[str, Literal["s", "ms", "us", "ns"]] = {
        "Nanoseconds": "ns",
        "Microseconds": "us",
        "Milliseconds": "ms",
        "Seconds": "s",
    }
    if unit not in units:
        raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")
    if isinstance(tz, dict):
        tz = tz["inner"]
    dtype = _dtype.timestamp(units[unit], tz=tz, nullable=value is None)
    return ve.literal(dtype, value)
