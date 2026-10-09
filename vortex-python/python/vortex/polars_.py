# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import operator
from collections.abc import Callable
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
    "Plus": operator.add,
    "Minus": operator.sub,
    "Multiply": operator.mul,
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

        if op == "TrueDivide":
            # Polars' `/` always divides in floating point, even between integers.
            f64 = _dtype.float_(64, nullable=True)
            return ve.div(ve.cast(lhs, f64), ve.cast(rhs, f64))
        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return cast(ve.Expr, _OPS[op](lhs, rhs))

    if "Cast" in expr:
        return _cast_to_vortex(expr["Cast"])

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
            value, precision, scale = scalar["Decimal"]
            return ve.literal(_dtype.decimal(precision=precision, scale=scale), value)
        elif "Date" in scalar:
            return ve.literal(_dtype.date("days"), scalar["Date"])
        elif "Binary" in scalar:
            return ve.literal(_dtype.binary(), bytes(scalar["Binary"]))
        elif "Datetime" in scalar:
            return _datetime_literal(scalar["Datetime"])
        elif "Duration" in scalar:
            raise NotImplementedError("Vortex has no duration type to represent a Polars Duration literal")
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
        raw_inputs: list[dict[str, Any]] = expr["input"]
        fn = expr["function"]

        # These take a literal argument that only has meaning in its raw serialized form.
        if isinstance(fn, dict) and isinstance(fn.get("Boolean"), dict) and "IsIn" in fn["Boolean"]:
            if fn["Boolean"]["IsIn"]["nulls_equal"]:
                raise NotImplementedError(f"Unsupported nulls_equal argument in fn {expr}")
            return _is_in_to_vortex(_polars_to_vortex(raw_inputs[0]), raw_inputs[1])
        if isinstance(fn, dict) and "StringExpr" in fn:
            fn = fn["StringExpr"]
            if fn == "StartsWith":
                return ve.like(_polars_to_vortex(raw_inputs[0]), _like_needle(raw_inputs[1]) + "%")
            if fn == "EndsWith":
                return ve.like(_polars_to_vortex(raw_inputs[0]), "%" + _like_needle(raw_inputs[1]))
            if isinstance(fn, dict) and "Contains" in fn:
                if not fn["Contains"]["literal"]:
                    raise NotImplementedError("Unsupported regex pattern in Polars StringExpr.Contains")
                return ve.like(_polars_to_vortex(raw_inputs[0]), "%" + _like_needle(raw_inputs[1]) + "%")
            raise NotImplementedError(f"Unsupported Polars string function: {fn}")

        _inputs = [_polars_to_vortex(e) for e in raw_inputs]

        if expr["function"] == "FillNull":
            if "Literal" in expr["input"][1]:
                return ve.fill_null(_inputs[0], _inputs[1])
            return ve.zip_(ve.is_null(_inputs[0]), _inputs[1], _inputs[0])
        fn = expr["function"]
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

        raise NotImplementedError(f"Unsupported Polars function: {fn}")

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")


_TIME_UNITS: dict[str, Literal["s", "ms", "us", "ns"]] = {
    "Nanoseconds": "ns",
    "Microseconds": "us",
    "Milliseconds": "ms",
    "Seconds": "s",
}


def _datetime_literal(data: list[Any]) -> ve.Expr:
    value, unit, tz = data
    if unit not in _TIME_UNITS:
        raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")
    if isinstance(tz, dict):
        tz = tz["inner"]
    dtype = _dtype.timestamp(_TIME_UNITS[unit], tz=tz, nullable=value is None)
    return ve.literal(dtype, value)


_CAST_DTYPES: dict[str, Callable[[], _dtype.DType]] = {
    "Boolean": lambda: _dtype.bool_(nullable=True),
    "Int8": lambda: _dtype.int_(8, nullable=True),
    "Int16": lambda: _dtype.int_(16, nullable=True),
    "Int32": lambda: _dtype.int_(32, nullable=True),
    "Int64": lambda: _dtype.int_(64, nullable=True),
    "UInt8": lambda: _dtype.uint(8, nullable=True),
    "UInt16": lambda: _dtype.uint(16, nullable=True),
    "UInt32": lambda: _dtype.uint(32, nullable=True),
    "UInt64": lambda: _dtype.uint(64, nullable=True),
    "Float32": lambda: _dtype.float_(32, nullable=True),
    "Float64": lambda: _dtype.float_(64, nullable=True),
    "String": lambda: _dtype.utf8(nullable=True),
    "Binary": lambda: _dtype.binary(nullable=True),
    "Date": lambda: _dtype.date("days", nullable=True),
    "Time": lambda: _dtype.time("ns", nullable=True),
}


def _cast_to_vortex(cast_expr: dict[str, Any]) -> ve.Expr:
    """Convert a Polars ``Cast`` node. Casts are nullable, since every Polars column is."""
    if cast_expr["options"] != "Strict":
        # Non-strict casts produce null where Polars' strict cast, like Vortex's, raises.
        raise NotImplementedError(f"Unsupported Polars cast options: {cast_expr['options']}")

    target = cast_expr["dtype"]
    # Post https://github.com/pola-rs/polars/pull/21797 the target is a DataTypeExpr.
    if isinstance(target, dict) and "Literal" in target:
        target = target["Literal"]

    if isinstance(target, dict) and "Datetime" in target:
        unit, tz = target["Datetime"]
        # Polars encodes a naive datetime literal as a UTC datetime cast to a naive one. Fold
        # the cast so the timezone relabel happens here rather than as a Vortex timestamp cast.
        scalar = cast_expr["expr"].get("Literal", {}).get("Scalar", {})
        if "Datetime" in scalar and scalar["Datetime"][1] == unit:
            return _datetime_literal([scalar["Datetime"][0], unit, tz])
        if unit not in _TIME_UNITS:
            raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")
        if isinstance(tz, dict):
            tz = tz["inner"]
        dtype = _dtype.timestamp(_TIME_UNITS[unit], tz=tz, nullable=True)
    elif isinstance(target, dict) and "Decimal" in target:
        precision, scale = target["Decimal"]
        dtype = _dtype.decimal(precision=38 if precision is None else precision, scale=scale or 0, nullable=True)
    elif isinstance(target, str) and target in _CAST_DTYPES:
        dtype = _CAST_DTYPES[target]()
    else:
        raise NotImplementedError(f"Unsupported Polars cast target: {target}")
    return ve.cast(_polars_to_vortex(cast_expr["expr"]), dtype)


def _like_needle(expr: dict[str, Any]) -> str:
    """Escape LIKE wildcards in a Polars string literal so that it matches literally."""
    value: object = expr.get("Literal", {}).get("Scalar", {}).get("String")
    if not isinstance(value, str):
        raise NotImplementedError(f"Expected a Polars string literal, got: {expr}")
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_IS_IN_STRINGS = (pa.string(), pa.large_string(), pa.string_view(), pa.binary(), pa.large_binary(), pa.binary_view())


def _is_in_to_vortex(child: ve.Expr, values: dict[str, Any]) -> ve.Expr:
    """Convert ``is_in`` over a literal set into a balanced OR of equalities."""
    scalar = values.get("Literal", {}).get("Scalar", {})
    if "List" not in scalar:
        raise NotImplementedError(f"Unsupported Polars is_in values: {values}")

    # Polars serializes the set of values as an Arrow IPC stream.
    column = pa.ipc.open_stream(bytes(scalar["List"])).read_all().column(0)
    if not (pa.types.is_integer(column.type) or pa.types.is_floating(column.type) or column.type in _IS_IN_STRINGS):
        raise NotImplementedError(f"Unsupported Polars is_in value type: {column.type}")
    if column.null_count:
        raise NotImplementedError("Unsupported null value in Polars is_in values")

    return ve.or_collect(ve.eq(child, value) for value in column.to_pylist()) or ve.literal(_dtype.bool_(), False)
