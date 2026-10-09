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
from ._lib.arrays import Array


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    data = json.loads(expr.meta.serialize(format="json"))
    assert isinstance(data, dict)
    return _polars_to_vortex(data)


def decompose_predicate(predicate: pl.Expr, scope: Array | None = None) -> tuple[ve.Expr | None, pl.Expr | None]:
    """Split a Polars predicate into a pushable Vortex expression and a Polars residual.

    The predicate is decomposed into its top-level AND conjuncts. A conjunct that converts to
    Vortex is pushed down exactly. A conjunct that does not is kept in the residual Polars
    expression, which the caller must apply itself, e.g. per batch with ``DataFrame.filter``.
    Even then, the translatable parts of an unsupported conjunct may still yield a weaker
    Vortex predicate that every row passing the conjunct satisfies, such as ``a | b`` for
    ``(a & x) | (b & y)``; that weaker predicate is pushed down as a pre-filter.

    If ``scope`` is given, typically an empty array with the dtype of the data being scanned,
    every converted expression is also applied to it, and one that Vortex rejects or that is
    not boolean is treated as unsupported. This catches translations that only fail once the
    scan binds them to the data, such as a bitwise ``&`` of integers, which Polars serializes
    like a logical one.

    Returns ``(vortex_expr | None, residual_polars_expr | None)``. Filtering by the Vortex
    expression and then by the residual is equivalent to filtering by the original
    predicate, so both parts must be applied when present.
    """
    data = json.loads(predicate.meta.serialize(format="json"))
    assert isinstance(data, dict)

    pushed: ve.Expr | None = None
    residual: pl.Expr | None = None
    for conjunct in _conjuncts(data):
        exact, implied = _implied_predicate(conjunct, scope)
        if implied is not None:
            pushed = implied if pushed is None else pushed & implied
        if not exact:
            expr = pl.Expr.deserialize(io.BytesIO(json.dumps(conjunct).encode()), format="json")
            residual = expr if residual is None else residual & expr
    return pushed, residual


def _conjuncts(expr: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Iterate the top-level AND conjuncts of a serialized Polars expression."""
    if _bool_op(expr) == "and":
        yield from _conjuncts(expr["BinaryExpr"]["left"])
        yield from _conjuncts(expr["BinaryExpr"]["right"])
    else:
        yield expr


def _implied_predicate(expr: dict[str, Any], scope: Array | None) -> tuple[bool, ve.Expr | None]:
    """A Vortex predicate that holds for every row where the boolean Polars ``expr`` is true.

    Returns ``(exact, vortex_expr)``: ``exact`` when the result is equivalent to ``expr``, and
    ``None`` when nothing weaker than "true" can be derived. Filters keep only rows where the
    predicate is true, so the relaxation is sound under three-valued logic: a true AND has
    both sides true, and a true OR has at least one side true.
    """
    try:
        converted = _polars_to_vortex(expr)
        if scope is not None:
            dtype = scope.apply(converted).dtype
            if dtype != _dtype.bool_(nullable=dtype.is_nullable()):
                raise TypeError(f"Polars predicate translates to a non-boolean {dtype}")
        return True, converted
    except Exception:
        # Unsupported expressions fall back to evaluation by Polars rather than failing
        # the whole scan.
        pass

    op = _bool_op(expr)
    if op is None:
        return False, None
    _, left = _implied_predicate(expr["BinaryExpr"]["left"], scope)
    _, right = _implied_predicate(expr["BinaryExpr"]["right"], scope)
    if op == "and":
        if left is None or right is None:
            return False, left if right is None else right
        return False, left & right
    if left is None or right is None:
        return False, None
    return False, left | right


def _bool_op(expr: dict[str, Any]) -> Literal["and", "or"] | None:
    if "BinaryExpr" not in expr:
        return None
    op = expr["BinaryExpr"].get("op")
    if op in ("And", "LogicalAnd"):
        return "and"
    if op in ("Or", "LogicalOr"):
        return "or"
    return None


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

        # The members of `is_in` are a list literal, which is not a scalar of its own.
        is_in = expr["function"].get("Boolean", {}) if isinstance(expr["function"], dict) else {}
        if isinstance(is_in, dict) and "IsIn" in is_in:
            needle, members = expr["input"]
            return _is_in(_polars_to_vortex(needle), members, nulls_equal=is_in["IsIn"]["nulls_equal"])

        # The string functions take a literal needle that only has meaning in its raw serialized form.
        raw_inputs: list[dict[str, Any]] = expr["input"]
        fn = expr["function"]
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
