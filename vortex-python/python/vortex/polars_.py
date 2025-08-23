# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import operator
from collections.abc import Callable
from typing import Any

import polars as pl

import vortex.expr as ve

from ._lib import dtype as _dtype  # pyright: ignore[reportMissingModuleSource]


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    data = json.loads(expr.meta.serialize(format="json"))  # pyright: ignore[reportAny]
    assert isinstance(data, dict)
    return _polars_to_vortex(data)  # pyright: ignore[reportUnknownArgumentType]


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


_LITERAL_TYPES: dict[str, Callable[[Any | None], _dtype.DType]] = {  # pyright: ignore[reportExplicitAny]
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


def _polars_to_vortex(expr: dict[str, Any]) -> ve.Expr:  # pyright: ignore[reportExplicitAny]
    """Convert a Polars expression to a Vortex expression."""
    if "BinaryExpr" in expr:
        expr = expr["BinaryExpr"]  # pyright: ignore[reportAny]
        lhs = _polars_to_vortex(expr["left"])  # pyright: ignore[reportAny]
        rhs = _polars_to_vortex(expr["right"])  # pyright: ignore[reportAny]
        op = expr["op"]  # pyright: ignore[reportAny]

        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return _OPS[op](lhs, rhs)  # pyright: ignore[reportAny]

    if "Column" in expr:
        return ve.column(expr["Column"])  # pyright: ignore[reportAny]

    # See https://github.com/pola-rs/polars/pull/21849
    if "Scalar" in expr:
        scalar = expr["Scalar"]  # pyright: ignore[reportAny]

        if "Null" in scalar:
            value = None
            dtype = "Null"
        elif "String" in scalar:
            value = scalar["String"]  # pyright: ignore[reportAny]
            dtype = "String"
        elif "Int" in scalar:
            value = scalar["Int"]  # pyright: ignore[reportAny]
            dtype = "Int64"
        elif "Float" in scalar:
            value = scalar["Float"]  # pyright: ignore[reportAny]
            dtype = "Float64"
        else:
            raise ValueError(f"Unsupported Polars scalar value type {scalar}")

        return ve.literal(_LITERAL_TYPES[dtype](value), value)

    if "Literal" in expr:
        expr = expr["Literal"]  # pyright: ignore[reportAny]

        literal_type = next(iter(expr.keys()), None)

        if literal_type == "Scalar":
            return _polars_to_vortex(expr)

        # Special-case Series
        if literal_type == "Series":
            raise ValueError

        # Special-case date-times
        if literal_type == "DateTime":
            (value, unit, tz) = expr[literal_type]  # pyright: ignore[reportAny, reportAny]
            if unit == "Nanoseconds":
                metadata = b"\x00"
            elif unit == "Microseconds":
                metadata = b"\x01"
            elif unit == "Milliseconds":
                metadata = b"\x02"
            elif unit == "Seconds":
                metadata = b"\x03"
            else:
                raise NotImplementedError(f"Unsupported Polars date time unit: {unit}")

            # FIXME(ngates): datetime metadata should be human-readable
            if tz is not None:
                raise ValueError(f"Polars DateTime with timezone not supported: {tz}")
            metadata += b"\x00\x00"

            dtype = _dtype.ext("vortex.timestamp", _dtype.int_(64, nullable=value is None), metadata=metadata)
            return ve.literal(dtype, value)  # pyright: ignore[reportAny]

        # Unwrap 'Dyn' scalars, whose type hasn't been established yet.
        # (post https://github.com/pola-rs/polars/pull/21849)
        if literal_type == "Dyn":
            expr = expr["Dyn"]  # pyright: ignore[reportAny]
            literal_type = next(iter(expr.keys()), None)

        if literal_type not in _LITERAL_TYPES:
            raise NotImplementedError(f"Unsupported Polars literal type: {literal_type}")
        value = expr[literal_type]  # pyright: ignore[reportAny]
        return ve.literal(_LITERAL_TYPES[literal_type](value), value)  # pyright: ignore[reportAny]

    if "Function" in expr:
        expr = expr["Function"]  # pyright: ignore[reportAny]
        _inputs = [_polars_to_vortex(e) for e in expr["input"]]  # pyright: ignore[reportAny]

        fn = expr["function"]  # pyright: ignore[reportAny]
        if "Boolean" in fn:
            fn = fn["Boolean"]  # pyright: ignore[reportAny]

            if "IsIn" in fn:
                fn = fn["IsIn"]  # pyright: ignore[reportAny]
                if fn["nulls_equal"]:
                    raise ValueError(f"Unsupported nulls_equal argument in fn {expr}")

                # Vortex doesn't support is-in, so we need to construct a series of ORs?

        if "StringExpr" in fn:
            fn = fn["StringExpr"]  # pyright: ignore[reportAny]
            if "Contains" in fn:
                raise ValueError("Unsupported Polars StringExpr.Contains")

        raise NotImplementedError(f"Unsupported Polars function: {fn}")

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")
