import json
import operator

import polars as pl

import vortex as vx
import vortex.expr as ve


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    return _polars_to_vortex(json.loads(expr.meta.write_json()))


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

_LITERAL_TYPES = {
    "Int": lambda v: vx.int_(64, nullable=v is None),
    "Int8": lambda v: vx.int_(8, nullable=v is None),
    "Int16": lambda v: vx.int_(16, nullable=v is None),
    "Int32": lambda v: vx.int_(32, nullable=v is None),
    "Int64": lambda v: vx.int_(64, nullable=v is None),
    "UInt8": lambda v: vx.uint(8, nullable=v is None),
    "UInt16": lambda v: vx.uint(16, nullable=v is None),
    "UInt32": lambda v: vx.uint(32, nullable=v is None),
    "UInt64": lambda v: vx.uint(64, nullable=v is None),
    "Float32": lambda v: vx.float_(32, nullable=v is None),
    "Float64": lambda v: vx.float_(64, nullable=v is None),
    "Boolean": lambda v: vx.bool_(nullable=v is None),
    "Null": lambda v: vx.null(),
    "String": lambda v: vx.utf8(nullable=v is None),
    "Binary": lambda v: vx.binary(nullable=v is None),
}


def _polars_to_vortex(expr: dict) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    if "BinaryExpr" in expr:
        expr = expr["BinaryExpr"]
        lhs = _polars_to_vortex(expr["left"])
        rhs = _polars_to_vortex(expr["right"])
        op = expr["op"]

        if op not in _OPS:
            raise NotImplementedError(f"Unsupported Polars binary operator: {op}")
        return _OPS[op](lhs, rhs)

    if "Column" in expr:
        return ve.column(expr["Column"])

    if "Literal" in expr:
        expr = expr["Literal"]

        literal_type = next(iter(expr.keys()), None)

        # Special-case Series
        if literal_type == "Series":
            expr = pl.Expr.from_json(json.dumps({"Literal": expr}))
            raise ValueError

        # Special-case date-times
        if literal_type == "DateTime":
            (value, unit, tz) = expr[literal_type]
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

            dtype = vx.ext("vortex.timestamp", vx.int_(64, nullable=value is None), metadata=metadata)
            return ve.literal(dtype, value)

        if literal_type not in _LITERAL_TYPES:
            raise NotImplementedError(f"Unsupported Polars literal type: {literal_type}")
        value = expr[literal_type]
        return ve.literal(_LITERAL_TYPES[literal_type](value), value)

    if "Function" in expr:
        expr = expr["Function"]
        _inputs = [_polars_to_vortex(e) for e in expr["input"]]

        fn = expr["function"]
        if "Boolean" in fn:
            fn = fn["Boolean"]

            if "IsIn" in fn:
                fn = fn["IsIn"]
                if fn["nulls_equal"]:
                    raise ValueError(f"Unsupported nulls_equal argument in fn {expr}")

                # Vortex doesn't support is-in, so we need to construct a series of ORs?

        raise NotImplementedError(f"Unsupported Polars function: {fn}")

    raise NotImplementedError(f"Unsupported Polars expression: {expr}")
