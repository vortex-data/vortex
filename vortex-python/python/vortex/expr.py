# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from typing import Protocol
from typing import cast as typing_cast

import pyarrow as pa

from ._lib import expr as _expr  # pyright: ignore[reportMissingModuleSource]
from ._lib.dtype import DType  # pyright: ignore[reportMissingModuleSource]


class _HasDType(Protocol):
    @property
    def dtype(self) -> DType: ...


Expr = _expr.Expr
and_ = _expr.and_
cast = _expr.cast
column = _expr.column
col = column
is_null = _expr.is_null
is_not_null = _expr.is_not_null
literal = _expr.literal
not_ = _expr.not_
root = _expr.root


def plan(
    expr: Expr,
    *,
    schema: pa.Schema | None = None,
    file: object | None = None,
    kind: str = "expr",
) -> Expr:
    """Plan an expression against an Arrow schema or Vortex file."""
    if schema is not None and file is not None:
        raise ValueError("exactly one of schema or file must be provided")
    if schema is None and file is None:
        raise ValueError("exactly one of schema or file must be provided")

    if schema is not None:
        scope = DType.from_arrow_schema(schema)
    else:
        assert file is not None
        scope = typing_cast(_HasDType, file).dtype
    return _expr.plan(expr, scope, kind=kind)


__all__ = [
    "Expr",
    "column",
    "col",
    "literal",
    "root",
    "not_",
    "and_",
    "cast",
    "is_null",
    "is_not_null",
    "plan",
]
