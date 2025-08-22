# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import overload

import pyarrow as pa
import pyarrow.compute as pc
from substrait.proto import (  # pyright: ignore[reportMissingTypeStubs]
    ExtendedExpression,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)

from vortex._lib.expr import Expr  # pyright: ignore[reportMissingModuleSource]

from ..substrait import extended_expression


@overload
def ensure_vortex_expression(expression: None, *, schema: pa.Schema) -> None: ...
@overload
def ensure_vortex_expression(expression: pc.Expression | Expr, *, schema: pa.Schema) -> Expr: ...


def ensure_vortex_expression(expression: pc.Expression | Expr | None, *, schema: pa.Schema) -> Expr | None:
    if expression is None:
        return None
    if isinstance(expression, pc.Expression):
        return arrow_to_vortex(expression, schema)
    return expression


def arrow_to_vortex(arrow_expression: pc.Expression, schema: pa.Schema) -> Expr:
    substrait_object = ExtendedExpression()  # pyright: ignore[reportUnknownVariableType]
    substrait_object.ParseFromString(arrow_expression.to_substrait(schema))  # pyright: ignore[reportUnknownMemberType]

    expressions = extended_expression(substrait_object)  # pyright: ignore[reportUnknownArgumentType]

    if len(expressions) < 0 or len(expressions) > 1:
        raise ValueError("arrow_to_vortex: extended expression must have exactly one child")
    return expressions[0]
