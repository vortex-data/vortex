# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import overload

import pyarrow as pa
import pyarrow.compute as pc
from substrait.proto import (
    ExtendedExpression,
)

from vortex._lib.expr import Expr

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


def _schema_for_substrait(schema: pa.Schema) -> pa.Schema:
    # PyArrow's to_substrait doesn't support view types; map to string/binary.
    # This is safe because Vortex handles both equivalently.
    # If/When PyArrow to_substrait supports view types, revert.
    # Workaround for: https://github.com/vortex-data/vortex/issues/5759
    fields = []
    for field in schema:
        if field.type == pa.string_view():
            fields.append(field.with_type(pa.string()))
        elif field.type == pa.binary_view():
            fields.append(field.with_type(pa.binary()))
        else:
            fields.append(field)
    return pa.schema(fields)


def arrow_to_vortex(arrow_expression: pc.Expression, schema: pa.Schema) -> Expr:
    compat_schema = _schema_for_substrait(schema)
    substrait_object = ExtendedExpression()
    substrait_object.ParseFromString(bytes(arrow_expression.to_substrait(compat_schema)))

    expressions = extended_expression(substrait_object)

    if len(expressions) < 0 or len(expressions) > 1:
        raise ValueError("arrow_to_vortex: extended expression must have exactly one child")
    return expressions[0]
