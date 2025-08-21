# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pyarrow.compute as pc
from substrait.proto import (  # pyright: ignore[reportMissingTypeStubs]
    ExtendedExpression,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)

from vortex._lib.expr import Expr  # pyright: ignore[reportMissingModuleSource]

from ..substrait import extended_expression


def arrow_to_vortex(arrow_expression: pc.Expression, schema: pa.Schema) -> Expr:
    substrait_object = ExtendedExpression()  # pyright: ignore[reportUnknownVariableType]
    substrait_object.ParseFromString(arrow_expression.to_substrait(schema))  # pyright: ignore[reportUnknownMemberType]

    expressions = extended_expression(substrait_object)  # pyright: ignore[reportUnknownArgumentType]

    if len(expressions) < 0 or len(expressions) > 1:
        raise ValueError("arrow_to_vortex: extended expression must have exactly one child")
    return expressions[0]
