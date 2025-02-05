import pyarrow as pa
import pyarrow.compute as pc
from substrait.proto import ExtendedExpression

import vortex.expr
from vortex.substrait import extended_expression


def arrow_to_vortex(arrow_expression: pc.Expression, schema: pa.Schema) -> vortex.expr.Expr:
    substrait_object = ExtendedExpression()
    substrait_object.ParseFromString(arrow_expression.to_substrait(schema))

    expressions = extended_expression(substrait_object)

    if len(expressions) < 0 or len(expressions) > 1:
        raise ValueError("arrow_to_vortex: extended expression must have exactly one child")
    return expressions[0]
