// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::binary;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::transform::replace;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex::scalar_fn::fns::operators::Operator;

use crate::convert::ExpressionConverter;
use crate::convert::exprs::is_infallible;

pub(super) fn convert_dynamic_filter(
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
) -> Expression {
    // A converter can retain a comparison whose producer guarantees a fixed shape.
    if let Ok(Some(converted)) = converter.try_convert_dynamic(&expression, &schema) {
        return converted;
    }

    // AND/OR preserve a conservative true fallback for unsupported children; NOT does not.
    if let Some(expr) = expression.downcast_ref::<BinaryExpr>() {
        let operator = match expr.op() {
            DFOperator::And => Some(Operator::And),
            DFOperator::Or => Some(Operator::Or),
            _ => None,
        };
        if let Some(operator) = operator {
            return binary(
                operator,
                convert_dynamic_filter(
                    Arc::clone(expr.left()),
                    Arc::clone(&schema),
                    Arc::clone(&converter),
                ),
                convert_dynamic_filter(Arc::clone(expr.right()), schema, converter),
            );
        }
    }
    if !DynamicFilterTracking::classify(&expression).contains_dynamic_filter() {
        return lit(true);
    }

    // A DataFusion dynamic filter may replace its entire body, regardless of its current shape.
    let mut columns = collect_columns(&expression).into_iter().collect::<Vec<_>>();
    columns.sort_unstable_by_key(|column| column.index());
    let inputs = columns
        .iter()
        .map(|column| get_item(column.name(), root()))
        .collect::<Vec<_>>();
    let scope_projection = pack(
        columns
            .iter()
            .enumerate()
            .map(|(idx, column)| (column.name(), get_item(input_field_name(idx), root()))),
        Nullability::NonNullable,
    );
    dynamic_expression(
        Arc::new(DataFusionDynamicFilter {
            expression,
            schema,
            scope_projection,
            converter,
        }),
        inputs,
    )
}

struct DataFusionDynamicFilter {
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    scope_projection: Expression,
    converter: Arc<dyn ExpressionConverter>,
}

impl fmt::Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}

impl fmt::Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expression, f)
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.expression)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        // Unsupported leaves may share an AND with useful bounds. Preserve those bounds;
        // replacing unsupported OR children with true conservatively disables that branch.
        let expression = snapshot_physical_expr(Arc::clone(&self.expression))
            .map(|snapshot| {
                convert_snapshot(&snapshot, &self.schema, self.converter.as_ref(), true)
            })
            .map(|expression| replace(expression, &root(), self.scope_projection.clone()))
            .unwrap_or_else(|_| lit(true));
        expression.bind(scope)
    }
}

// A whole convertible snapshot retains the converter's existing behavior. Partial rescue is
// limited to simple integer comparisons: infallibility alone does not prove equal semantics
// for other domains, such as floating-point signed zero.
fn convert_snapshot(
    expression: &Arc<dyn PhysicalExpr>,
    schema: &arrow_schema::Schema,
    converter: &dyn ExpressionConverter,
    whole: bool,
) -> Expression {
    let integer_comparison = expression
        .downcast_ref::<BinaryExpr>()
        .is_some_and(|binary| {
            matches!(
                binary.op(),
                DFOperator::Eq
                    | DFOperator::NotEq
                    | DFOperator::Lt
                    | DFOperator::LtEq
                    | DFOperator::Gt
                    | DFOperator::GtEq
            ) && [binary.left(), binary.right()].into_iter().all(|operand| {
                (operand.is::<Column>() || operand.is::<Literal>())
                    && operand
                        .data_type(schema)
                        .is_ok_and(|dtype| dtype.is_integer())
            })
        });
    if (whole || integer_comparison || expression.is::<Literal>())
        && let Ok(Some(converted)) = converter.try_convert(expression, schema)
        && is_infallible(&converted)
    {
        return converted;
    }
    if let Some(binary_expr) = expression.downcast_ref::<BinaryExpr>() {
        let operator = match binary_expr.op() {
            DFOperator::And => Some(Operator::And),
            DFOperator::Or => Some(Operator::Or),
            _ => None,
        };
        if let Some(operator) = operator {
            return binary(
                operator,
                convert_snapshot(binary_expr.left(), schema, converter, false),
                convert_snapshot(binary_expr.right(), schema, converter, false),
            );
        }
    }
    lit(true)
}
