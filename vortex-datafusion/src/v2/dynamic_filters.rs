// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::split_conjunction;
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

pub(crate) fn convert_dynamic_filter(
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
) -> Expression {
    // A converter can retain a comparison whose producer guarantees a fixed shape.
    if let Ok(Some(converted)) = converter.try_convert(&expression, &schema)
        && is_infallible(&converted)
    {
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
        let converted = snapshot_physical_expr(Arc::clone(&self.expression))
            .ok()
            .and_then(|snapshot| self.convert_conjuncts(&snapshot));
        // The producing DataFusion operator retains exact enforcement when conversion fails.
        let expression = converted.map_or_else(
            || lit(true),
            |expr| replace(expr, &root(), self.scope_projection.clone()),
        );
        expression.bind(scope)
    }
}

impl DataFusionDynamicFilter {
    /// Converts each conjunct of `snapshot` that Vortex can evaluate.
    ///
    /// A join's dynamic filter bounds the probe side's keys and tests them against the build side
    /// with a hash table lookup, which has no Vortex equivalent. Dropping a conjunct only widens
    /// the filter, and the join enforces the keys exactly.
    fn convert_conjuncts(&self, snapshot: &Arc<dyn PhysicalExpr>) -> Option<Expression> {
        split_conjunction(snapshot)
            .into_iter()
            .filter_map(|conjunct| {
                self.converter
                    .try_convert(conjunct, &self.schema)
                    .ok()
                    .flatten()
                    .filter(is_infallible)
            })
            .reduce(|left, right| binary(Operator::And, left, right))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion_common::ScalarValue;
    use datafusion_expr::Operator as DFOperator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::BinaryExpr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::Literal;
    use datafusion_physical_expr::expressions::NoOp;
    use vortex::expr::Expression;
    use vortex::expr::root;

    use super::DataFusionDynamicFilter;
    use crate::convert::exprs::DefaultExpressionConverter;

    fn and(left: Arc<dyn PhysicalExpr>, right: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(left, DFOperator::And, right))
    }

    fn at_least(value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("key", 0)),
            DFOperator::GtEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(value)))),
        ))
    }

    fn convert(expression: Arc<dyn PhysicalExpr>) -> Option<Expression> {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        DataFusionDynamicFilter {
            expression: Arc::clone(&expression),
            schema,
            scope_projection: root(),
            converter: Arc::new(DefaultExpressionConverter::default()),
        }
        .convert_conjuncts(&expression)
    }

    #[test]
    fn keeps_bounds_beside_an_unsupported_conjunct() {
        let converted = convert(and(at_least(1), Arc::new(NoOp::new())));
        assert_eq!(converted, convert(at_least(1)));
        assert_ne!(converted, None);
    }

    #[test]
    fn converts_nothing_without_a_supported_conjunct() {
        assert_eq!(convert(Arc::new(NoOp::new())), None);
    }
}
