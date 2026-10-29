// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::and;
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
            .and_then(|snapshot| {
                pruning_conjuncts(&snapshot, &self.schema, self.converter.as_ref())
            });
        // The producing DataFusion operator retains exact enforcement when conversion fails.
        let expression = converted.map_or_else(
            || lit(true),
            |expr| replace(expr, &root(), self.scope_projection.clone()),
        );
        expression.bind(scope)
    }
}

// Only runtime pruning can weaken unsupported conjuncts: the producer retains exact enforcement.
fn pruning_conjuncts(
    expression: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    converter: &dyn ExpressionConverter,
) -> Option<Expression> {
    if let Ok(Some(converted)) = converter.try_convert(expression, schema)
        && is_infallible(&converted)
    {
        return Some(converted);
    }
    let binary = expression.downcast_ref::<BinaryExpr>()?;
    if binary.op() != &DFOperator::And {
        return None;
    }
    match (
        pruning_conjuncts(binary.left(), schema, converter),
        pruning_conjuncts(binary.right(), schema, converter),
    ) {
        (Some(left), Some(right)) => Some(and(left, right)),
        (bound, None) | (None, bound) => bound,
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use datafusion_physical_expr::expressions::NegativeExpr;
    use datafusion_physical_expr::expressions::NotExpr;
    use datafusion_physical_plan::expressions as df_expr;
    use vortex::VortexSessionDefault;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::BoolArray;
    use vortex::session::VortexSession;
    use vortex_array::assert_arrays_eq;
    use vortex_arrow::ArrowSessionExt;

    use super::*;
    use crate::convert::DefaultExpressionConverter;

    fn above(column: Arc<dyn PhysicalExpr>, value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(column, DFOperator::Gt, df_expr::lit(value)))
    }

    fn unsupported(column: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        above(Arc::new(NegativeExpr::new(column)), -25)
    }

    #[test]
    fn pruning_conjuncts_keep_supported_bounds_on_either_side() -> anyhow::Result<()> {
        let schema = Schema::new(vec![Field::new("value", DataType::Int32, false)]);
        let column = Arc::new(Column::new("value", 0)) as Arc<dyn PhysicalExpr>;
        let supported = above(Arc::clone(&column), 15);
        let unsupported = unsupported(column);
        let converter = DefaultExpressionConverter::default();
        assert!(converter.try_convert(&unsupported, &schema)?.is_none());
        let expected = converter.try_convert(&supported, &schema)?.unwrap();
        for (left, right) in [
            (Arc::clone(&supported), Arc::clone(&unsupported)),
            (unsupported, supported),
        ] {
            let expression =
                Arc::new(BinaryExpr::new(left, DFOperator::And, right)) as Arc<dyn PhysicalExpr>;
            assert!(converter.try_convert(&expression, &schema)?.is_none());
            assert_eq!(
                pruning_conjuncts(&expression, &schema, &converter),
                Some(expected.clone())
            );
        }
        Ok(())
    }

    #[test]
    fn dynamic_conjuncts_refresh_after_column_remapping() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let batch = arrow_array::record_batch!(("value", Int32, vec![10, 20, 30]))?;
        let schema = batch.schema();
        let input = session.arrow().from_arrow_record_batch(batch, &schema)?;
        let original = Arc::new(Column::new("original", 1)) as Arc<dyn PhysicalExpr>;
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&original)],
            df_expr::lit(true),
        ));
        let remapped = Arc::clone(&filter).with_new_children(vec![
            Arc::new(Column::new("value", 0)) as Arc<dyn PhysicalExpr>,
        ])?;
        let expression = convert_dynamic_filter(
            remapped,
            schema,
            Arc::new(DefaultExpressionConverter::new(session.clone())),
        );
        let supported = above(Arc::clone(&original), 15);
        let unsupported = unsupported(Arc::clone(&original));
        let conjunction = Arc::new(BinaryExpr::new(
            Arc::clone(&supported),
            DFOperator::And,
            Arc::clone(&unsupported),
        )) as Arc<dyn PhysicalExpr>;
        let disjunction = Arc::new(BinaryExpr::new(
            supported,
            DFOperator::Or,
            Arc::clone(&unsupported),
        )) as Arc<dyn PhysicalExpr>;
        let negation = Arc::new(NotExpr::new(Arc::clone(&conjunction))) as Arc<dyn PhysicalExpr>;
        let mut ctx = session.create_execution_ctx();
        for (predicate, expected) in [
            (df_expr::lit(true), [true, true, true]),
            (conjunction, [false, true, true]),
            (disjunction, [true, true, true]),
            (negation, [true, true, true]),
            (above(original, 25), [false, false, true]),
            (unsupported, [true, true, true]),
        ] {
            filter.update(predicate)?;
            assert_arrays_eq!(
                input.clone().apply(&expression)?,
                BoolArray::from_iter(expected.map(Some)),
                &mut ctx
            );
        }
        Ok(())
    }
}
