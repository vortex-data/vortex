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

use super::arrow_predicate::dynamic_predicate;
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
    let key_schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|column| Arc::clone(&schema.fields()[column.index()]))
            .collect::<Vec<_>>(),
    ));
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
            key_schema,
            converter,
        }),
        inputs,
    )
}

struct DataFusionDynamicFilter {
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    scope_projection: Expression,
    key_schema: SchemaRef,
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
        let expression = match snapshot_physical_expr(Arc::clone(&self.expression)) {
            Ok(snapshot) => {
                let converted = self
                    .converter
                    .try_convert(&snapshot, &self.schema)
                    .ok()
                    .flatten()
                    .filter(is_infallible);
                if let Some(expression) = converted {
                    replace(expression, &root(), self.scope_projection.clone())
                } else {
                    let bounds = pruning_conjuncts(
                        &snapshot,
                        &self.schema,
                        self.converter.as_ref(),
                    )
                    .map(|expression| replace(expression, &root(), self.scope_projection.clone()));
                    let predicate = dynamic_predicate(
                        snapshot,
                        Arc::clone(&self.key_schema),
                        self.scope_projection.clone(),
                    );
                    match (bounds, predicate) {
                        (Some(bounds), Some(predicate)) => and(bounds, predicate),
                        (bound, None) | (None, bound) => bound.unwrap_or_else(|| lit(true)),
                    }
                }
            }
            Err(_) => lit(true),
        };
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
    use arrow_array::ArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringViewArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use datafusion_common::hash_utils::create_hashes;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use datafusion_physical_expr::expressions::IsNullExpr;
    use datafusion_physical_expr::expressions::NegativeExpr;
    use datafusion_physical_expr::expressions::NotExpr;
    use datafusion_physical_plan::expressions as df_expr;
    use datafusion_physical_plan::joins::HashTableLookupExpr;
    use datafusion_physical_plan::joins::Map;
    use datafusion_physical_plan::joins::SeededRandomState;
    use datafusion_physical_plan::joins::join_hash_map::JoinHashMapU32;
    use datafusion_physical_plan::joins::utils::JoinHashMapType;
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
    fn live_hash_membership_filters_composite_view_keys_and_preserves_nulls() -> anyhow::Result<()>
    {
        let batch = RecordBatch::try_from_iter(vec![
            (
                "id",
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])) as ArrayRef,
            ),
            (
                "number",
                Arc::new(Int32Array::from(vec![
                    Some(1),
                    Some(1),
                    Some(2),
                    None,
                    Some(3),
                    Some(4),
                ])) as ArrayRef,
            ),
            (
                "text",
                Arc::new(StringViewArray::from(vec![
                    Some("one"),
                    Some("other"),
                    Some("two"),
                    Some("null"),
                    Some("three"),
                    None,
                ])) as ArrayRef,
            ),
        ])?;
        let keys = vec![
            Arc::new(Column::new("number", 1)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("text", 2)) as Arc<dyn PhysicalExpr>,
        ];
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            keys.clone(),
            df_expr::lit(true),
        ));
        let session = VortexSession::default();
        let input = session
            .arrow()
            .from_arrow_record_batch(batch.clone(), &batch.schema())?;
        let expression = convert_dynamic_filter(
            Arc::clone(&filter) as Arc<dyn PhysicalExpr>,
            batch.schema(),
            Arc::new(DefaultExpressionConverter::new(session.clone())),
        );
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(
            input.clone().apply(&expression)?,
            BoolArray::from_iter([Some(true); 6]),
            &mut ctx
        );

        for (number, text, expected) in [(1, "one", vec![0, 3, 5]), (3, "three", vec![3, 4, 5])] {
            let state = SeededRandomState::with_seed(42);
            let build_keys = vec![
                Arc::new(Int32Array::from(vec![number])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![text])) as ArrayRef,
            ];
            let mut hashes = vec![0];
            create_hashes(&build_keys, state.random_state(), &mut hashes)?;
            let mut map = JoinHashMapU32::with_capacity(1);
            map.update_from_iter(Box::new(hashes.iter().enumerate()), 0);
            let membership = Arc::new(HashTableLookupExpr::new(
                keys.clone(),
                state,
                Arc::new(Map::HashMap(Box::new(map))),
                "hash_lookup".into(),
            )) as Arc<dyn PhysicalExpr>;
            let nulls = Arc::new(BinaryExpr::new(
                Arc::new(IsNullExpr::new(Arc::clone(&keys[0]))),
                DFOperator::Or,
                Arc::new(IsNullExpr::new(Arc::clone(&keys[1]))),
            )) as Arc<dyn PhysicalExpr>;
            filter.update(Arc::new(BinaryExpr::new(membership, DFOperator::Or, nulls)))?;
            assert_arrays_eq!(
                input.clone().apply(&expression)?,
                BoolArray::from_iter((0..6).map(|id| Some(expected.contains(&id)))),
                &mut ctx
            );
        }
        Ok(())
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
            (conjunction, [false, true, false]),
            (disjunction, [true, true, true]),
            (negation, [true, false, true]),
            (above(original, 25), [false, false, true]),
            (unsupported, [true, true, false]),
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
