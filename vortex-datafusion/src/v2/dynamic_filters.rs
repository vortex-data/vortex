// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converts DataFusion dynamic filters, such as a hash join's build-side keys or a top-K bound,
//! into Vortex scan predicates that follow the filter's current value.
//!
//! Vortex converts each conjunct of the filter's current value, or a widening of it that holds
//! wherever the conjunct holds. The scan switches the filter off once it stops rejecting rows.

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_common::ScalarValue;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::conjunction;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::CaseExpr;
use datafusion_physical_expr::expressions::Literal;
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
use vortex::expr::transform::find_between;
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
    if let Some(expr) = expression.downcast_ref::<BinaryExpr>()
        && let Some(operator) = boolean_operator(expr.op())
    {
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
    let conjuncts = DynamicConjuncts {
        expression,
        schema,
        converter,
    };
    dynamic_expression(
        Arc::new(DataFusionDynamicFilter {
            conjuncts,
            scope_projection,
        }),
        inputs,
    )
}

fn boolean_operator(operator: &DFOperator) -> Option<Operator> {
    match operator {
        DFOperator::And => Some(Operator::And),
        DFOperator::Or => Some(Operator::Or),
        _ => None,
    }
}

/// A dynamic filter, and how to convert the conjuncts of its current value.
struct DynamicConjuncts {
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
}

impl DynamicConjuncts {
    /// The current value of the filter, or `None` when DataFusion cannot snapshot it.
    fn snapshot(&self) -> Option<Arc<dyn PhysicalExpr>> {
        snapshot_physical_expr(Arc::clone(&self.expression)).ok()
    }

    /// `expr` converted exactly, when Vortex can evaluate it on any row.
    fn convert(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<Expression> {
        self.converter
            .try_convert(expr, &self.schema)
            .ok()
            .flatten()
            .filter(is_infallible)
    }

    /// A Vortex expression that holds wherever `expr` holds, or `None` when it would be `true`.
    ///
    /// The operator that produces a dynamic filter enforces it exactly, so the scan may keep rows
    /// the filter rejects.
    fn widen(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<Expression> {
        if let Some(converted) = self.convert(expr) {
            return Some(converted);
        }
        if let Some(expr) = expr.downcast_ref::<BinaryExpr>() {
            match expr.op() {
                DFOperator::And => {
                    return match (self.widen(expr.left()), self.widen(expr.right())) {
                        (Some(left), Some(right)) => Some(binary(Operator::And, left, right)),
                        (left, right) => left.or(right),
                    };
                }
                DFOperator::Or => {
                    return Some(binary(
                        Operator::Or,
                        self.widen(expr.left())?,
                        self.widen(expr.right())?,
                    ));
                }
                _ => {}
            }
        }
        if let Some(expr) = expr.downcast_ref::<CaseExpr>() {
            return self.widen_case(expr);
        }
        None
    }

    /// Widens a CASE to the union of its branches.
    ///
    /// A partitioned hash join's filter routes each row by the hash of its keys to the bounds and
    /// hash table of the build partition the row would join, `CASE hash(keys) % n WHEN 0 THEN
    /// filter_0 ... ELSE false END`. Vortex cannot compute the hash, but a row the CASE keeps
    /// passes the branch it is routed to, so it passes the union of all branches.
    fn widen_case(&self, expr: &CaseExpr) -> Option<Expression> {
        let mut branches = Vec::new();
        for branch in expr
            .when_then_expr()
            .iter()
            .map(|(_, then)| then)
            .chain(expr.else_expr())
        {
            match boolean_literal(branch) {
                // A row routed to this branch is rejected.
                Some(Some(false) | None) => continue,
                Some(Some(true)) => return None,
                None => branches.push(branch),
            }
        }
        if branches.is_empty() {
            return Some(lit(false));
        }
        // The branches of a join bound the same keys, so the hull of their bounds is about as
        // selective as their union, and Vortex evaluates and prunes one range per key instead of
        // one per partition.
        if let Some(hull) = bounds_hull(&branches).and_then(|hull| self.convert(&hull)) {
            return Some(hull);
        }
        branches
            .into_iter()
            .map(|branch| self.widen(branch))
            .reduce(|left, right| Some(binary(Operator::Or, left?, right?)))
            .flatten()
    }
}

/// The value of a boolean literal, `Some(None)` for a null, or `None` when `expr` is not one.
fn boolean_literal(expr: &Arc<dyn PhysicalExpr>) -> Option<Option<bool>> {
    match expr.downcast_ref::<Literal>()?.value() {
        ScalarValue::Boolean(value) => Some(*value),
        _ => None,
    }
}

/// The lower and upper bound that a conjunction places on one expression.
struct KeyBounds {
    key: Arc<dyn PhysicalExpr>,
    lower: Option<ScalarValue>,
    upper: Option<ScalarValue>,
}

/// The bounds `key >= lower` and `key <= upper` among the conjuncts of `expr`, the tightest for
/// each key.
fn conjunct_bounds(expr: &Arc<dyn PhysicalExpr>) -> Vec<KeyBounds> {
    let mut bounds: Vec<KeyBounds> = Vec::new();
    for conjunct in split_conjunction(expr) {
        let Some(comparison) = conjunct.downcast_ref::<BinaryExpr>() else {
            continue;
        };
        let Some(value) = comparison
            .right()
            .downcast_ref::<Literal>()
            .map(Literal::value)
            .filter(|value| !value.is_null())
        else {
            continue;
        };
        let is_lower = match comparison.op() {
            DFOperator::GtEq => true,
            DFOperator::LtEq => false,
            _ => continue,
        };
        let key = comparison.left();
        let index = match bounds
            .iter()
            .position(|bound| bound.key.as_ref() == key.as_ref())
        {
            Some(index) => index,
            None => {
                bounds.push(KeyBounds {
                    key: Arc::clone(key),
                    lower: None,
                    upper: None,
                });
                bounds.len() - 1
            }
        };
        let entry = &mut bounds[index];
        let (current, tighter) = if is_lower {
            (&mut entry.lower, std::cmp::Ordering::Greater)
        } else {
            (&mut entry.upper, std::cmp::Ordering::Less)
        };
        match current {
            Some(current) if value.partial_cmp(current) != Some(tighter) => {}
            _ => *current = Some(value.clone()),
        }
    }
    bounds
}

/// The bounds that hold for every row passing any of `branches`, as `key >= lower AND key <=
/// upper` for each key that every branch bounds.
fn bounds_hull(branches: &[&Arc<dyn PhysicalExpr>]) -> Option<Arc<dyn PhysicalExpr>> {
    let (first, rest) = branches.split_first()?;
    let mut hull = conjunct_bounds(first);
    for branch in rest {
        let bounds = conjunct_bounds(branch);
        hull.retain_mut(|hull| {
            let Some(bound) = bounds
                .iter()
                .find(|bound| bound.key.as_ref() == hull.key.as_ref())
            else {
                return false;
            };
            hull.lower = widest(
                hull.lower.take(),
                bound.lower.as_ref(),
                std::cmp::Ordering::Less,
            );
            hull.upper = widest(
                hull.upper.take(),
                bound.upper.as_ref(),
                std::cmp::Ordering::Greater,
            );
            hull.lower.is_some() || hull.upper.is_some()
        });
    }
    let comparisons = hull.into_iter().flat_map(|bounds| {
        [
            bounds.lower.map(|lower| (DFOperator::GtEq, lower)),
            bounds.upper.map(|upper| (DFOperator::LtEq, upper)),
        ]
        .into_iter()
        .flatten()
        .map(move |(operator, value)| {
            Arc::new(BinaryExpr::new(
                Arc::clone(&bounds.key),
                operator,
                Arc::new(Literal::new(value)),
            )) as Arc<dyn PhysicalExpr>
        })
    });
    let comparisons = comparisons.collect::<Vec<_>>();
    (!comparisons.is_empty()).then(|| conjunction(comparisons))
}

/// The wider of two bounds, where `wider` orders a wider bound before the other. A missing or
/// incomparable bound leaves the key unbounded on that side.
fn widest(
    left: Option<ScalarValue>,
    right: Option<&ScalarValue>,
    wider: std::cmp::Ordering,
) -> Option<ScalarValue> {
    let (left, right) = (left?, right?);
    match right.partial_cmp(&left)? {
        ordering if ordering == wider => Some(right.clone()),
        _ => Some(left),
    }
}

/// The native conjunct of a dynamic filter: each conjunct of its current value that Vortex
/// converts, or a widening of it that Vortex converts.
struct DataFusionDynamicFilter {
    conjuncts: DynamicConjuncts,
    scope_projection: Expression,
}

impl fmt::Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.conjuncts.expression, f)
    }
}

impl fmt::Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.conjuncts.expression, f)
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.conjuncts.expression)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let converted = self
            .conjuncts
            .snapshot()
            .and_then(|snapshot| self.convert_conjuncts(&snapshot));
        // The producing DataFusion operator retains exact enforcement when conversion fails.
        let expression = converted.map_or_else(
            || lit(true),
            |expr| replace(expr, &root(), self.scope_projection.clone()),
        );
        // A join bounds its probe keys from both sides. Each comparison would decode the keys
        // again, while a between decodes them once.
        find_between(expression.bind(scope)?)
    }
}

impl DataFusionDynamicFilter {
    /// Converts each conjunct of `snapshot` that Vortex can evaluate, or else a widening of it.
    ///
    /// A join's dynamic filter bounds the probe side's keys and tests them against the build side
    /// with a hash table lookup, which has no Vortex equivalent. Dropping or widening a conjunct
    /// only widens the filter, and the join enforces the keys exactly.
    fn convert_conjuncts(&self, snapshot: &Arc<dyn PhysicalExpr>) -> Option<Expression> {
        split_conjunction(snapshot)
            .into_iter()
            .filter_map(|conjunct| self.conjuncts.widen(conjunct))
            .reduce(|left, right| binary(Operator::And, left, right))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use arrow_schema::SchemaRef;
    use datafusion_common::ScalarValue;
    use datafusion_expr::Operator as DFOperator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::BinaryExpr;
    use datafusion_physical_expr::expressions::CaseExpr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::IsNullExpr;
    use datafusion_physical_expr::expressions::Literal;
    use datafusion_physical_expr::expressions::NoOp;
    use datafusion_physical_expr::expressions::lit as df_lit;
    use datafusion_physical_plan::joins::HashExpr;
    use datafusion_physical_plan::joins::SeededRandomState;
    use rstest::rstest;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::expr::Expression;
    use vortex::expr::binary;
    use vortex::expr::root;
    use vortex::scalar_fn::fns::between::Between;
    use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
    use vortex::scalar_fn::fns::operators::Operator;

    use super::DataFusionDynamicFilter;
    use super::DynamicConjuncts;
    use super::boolean_literal;
    use crate::convert::exprs::DefaultExpressionConverter;

    fn key() -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new("key", 0))
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, true)]))
    }

    fn and(left: Arc<dyn PhysicalExpr>, right: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(left, DFOperator::And, right))
    }

    fn or(left: Arc<dyn PhysicalExpr>, right: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(left, DFOperator::Or, right))
    }

    fn compare(operator: DFOperator, value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            key(),
            operator,
            Arc::new(Literal::new(ScalarValue::Int32(Some(value)))),
        ))
    }

    fn at_least(value: i32) -> Arc<dyn PhysicalExpr> {
        compare(DFOperator::GtEq, value)
    }

    fn at_most(value: i32) -> Arc<dyn PhysicalExpr> {
        compare(DFOperator::LtEq, value)
    }

    fn bounds(lower: i32, upper: i32) -> Arc<dyn PhysicalExpr> {
        and(at_least(lower), at_most(upper))
    }

    /// `key % divisor = 0`, which Vortex does not convert, standing in for a hash table lookup.
    fn multiple_of(divisor: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                key(),
                DFOperator::Modulo,
                Arc::new(Literal::new(ScalarValue::Int32(Some(divisor)))),
            )),
            DFOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(0)))),
        ))
    }

    /// A partitioned hash join's filter, routing each key by its hash to one of `branches`.
    fn partitioned(
        branches: Vec<Arc<dyn PhysicalExpr>>,
        else_expr: Arc<dyn PhysicalExpr>,
    ) -> anyhow::Result<Arc<dyn PhysicalExpr>> {
        let partitions = u64::try_from(branches.len())?;
        let route = Arc::new(BinaryExpr::new(
            Arc::new(HashExpr::new(
                vec![key()],
                SeededRandomState::with_seed(0),
                "hash_repartition".to_string(),
            )),
            DFOperator::Modulo,
            df_lit(ScalarValue::UInt64(Some(partitions))),
        ));
        let when_then = branches
            .into_iter()
            .zip(0u64..)
            .map(|(then, partition)| (df_lit(ScalarValue::UInt64(Some(partition))), then))
            .collect();
        Ok(Arc::new(CaseExpr::try_new(
            Some(route),
            when_then,
            Some(else_expr),
        )?))
    }

    fn conjuncts(expression: Arc<dyn PhysicalExpr>) -> DynamicConjuncts {
        DynamicConjuncts {
            expression,
            schema: schema(),
            converter: Arc::new(DefaultExpressionConverter::default()),
        }
    }

    fn native_filter(expression: Arc<dyn PhysicalExpr>) -> DataFusionDynamicFilter {
        DataFusionDynamicFilter {
            conjuncts: conjuncts(expression),
            scope_projection: root(),
        }
    }

    fn native(expression: Arc<dyn PhysicalExpr>) -> Option<Expression> {
        native_filter(Arc::clone(&expression)).convert_conjuncts(&expression)
    }

    #[test]
    fn keeps_bounds_beside_an_unsupported_conjunct() {
        let converted = native(and(at_least(1), Arc::new(NoOp::new())));
        assert_eq!(converted, native(at_least(1)));
        assert_ne!(converted, None);
    }

    #[test]
    fn converts_nothing_without_a_supported_conjunct() {
        assert_eq!(native(Arc::new(NoOp::new())), None);
    }

    #[test]
    fn merges_key_bounds_into_a_between() -> VortexResult<()> {
        let scope = DType::struct_(
            [("key", DType::Primitive(PType::I32, Nullability::Nullable))],
            Nullability::NonNullable,
        );
        let current = native_filter(bounds(1, 5)).current(&scope)?;
        assert!(current.is::<Between>(), "{current}");
        Ok(())
    }

    #[rstest]
    #[case::hull(vec![bounds(1, 5), bounds(10, 20)], df_lit(false), Some(bounds(1, 20)))]
    #[case::rejecting_branch(vec![df_lit(false), bounds(10, 20)], df_lit(false), Some(bounds(10, 20)))]
    #[case::unbounded_side(vec![at_least(3), bounds(10, 20)], df_lit(false), Some(at_least(3)))]
    #[case::keeping_else(vec![bounds(1, 5), bounds(10, 20)], df_lit(true), None)]
    #[case::unbounded_branch(vec![bounds(1, 5), Arc::new(NoOp::new())], df_lit(false), None)]
    fn widens_partitioned_filters_to_the_hull_of_their_branches(
        #[case] branches: Vec<Arc<dyn PhysicalExpr>>,
        #[case] else_expr: Arc<dyn PhysicalExpr>,
        #[case] expected: Option<Arc<dyn PhysicalExpr>>,
    ) -> anyhow::Result<()> {
        let branches = branches
            .into_iter()
            .map(|branch| match boolean_literal(&branch) {
                Some(_) => branch,
                None => and(branch, multiple_of(2)),
            })
            .collect();
        let filter = partitioned(branches, else_expr)?;
        assert_eq!(native(filter), expected.and_then(native));
        Ok(())
    }

    #[test]
    fn widens_null_preserving_filters() {
        let is_null = Arc::new(IsNullExpr::new(key())) as Arc<dyn PhysicalExpr>;
        let filter = or(
            Arc::clone(&is_null),
            and(bounds(1, 5), Arc::new(NoOp::new())),
        );
        assert_eq!(
            native(filter),
            native(is_null)
                .zip(native(bounds(1, 5)))
                .map(|(left, right)| binary(Operator::Or, left, right))
        );
    }
}
