// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converts DataFusion dynamic filters, such as the build-side keys of a hash join or the threshold
//! of a TopK operator, into Vortex expressions.
//!
//! DataFusion keeps enforcing every dynamic filter, so a scan receives each one as a
//! [`switch`](vortex::scalar_fn::fns::switch::Switch) conjunct: it prunes zones with it and removes
//! rows before it decodes the other columns, and it skips the conjunct when it keeps too many rows.
//! Parts of a filter that Vortex cannot express are dropped, apart from the hash table lookup of a
//! join, which DataFusion evaluates inside the scan.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;

use arrow_schema::DataType;
use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions as df_expr;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use itertools::Itertools;
use parking_lot::RwLock;
use parking_lot::RwLockUpgradableReadGuard;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::binary;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::switch;
use vortex::expr::transform::find_between;
use vortex::expr::transform::replace;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::switch::SwitchState;

use crate::convert::ExpressionConverter;
use crate::convert::df_predicate::DataFusionPredicate;
use crate::convert::exprs::is_infallible;

/// The largest percentage of rows that the range bounds of a dynamic filter can keep and still be
/// evaluated. DuckDB keeps evaluating the min/max filter of a join up to 90%, but in a Vortex scan
/// a mask that keeps more than half of the rows makes the projection cost more than it saves.
const RANGE_MAX_KEPT_PERCENT: u8 = 50;

/// The largest percentage of rows that a membership test of a dynamic filter, an IN list or a hash
/// table lookup, can keep and still be evaluated. DuckDB uses the same limit for its join Bloom
/// filters.
const MEMBERSHIP_MAX_KEPT_PERCENT: u8 = 50;

/// The largest percentage of rows that a dynamic filter that can still change, such as the
/// threshold of a TopK operator, can keep and still be evaluated.
const DYNAMIC_MAX_KEPT_PERCENT: u8 = 50;

/// Converts a filter over `schema` that contains dynamic filters into a Vortex expression that
/// reads the current value of each dynamic filter.
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
    if let Some(expr) = expression.downcast_ref::<df_expr::BinaryExpr>() {
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
    and_collect(DynamicFilter::new(expression, schema, converter).dynamic_expressions())
        .unwrap_or_else(|| lit(true))
}

/// The Vortex conjuncts of a dynamic filter, each with the state of its switch.
type Conjuncts = Arc<[(Expression, SwitchState)]>;

/// A DataFusion dynamic filter over a schema, with its conversion as of a generation.
pub(crate) struct DynamicFilter {
    expression: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
    conversion: RwLock<Option<(u64, Conjuncts)>>,
    /// The switched dynamic expressions of the filter, built once so that every scan shares their
    /// switches.
    dynamic: OnceLock<Vec<Expression>>,
    /// The switched expressions of the bounds the filter implies on each of its columns, built
    /// once so that every scan shares their switches.
    bounds: OnceLock<Vec<(FieldName, Expression)>>,
}

impl DynamicFilter {
    /// A dynamic filter `expression` over the columns of `schema`.
    pub(crate) fn new(
        expression: Arc<dyn PhysicalExpr>,
        schema: SchemaRef,
        converter: Arc<dyn ExpressionConverter>,
    ) -> Arc<Self> {
        Arc::new(Self {
            expression,
            schema,
            converter,
            conversion: RwLock::default(),
            dynamic: OnceLock::new(),
            bounds: OnceLock::new(),
        })
    }

    /// The DataFusion expression of the filter.
    pub(crate) fn expression(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expression
    }

    /// The generation and the Vortex conjuncts of the current value of the filter, over the
    /// columns of its schema.
    pub(crate) fn current(&self) -> (u64, Conjuncts) {
        // A generation read before the snapshot is at most older than the snapshot, which only
        // makes the next call convert again.
        let generation = snapshot_generation(&self.expression);
        if let Some((converted, conjuncts)) = self.conversion.read().as_ref()
            && *converted == generation
        {
            return (generation, Arc::clone(conjuncts));
        }

        // One scan that misses converts the filter while the others that miss wait for it, as the
        // partitions of a scan open their files at the same time. Scans that hit keep reading.
        let conversion = self.conversion.upgradable_read();
        if let Some((converted, conjuncts)) = conversion.as_ref()
            && *converted == generation
        {
            return (generation, Arc::clone(conjuncts));
        }

        let conjuncts: Conjuncts =
            match snapshot_conjuncts(&self.expression, &self.schema, self.converter.as_ref()) {
                Ok(conjuncts) => conjuncts
                    .into_iter()
                    .map(|(conjunct, max_kept_percent)| {
                        (conjunct, SwitchState::new(max_kept_percent))
                    })
                    .collect(),
                Err(err) => {
                    tracing::debug!(filter = %self.expression, %err, "dropping dynamic filter");
                    Arc::from([])
                }
            };
        *RwLockUpgradableReadGuard::upgrade(conversion) =
            Some((generation, Arc::clone(&conjuncts)));
        (generation, conjuncts)
    }

    /// Switched dynamic expressions that evaluate the current value of the filter.
    ///
    /// For a filter over several columns, such as the threshold of a TopK over several sort keys,
    /// `a < x OR (a = x AND b < y)`, there is first one expression for each column with the
    /// bounds the filter implies on it, here `a <= x`. A scan prunes zones with these and
    /// evaluates them cheaply, so the full filter only sees the rows they keep.
    pub(crate) fn dynamic_expressions(self: &Arc<Self>) -> Vec<Expression> {
        self.dynamic
            .get_or_init(|| {
                let columns = collect_columns(&self.expression)
                    .into_iter()
                    .sorted_by_key(|column| column.index())
                    .map(|column| FieldName::from(column.name()))
                    .collect::<Vec<_>>();
                let mut expressions = Vec::with_capacity(columns.len() + 1);
                if columns.len() > 1 {
                    for column in &columns {
                        expressions.push(
                            self.dynamic_expression(vec![column.clone()], Some(column.clone())),
                        );
                    }
                }
                expressions.push(self.dynamic_expression(columns, None));
                expressions
            })
            .clone()
    }

    /// Switched dynamic expressions that evaluate only the bounds that the current value of the
    /// filter implies on each of its columns, with their columns.
    pub(crate) fn bound_expressions(self: &Arc<Self>) -> Vec<(FieldName, Expression)> {
        self.bounds
            .get_or_init(|| {
                collect_columns(&self.expression)
                    .into_iter()
                    .sorted_by_key(|column| column.index())
                    .map(|column| {
                        let column = FieldName::from(column.name());
                        let bound =
                            self.dynamic_expression(vec![column.clone()], Some(column.clone()));
                        (column, bound)
                    })
                    .collect()
            })
            .clone()
    }

    /// A switched dynamic expression over `columns` that evaluates the current value of the
    /// filter, or only the bounds it implies on `bound_column`.
    fn dynamic_expression(
        self: &Arc<Self>,
        columns: Vec<FieldName>,
        bound_column: Option<FieldName>,
    ) -> Expression {
        let inputs = columns
            .iter()
            .map(|column| get_item(column.clone(), root()))
            .collect::<Vec<_>>();
        let scope_projection = pack(
            columns
                .iter()
                .enumerate()
                .map(|(idx, column)| (column.clone(), get_item(input_field_name(idx), root()))),
            Nullability::NonNullable,
        );
        let dynamic = dynamic_expression(
            Arc::new(DataFusionDynamicFilter {
                filter: Arc::clone(self),
                scope_projection,
                bound_column,
            }),
            inputs,
        );
        switch(dynamic, DYNAMIC_MAX_KEPT_PERCENT)
    }
}

/// The source of a dynamic expression for a DataFusion dynamic filter that can still change.
struct DataFusionDynamicFilter {
    filter: Arc<DynamicFilter>,
    /// The columns of the filter, as a struct over the inputs of the dynamic expression.
    scope_projection: Expression,
    /// The column whose bounds alone this source evaluates, if it does not evaluate the filter.
    bound_column: Option<FieldName>,
}

impl fmt::Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.filter.expression, f)
    }
}

impl fmt::Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(column) = &self.bound_column {
            write!(f, "bounds({column}) of ")?;
        }
        fmt::Display::fmt(&self.filter.expression, f)
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.filter.expression)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        // The producing DataFusion operator retains exact enforcement, so the parts of the filter
        // that do not convert are dropped, such as the hash table lookup next to the bounds of a
        // join, rather than the whole filter.
        let (_, conjuncts) = self.filter.current();
        let conjuncts = conjuncts.iter().map(|(conjunct, _)| conjunct);
        let conjuncts = match &self.bound_column {
            Some(column) => {
                let column = get_item(column.clone(), root());
                conjuncts
                    .filter_map(|conjunct| implied_bound(conjunct, &column))
                    .collect::<Vec<_>>()
            }
            None => conjuncts.cloned().collect(),
        };

        let predicate = and_collect(conjuncts).map_or_else(
            || lit(true),
            |predicate| replace(predicate, &root(), self.scope_projection.clone()),
        );
        let predicate = predicate
            .bind(scope)
            .and_then(|predicate| predicate.optimize_recursive())
            .map_err(|err| vortex_err!("Failed to bind dynamic filter {self}: {err}"))?;

        // A join bounds its probe keys from both sides. Each comparison would decode the keys
        // again, while a between decodes them once.
        find_between(predicate)
    }
}

/// The Vortex conjuncts of the current value of `filter`, a filter over `schema`.
///
/// The range bounds of the filter form one optional conjunct, so the scan can merge them into
/// betweens, and each membership test forms another. Conjuncts that Vortex cannot express are
/// dropped, apart from those DataFusion can evaluate on the columns they read.
fn snapshot_conjuncts(
    filter: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    converter: &dyn ExpressionConverter,
) -> DFResult<Vec<(Expression, u8)>> {
    let snapshot = PhysicalExprSimplifier::new(schema)
        .simplify(snapshot_physical_expr(Arc::clone(filter))?)?;

    let conjuncts = split_conjunction(&snapshot);
    let has_bounds = conjuncts.iter().any(|conjunct| is_range_bound(conjunct));

    let mut ranges = Vec::new();
    let mut memberships = Vec::new();
    for conjunct in conjuncts {
        // A partitioned join routes each row by its hash to the filter of its partition. The
        // bounds that hold in every partition hold for the whole filter, and are cheap to check
        // before DataFusion evaluates the routed filter on the rows they keep.
        let parts = match conjunct.downcast_ref::<df_expr::CaseExpr>() {
            Some(case) => {
                let mut parts = partition_bounds(case);
                parts.push(Arc::clone(conjunct));
                parts
            }
            None => vec![Arc::clone(conjunct)],
        };
        for part in parts {
            // The bounds of the filter already keep exactly the rows of such a list.
            if has_bounds && is_dense_integer_list(&part) {
                continue;
            }

            match converter.try_convert(&part, schema) {
                Ok(Some(converted)) if is_infallible(&converted) => {
                    if is_range_bound(&part) {
                        ranges.push(converted);
                    } else {
                        memberships.push((converted, MEMBERSHIP_MAX_KEPT_PERCENT));
                    }
                }
                Err(err) => {
                    tracing::debug!(%part, %err, "dropping dynamic filter conjunct");
                }
                _ => match DataFusionPredicate::expression(&part, schema) {
                    Some(predicate) => {
                        memberships.push((predicate, MEMBERSHIP_MAX_KEPT_PERCENT));
                    }
                    None => tracing::debug!(%part, "dropping dynamic filter conjunct"),
                },
            }
        }
    }

    Ok(and_collect(ranges)
        .map(|ranges| (ranges, RANGE_MAX_KEPT_PERCENT))
        .into_iter()
        .chain(memberships)
        .collect())
}

/// A bound on `column` that `expr` implies, as a comparison of `column` with a literal.
///
/// Every row for which `expr` is true satisfies the bound. A disjunction implies the hull of the
/// bounds of its sides, so `a < x OR (a = x AND b < y)` implies `a <= x`.
fn implied_bound(expr: &Expression, column: &Expression) -> Option<Expression> {
    let operator = *expr.as_opt::<Binary>()?;
    match operator {
        Operator::And => {
            implied_bound(expr.child(0), column).or_else(|| implied_bound(expr.child(1), column))
        }
        Operator::Or => hull(
            &implied_bound(expr.child(0), column)?,
            &implied_bound(expr.child(1), column)?,
        ),
        Operator::Eq | Operator::Lt | Operator::Lte | Operator::Gt | Operator::Gte => {
            let value = expr.child(1).as_opt::<Literal>()?;
            (expr.child(0) == column && !value.is_null()).then(|| expr.clone())
        }
        _ => None,
    }
}

/// The tightest comparison that both bounds imply, if they bound the same side of the column.
fn hull(lhs: &Expression, rhs: &Expression) -> Option<Expression> {
    let (lhs_op, lhs_value) = (*lhs.as_::<Binary>(), lhs.child(1).as_::<Literal>());
    let (rhs_op, rhs_value) = (*rhs.as_::<Binary>(), rhs.child(1).as_::<Literal>());

    let is_upper = |op| matches!(op, Operator::Eq | Operator::Lt | Operator::Lte);
    let is_lower = |op| matches!(op, Operator::Eq | Operator::Gt | Operator::Gte);
    let inclusive = |op| matches!(op, Operator::Eq | Operator::Lte | Operator::Gte);

    if lhs_op == Operator::Eq && rhs_op == Operator::Eq {
        return (lhs_value == rhs_value).then(|| lhs.clone());
    }
    let (upper, ordering) = if is_upper(lhs_op) && is_upper(rhs_op) {
        (true, lhs_value.partial_cmp(rhs_value)?)
    } else if is_lower(lhs_op) && is_lower(rhs_op) {
        (false, rhs_value.partial_cmp(lhs_value)?)
    } else {
        return None;
    };

    // The looser side wins; on equal values, the bound is inclusive if either side is.
    let (value, inclusive) = match ordering {
        Ordering::Greater => (lhs_value, inclusive(lhs_op)),
        Ordering::Less => (rhs_value, inclusive(rhs_op)),
        Ordering::Equal => (lhs_value, inclusive(lhs_op) || inclusive(rhs_op)),
    };
    let operator = match (upper, inclusive) {
        (true, true) => Operator::Lte,
        (true, false) => Operator::Lt,
        (false, true) => Operator::Gte,
        (false, false) => Operator::Gt,
    };
    Some(binary(operator, lhs.child(0).clone(), lit(value.clone())))
}

/// Whether `expr` compares an expression with a literal by order or equality.
fn is_range_bound(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<df_expr::BinaryExpr>()
        .is_some_and(|binary| {
            matches!(
                binary.op(),
                DFOperator::Eq
                    | DFOperator::Lt
                    | DFOperator::LtEq
                    | DFOperator::Gt
                    | DFOperator::GtEq
            ) && binary.right().downcast_ref::<df_expr::Literal>().is_some()
        })
}

/// Whether `expr` is an IN list of every integer between its smallest and its largest value.
///
/// A join filter bounds its keys by the smallest and the largest build key, so such a list keeps
/// the same rows as the bounds, at a higher cost.
fn is_dense_integer_list(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let Some(in_list) = expr.downcast_ref::<df_expr::InListExpr>() else {
        return false;
    };
    if in_list.negated() || in_list.is_empty() {
        return false;
    }

    let values = in_list
        .list()
        .iter()
        .map(
            |value| match value.downcast_ref::<df_expr::Literal>()?.value() {
                value if value.data_type().is_integer() => {
                    match value.cast_to(&DataType::Int64).ok()? {
                        ScalarValue::Int64(Some(value)) => Some(value),
                        _ => None,
                    }
                }
                _ => None,
            },
        )
        .collect::<Option<Vec<_>>>();
    let Some(values) = values else {
        return false;
    };

    let distinct = values.iter().unique().count();
    let (Some(min), Some(max)) = (values.iter().min(), values.iter().max()) else {
        return false;
    };
    max.abs_diff(*min) < distinct as u64
}

/// The bounds of each key that every partition of a partitioned join filter implies, as
/// `key >= min AND key <= max` over the smallest minimum and the largest maximum.
///
/// The filter is `CASE hash(keys) % n WHEN i THEN filter_i ... ELSE false END`, where `filter_i`
/// holds the bounds of partition `i`. A partition without rows has no branch or a `false` branch,
/// so it bounds nothing. A filter whose default is not `false`, as when a build partition was
/// canceled, implies no bounds.
fn partition_bounds(case: &df_expr::CaseExpr) -> Vec<Arc<dyn PhysicalExpr>> {
    let else_is_false = case.else_expr().is_none_or(is_false);
    if !else_is_false {
        return Vec::new();
    }

    // The bounds of each key so far.
    let mut hull: Option<Vec<KeyBounds>> = None;
    for (_, then) in case.when_then_expr() {
        if is_false(then) {
            continue;
        }
        let bounds = branch_bounds(then);
        hull = Some(match hull {
            None => bounds,
            Some(hull) => hull
                .into_iter()
                .filter_map(|(key, min, max)| {
                    let (_, other_min, other_max) =
                        bounds.iter().find(|(other, ..)| other == &key)?;
                    let min = match other_min.partial_cmp(&min)? {
                        Ordering::Less => other_min.clone(),
                        _ => min,
                    };
                    let max = match other_max.partial_cmp(&max)? {
                        Ordering::Greater => other_max.clone(),
                        _ => max,
                    };
                    Some((key, min, max))
                })
                .collect(),
        });
    }

    hull.unwrap_or_default()
        .into_iter()
        .flat_map(|(key, min, max)| {
            [
                Arc::new(df_expr::BinaryExpr::new(
                    Arc::clone(&key),
                    DFOperator::GtEq,
                    df_expr::lit(min),
                )) as Arc<dyn PhysicalExpr>,
                Arc::new(df_expr::BinaryExpr::new(
                    key,
                    DFOperator::LtEq,
                    df_expr::lit(max),
                )),
            ]
        })
        .collect()
}

/// A join key with its minimum and maximum.
type KeyBounds = (Arc<dyn PhysicalExpr>, ScalarValue, ScalarValue);

/// The `key >= min AND key <= max` bounds in the conjuncts of a partition filter.
fn branch_bounds(filter: &Arc<dyn PhysicalExpr>) -> Vec<KeyBounds> {
    let comparisons = split_conjunction(filter)
        .into_iter()
        .filter_map(|conjunct| {
            let binary = conjunct.downcast_ref::<df_expr::BinaryExpr>()?;
            let value = binary.right().downcast_ref::<df_expr::Literal>()?.value();
            (!value.is_null()).then(|| (binary.left(), *binary.op(), value))
        })
        .collect::<Vec<_>>();
    comparisons
        .iter()
        .filter(|(_, op, _)| *op == DFOperator::GtEq)
        .filter_map(|(key, _, min)| {
            let (_, _, max) = comparisons
                .iter()
                .find(|(other, op, _)| *op == DFOperator::LtEq && other == key)?;
            Some((Arc::clone(key), (*min).clone(), (*max).clone()))
        })
        .collect()
}

fn is_false(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<df_expr::Literal>()
        .is_some_and(|literal| literal.value() == &ScalarValue::Boolean(Some(false)))
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use rstest::rstest;
    use vortex::dtype::PType;
    use vortex::expr::eq;
    use vortex::expr::gt_eq;
    use vortex::expr::lt;
    use vortex::expr::lt_eq;
    use vortex::scalar_fn::fns::between::Between;
    use vortex::scalar_fn::fns::switch::Switch;

    use super::*;
    use crate::convert::DefaultExpressionConverter;

    fn col(name: &str) -> Expression {
        get_item(name, root())
    }

    fn column(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::Column::new(name, index))
    }

    fn compare(lhs: Arc<dyn PhysicalExpr>, op: DFOperator, value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::BinaryExpr::new(lhs, op, df_expr::lit(value)))
    }

    fn and(lhs: Arc<dyn PhysicalExpr>, rhs: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::BinaryExpr::new(lhs, DFOperator::And, rhs))
    }

    fn bounds(min: i32, max: i32) -> Arc<dyn PhysicalExpr> {
        and(
            compare(column("a", 0), DFOperator::GtEq, min),
            compare(column("a", 0), DFOperator::LtEq, max),
        )
    }

    fn partitioned(
        branches: Vec<Arc<dyn PhysicalExpr>>,
        default: bool,
    ) -> DFResult<df_expr::CaseExpr> {
        let routing = Arc::new(df_expr::BinaryExpr::new(
            column("a", 0),
            DFOperator::Modulo,
            df_expr::lit(branches.len() as u64),
        ));
        df_expr::CaseExpr::try_new(
            Some(routing),
            branches
                .into_iter()
                .enumerate()
                .map(|(idx, then)| (df_expr::lit(idx as u64), then))
                .collect(),
            Some(df_expr::lit(default)),
        )
    }

    #[rstest]
    #[case::topk_threshold(
        vortex::expr::or(
            lt(col("a"), lit(5)),
            vortex::expr::and(eq(col("a"), lit(5)), lt(col("b"), lit("x")))
        ),
        Some(lt_eq(col("a"), lit(5)))
    )]
    #[case::looser_upper_bound(
        vortex::expr::or(lt(col("a"), lit(5)), lt_eq(col("a"), lit(3))),
        Some(lt(col("a"), lit(5)))
    )]
    #[case::lower_bounds(
        vortex::expr::or(vortex::expr::gt(col("a"), lit(5)), gt_eq(col("a"), lit(7))),
        Some(vortex::expr::gt(col("a"), lit(5)))
    )]
    #[case::conjunction(
        vortex::expr::and(lt(col("b"), lit(1)), gt_eq(col("a"), lit(2))),
        Some(gt_eq(col("a"), lit(2)))
    )]
    #[case::opposite_sides(
        vortex::expr::or(lt(col("a"), lit(5)), vortex::expr::gt(col("a"), lit(9))),
        None
    )]
    #[case::other_column(vortex::expr::or(lt(col("a"), lit(5)), lt(col("b"), lit(9))), None)]
    fn implied_bounds(#[case] expr: Expression, #[case] expected: Option<Expression>) {
        assert_eq!(implied_bound(&expr, &col("a")), expected);
    }

    #[test]
    fn partition_bounds_cover_every_partition() -> DFResult<()> {
        let case = partitioned(
            vec![bounds(5, 10), df_expr::lit(false), bounds(1, 7)],
            false,
        )?;
        let hull = partition_bounds(&case);
        let expected: Vec<Arc<dyn PhysicalExpr>> = vec![
            compare(column("a", 0), DFOperator::GtEq, 1),
            compare(column("a", 0), DFOperator::LtEq, 10),
        ];
        assert_eq!(hull, expected);
        Ok(())
    }

    #[test]
    fn partition_bounds_need_a_false_default() -> DFResult<()> {
        let case = partitioned(vec![bounds(5, 10), bounds(1, 7)], true)?;
        assert!(partition_bounds(&case).is_empty());
        Ok(())
    }

    #[test]
    fn partition_bounds_need_bounds_in_every_partition() -> DFResult<()> {
        let case = partitioned(vec![bounds(5, 10), df_expr::lit(true)], false)?;
        assert!(partition_bounds(&case).is_empty());
        Ok(())
    }

    #[test]
    fn snapshots_split_into_range_bounds_and_memberships() -> DFResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        // `IS NOT DISTINCT FROM` stands in for a hash table lookup: Vortex cannot express it.
        let membership: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
            column("a", 0),
            DFOperator::IsNotDistinctFrom,
            df_expr::lit(3i32),
        ));
        let filter = and(bounds(1, 7), membership);
        let converter = DefaultExpressionConverter::default();

        let conjuncts = snapshot_conjuncts(&filter, &schema, &converter)?;
        assert_eq!(conjuncts.len(), 2);
        let (ranges, ranges_percent) = &conjuncts[0];
        assert_eq!(
            ranges,
            &vortex::expr::and(
                gt_eq(get_item("a", root()), lit(1i32)),
                lt_eq(get_item("a", root()), lit(7i32))
            )
        );
        assert_eq!(*ranges_percent, RANGE_MAX_KEPT_PERCENT);
        assert!(conjuncts[1].0.is::<DataFusionPredicate>());
        assert!(
            !conjuncts
                .iter()
                .any(|(conjunct, _)| conjunct.is::<Switch>())
        );
        Ok(())
    }

    #[test]
    fn merges_key_bounds_into_a_between() -> VortexResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let source = DataFusionDynamicFilter {
            filter: DynamicFilter::new(
                bounds(1, 5),
                schema,
                Arc::new(DefaultExpressionConverter::default()),
            ),
            scope_projection: root(),
            bound_column: None,
        };
        let scope = DType::struct_(
            [("a", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        );

        let current = source.current(&scope)?;
        assert!(current.is::<Between>(), "{current}");
        Ok(())
    }
}
