// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ScalarValue;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::CaseExpr;
use datafusion_physical_expr::expressions::InListExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::lit as df_lit;
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
use vortex_utils::aliases::hash_set::HashSet;

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
        // A join bounds its probe keys from both sides. Each comparison would decode the keys
        // again, while a between decodes them once.
        find_between(expression.bind(scope)?)
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
            .flat_map(|conjunct| match self.convert(conjunct) {
                Some(converted) => vec![converted],
                // A partitioned join routes each key to its partition's filter with a hash,
                // which Vortex can't evaluate. The bounds and key lists every partition shares
                // still hold for every row the join keeps.
                None => partition_hull(conjunct, &self.schema)
                    .iter()
                    .filter_map(|implied| self.convert(implied))
                    .collect(),
            })
            .reduce(|left, right| binary(Operator::And, left, right))
    }

    fn convert(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<Expression> {
        self.converter
            .try_convert(expr, &self.schema)
            .ok()
            .flatten()
            .filter(is_infallible)
    }
}

/// Distinct keys a hull of partitioned join key lists holds before it stops listing them.
const MAX_HULL_MEMBERS: usize = 256;

/// Predicates that every row kept by a partitioned hash join's filter satisfies.
///
/// A join with partitioned build sides filters its probe side with
/// `CASE hash_repartition(keys) % n WHEN i THEN <partition i's filter> ... ELSE false END`, where
/// each partition's filter bounds the keys and tests them against the partition's keys. A kept
/// row satisfies one partition's filter, so a key bound or key list that every partition has
/// still holds, loosened to cover all of them.
///
/// Returns no predicates for anything else, or when the partitions share no bound.
fn partition_hull(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Vec<Arc<dyn PhysicalExpr>> {
    let Some(case) = expr.downcast_ref::<CaseExpr>() else {
        return Vec::new();
    };
    // Only the routing form compares one value against each WHEN.
    if case.expr().is_none() {
        return Vec::new();
    }
    let hull = case
        .when_then_expr()
        .iter()
        .map(|(_, then)| then)
        .chain(case.else_expr())
        .map(KeyConstraint::of_partition)
        .reduce(|left, right| match (left, right) {
            (Some(left), Some(right)) => Some(KeyConstraint::hull(left, right)),
            (side, None) | (None, side) => side,
        });
    match hull {
        // A CASE that keeps no row in any partition, e.g. when every build partition is empty.
        Some(None) => vec![df_lit(false)],
        Some(Some(constraints)) => constraints
            .into_iter()
            .flat_map(|constraint| constraint.into_predicates(schema))
            .collect(),
        None => Vec::new(),
    }
}

/// A key's bound on one side, and whether the bound is inclusive.
type KeyBound = (ScalarValue, bool);

/// What every row a filter keeps satisfies for one key, unless the key is null.
struct KeyConstraint {
    key: Arc<dyn PhysicalExpr>,
    lower: Option<KeyBound>,
    upper: Option<KeyBound>,
    members: Option<HashSet<ScalarValue>>,
}

impl KeyConstraint {
    fn new(key: &Arc<dyn PhysicalExpr>) -> Self {
        Self {
            key: Arc::clone(key),
            lower: None,
            upper: None,
            members: None,
        }
    }

    /// The constraints one partition's filter places on its keys, or `None` if it keeps no row.
    fn of_partition(filter: &Arc<dyn PhysicalExpr>) -> Option<Vec<Self>> {
        let mut constraints: Vec<Self> = Vec::new();
        let mut constraint = |key: &Arc<dyn PhysicalExpr>| -> usize {
            match constraints
                .iter()
                .position(|c| c.key.as_ref() == key.as_ref())
            {
                Some(idx) => idx,
                None => {
                    constraints.push(Self::new(key));
                    constraints.len() - 1
                }
            }
        };
        let mut found = Vec::new();
        for conjunct in split_conjunction(filter) {
            if let Some(literal) = conjunct.downcast_ref::<Literal>() {
                if !matches!(literal.value(), ScalarValue::Boolean(Some(true))) {
                    return None;
                }
            } else if let Some(in_list) = conjunct.downcast_ref::<InListExpr>() {
                if in_list.negated() {
                    continue;
                }
                let Some(members) = list_members(in_list) else {
                    continue;
                };
                // Only null keys could match, and they never pass a membership test.
                if members.is_empty() {
                    return None;
                }
                found.push((constraint(in_list.expr()), Found::Members(members)));
            } else if let Some((key, op, value)) = key_comparison(conjunct) {
                let idx = constraint(key);
                if matches!(op, DFOperator::Gt | DFOperator::GtEq | DFOperator::Eq) {
                    found.push((idx, Found::Lower((value.clone(), op != DFOperator::Gt))));
                }
                if matches!(op, DFOperator::Lt | DFOperator::LtEq | DFOperator::Eq) {
                    found.push((idx, Found::Upper((value, op != DFOperator::Lt))));
                }
            }
        }
        for (idx, found) in found {
            let constraint = &mut constraints[idx];
            match found {
                Found::Lower(bound) => {
                    constraint.lower = tighter(constraint.lower.take(), bound, Ordering::Greater)
                }
                Found::Upper(bound) => {
                    constraint.upper = tighter(constraint.upper.take(), bound, Ordering::Less)
                }
                Found::Members(members) => {
                    if constraint
                        .members
                        .as_ref()
                        .is_none_or(|existing| members.len() < existing.len())
                    {
                        constraint.members = Some(members);
                    }
                }
            }
        }
        Some(constraints)
    }

    /// The constraints that hold when either side's do: a key keeps a bound or a key list only
    /// when both sides have one, loosened to cover both.
    fn hull(left: Vec<Self>, right: Vec<Self>) -> Vec<Self> {
        left.into_iter()
            .filter_map(|left| {
                let right = right
                    .iter()
                    .find(|right| right.key.as_ref() == left.key.as_ref())?;
                let members = left
                    .members
                    .zip(right.members.as_ref())
                    .map(|(mut members, other)| {
                        members.extend(other.iter().cloned());
                        members
                    })
                    .filter(|members| members.len() <= MAX_HULL_MEMBERS);
                let constraint = Self {
                    key: left.key,
                    lower: looser(left.lower, right.lower.as_ref(), Ordering::Less),
                    upper: looser(left.upper, right.upper.as_ref(), Ordering::Greater),
                    members,
                };
                (constraint.lower.is_some()
                    || constraint.upper.is_some()
                    || constraint.members.is_some())
                .then_some(constraint)
            })
            .collect()
    }

    fn into_predicates(self, schema: &Schema) -> Vec<Arc<dyn PhysicalExpr>> {
        let compare = |op, value| -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(Arc::clone(&self.key), op, df_lit(value)))
        };
        let mut predicates = Vec::new();
        if let Some((value, inclusive)) = self.lower.clone() {
            predicates.push(compare(
                if inclusive {
                    DFOperator::GtEq
                } else {
                    DFOperator::Gt
                },
                value,
            ));
        }
        if let Some((value, inclusive)) = self.upper.clone() {
            predicates.push(compare(
                if inclusive {
                    DFOperator::LtEq
                } else {
                    DFOperator::Lt
                },
                value,
            ));
        }
        if let Some(members) = &self.members {
            let mut members = members.iter().collect::<Vec<_>>();
            members.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
            let list = members
                .into_iter()
                .map(|value| df_lit(value.clone()))
                .collect();
            if let Ok(in_list) = InListExpr::try_new(Arc::clone(&self.key), list, false, schema) {
                predicates.push(Arc::new(in_list));
            }
        }
        predicates
    }
}

/// A constraint one conjunct of a partition's filter places on a key.
enum Found {
    Lower(KeyBound),
    Upper(KeyBound),
    Members(HashSet<ScalarValue>),
}

/// The distinct non-null values of a list of literals, or `None` if the list has anything else
/// or more than [`MAX_HULL_MEMBERS`] distinct values.
fn list_members(in_list: &InListExpr) -> Option<HashSet<ScalarValue>> {
    let mut members = HashSet::new();
    for element in in_list.list() {
        let value = element.downcast_ref::<Literal>()?.value();
        if !value.is_null() && members.insert(value.clone()) && members.len() > MAX_HULL_MEMBERS {
            return None;
        }
    }
    Some(members)
}

/// A comparison of a key against a non-null literal, as `(key, op, literal)` with the key on the
/// left.
fn key_comparison(
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<(&Arc<dyn PhysicalExpr>, DFOperator, ScalarValue)> {
    let binary = expr.downcast_ref::<BinaryExpr>()?;
    let op = *binary.op();
    if !matches!(
        op,
        DFOperator::Eq | DFOperator::Lt | DFOperator::LtEq | DFOperator::Gt | DFOperator::GtEq
    ) {
        return None;
    }
    let literal = |expr: &Arc<dyn PhysicalExpr>| {
        expr.downcast_ref::<Literal>()
            .map(|literal| literal.value().clone())
            .filter(|value| !value.is_null())
    };
    if let Some(value) = literal(binary.right()) {
        return Some((binary.left(), op, value));
    }
    let value = literal(binary.left())?;
    Some((binary.right(), op.swap()?, value))
}

/// The tighter of two bounds on the same side, where `tighter` is how the tighter value orders
/// against the other. Bounds that don't compare keep the existing one.
fn tighter(existing: Option<KeyBound>, bound: KeyBound, tighter: Ordering) -> Option<KeyBound> {
    let Some(existing) = existing else {
        return Some(bound);
    };
    Some(match bound.0.partial_cmp(&existing.0) {
        Some(Ordering::Equal) => (bound.0, bound.1 && existing.1),
        Some(order) if order == tighter => bound,
        _ => existing,
    })
}

/// The looser of two bounds on the same side, where `looser` is how the looser value orders
/// against the other, or `None` if either side is unbounded or the bounds don't compare.
fn looser(left: Option<KeyBound>, right: Option<&KeyBound>, looser: Ordering) -> Option<KeyBound> {
    let (left, right) = (left?, right?);
    match left.0.partial_cmp(&right.0)? {
        Ordering::Equal => Some((left.0, left.1 || right.1)),
        order if order == looser => Some(left),
        _ => Some(right.clone()),
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
    use datafusion_physical_expr::expressions::CaseExpr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::InListExpr;
    use datafusion_physical_expr::expressions::Literal;
    use datafusion_physical_expr::expressions::NoOp;
    use datafusion_physical_expr::expressions::lit as df_lit;
    use rstest::rstest;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::expr::Expression;
    use vortex::expr::root;
    use vortex::scalar_fn::fns::between::Between;
    use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;

    use super::DataFusionDynamicFilter;
    use super::MAX_HULL_MEMBERS;
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

    fn at_most(value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("key", 0)),
            DFOperator::LtEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(value)))),
        ))
    }

    fn filter(expression: Arc<dyn PhysicalExpr>) -> DataFusionDynamicFilter {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        DataFusionDynamicFilter {
            expression,
            schema,
            scope_projection: root(),
            converter: Arc::new(DefaultExpressionConverter::default()),
        }
    }

    fn convert(expression: Arc<dyn PhysicalExpr>) -> Option<Expression> {
        filter(Arc::clone(&expression)).convert_conjuncts(&expression)
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

    #[test]
    fn merges_key_bounds_into_a_between() -> VortexResult<()> {
        let scope = DType::struct_(
            [(
                "key",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )],
            Nullability::NonNullable,
        );
        let current = filter(and(at_least(1), at_most(5))).current(&scope)?;
        assert!(current.is::<Between>(), "{current}");
        Ok(())
    }

    fn key_in(values: impl IntoIterator<Item = i32>) -> Arc<dyn PhysicalExpr> {
        let schema = Schema::new(vec![Field::new("key", DataType::Int32, false)]);
        let list = values
            .into_iter()
            .map(|value| df_lit(ScalarValue::Int32(Some(value))))
            .collect();
        Arc::new(
            InListExpr::try_new(Arc::new(Column::new("key", 0)), list, false, &schema)
                .expect("valid IN list"),
        )
    }

    /// A partitioned hash join's filter: the key's hash picks its partition's filter.
    fn routed(partitions: Vec<Arc<dyn PhysicalExpr>>) -> Arc<dyn PhysicalExpr> {
        let route = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("key", 0)),
            DFOperator::Modulo,
            df_lit(ScalarValue::UInt64(Some(partitions.len() as u64))),
        ));
        let when_then = partitions
            .into_iter()
            .enumerate()
            .map(|(idx, then)| (df_lit(ScalarValue::UInt64(Some(idx as u64))), then))
            .collect();
        Arc::new(
            CaseExpr::try_new(Some(route), when_then, Some(df_lit(false))).expect("valid CASE"),
        )
    }

    fn partition(lower: i32, upper: i32, keys: &[i32]) -> Arc<dyn PhysicalExpr> {
        and(
            and(at_least(lower), at_most(upper)),
            key_in(keys.iter().copied()),
        )
    }

    fn hash_lookup_partition(lower: i32, upper: i32) -> Arc<dyn PhysicalExpr> {
        and(and(at_least(lower), at_most(upper)), Arc::new(NoOp::new()))
    }

    fn conjunction(predicates: Vec<Arc<dyn PhysicalExpr>>) -> Arc<dyn PhysicalExpr> {
        predicates.into_iter().reduce(and).expect("a predicate")
    }

    #[rstest]
    #[case::bounds_and_keys(
        vec![partition(1, 5, &[1, 3, 5]), partition(10, 20, &[10, 12, 20, 12])],
        Some(conjunction(vec![at_least(1), at_most(20), key_in([1, 3, 5, 10, 12, 20])])),
    )]
    #[case::hash_lookup_keeps_bounds(
        vec![hash_lookup_partition(1, 5), partition(10, 20, &[10, 20])],
        Some(and(at_least(1), at_most(20))),
    )]
    #[case::empty_partition(
        vec![partition(1, 5, &[1, 5]), df_lit(false), partition(10, 20, &[10, 20])],
        Some(conjunction(vec![at_least(1), at_most(20), key_in([1, 5, 10, 20])])),
    )]
    #[case::unbounded_partition(vec![partition(1, 5, &[1, 5]), df_lit(true)], None)]
    #[case::too_many_keys(
        vec![
            partition(0, 200, &(0..200).collect::<Vec<_>>()),
            partition(300, 500, &(300..500).collect::<Vec<_>>()),
        ],
        Some(and(at_least(0), at_most(500))),
    )]
    #[case::every_partition_empty(vec![df_lit(false), df_lit(false)], Some(df_lit(false)))]
    fn partitioned_join_filter_keeps_shared_bounds(
        #[case] partitions: Vec<Arc<dyn PhysicalExpr>>,
        #[case] expected: Option<Arc<dyn PhysicalExpr>>,
    ) {
        assert_eq!(convert(routed(partitions)), expected.and_then(convert));
    }

    #[test]
    fn hull_drops_key_lists_past_the_limit() {
        let limit = i32::try_from(MAX_HULL_MEMBERS).expect("small limit");
        let half = limit / 2;
        let at_limit = routed(vec![
            partition(0, half, &(0..half).collect::<Vec<_>>()),
            partition(half, limit, &(half..limit).collect::<Vec<_>>()),
        ]);
        assert_eq!(
            convert(at_limit),
            convert(conjunction(vec![
                at_least(0),
                at_most(limit),
                key_in(0..limit)
            ]))
        );
    }
}
