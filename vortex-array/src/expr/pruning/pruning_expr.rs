// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;
use std::iter;

use itertools::Itertools;
use vortex_utils::aliases::hash_map::HashMap;

use super::relation::Relation;
use crate::dtype::Field;
use crate::dtype::FieldName;
use crate::dtype::FieldPath;
use crate::dtype::FieldPathSet;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::VTableExt;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::binary::and_collect;
use crate::expr::exprs::binary::eq;
use crate::expr::exprs::get_item::get_item;
use crate::expr::exprs::literal::lit;
use crate::expr::exprs::operators::Operator;
use crate::expr::exprs::root::root;
use crate::expr::stats::Stat;

pub type RequiredStats = Relation<FieldPath, Stat>;

// A catalog that return a stat column whenever it is required, tracking all accessed
// stats and returning them later.
#[derive(Default)]
pub(crate) struct TrackingStatsCatalog {
    usage: RefCell<HashMap<(FieldPath, Stat), Expression>>,
}

impl TrackingStatsCatalog {
    /// Consume the catalog, yielding a map of field statistics that were required
    /// for each expression.
    fn into_usages(self) -> HashMap<(FieldPath, Stat), Expression> {
        self.usage.into_inner()
    }
}

// A catalog that return a stat column if it exists in the given scope.
struct ScopeStatsCatalog<'a> {
    inner: TrackingStatsCatalog,
    available_stats: &'a FieldPathSet,
}

impl StatsCatalog for ScopeStatsCatalog<'_> {
    fn stats_ref(&self, field_path: &FieldPath, stat: Stat) -> Option<Expression> {
        let stat_path = field_path.clone().push(stat.name());

        if self.available_stats.contains(&stat_path) {
            self.inner.stats_ref(field_path, stat)
        } else {
            None
        }
    }
}

impl StatsCatalog for TrackingStatsCatalog {
    fn stats_ref(&self, field_path: &FieldPath, stat: Stat) -> Option<Expression> {
        let mut expr = root();
        let name = field_path_stat_field_name(field_path, stat);
        expr = get_item(name, expr);
        self.usage
            .borrow_mut()
            .insert((field_path.clone(), stat), expr.clone());
        Some(expr)
    }
}

#[doc(hidden)]
pub fn field_path_stat_field_name(field_path: &FieldPath, stat: Stat) -> FieldName {
    field_path
        .parts()
        .iter()
        .map(|f| match f {
            Field::Name(n) => n.as_ref(),
            Field::ElementType => todo!("element type not currently handled"),
        })
        .chain(iter::once(stat.name()))
        .join("_")
        .into()
}

/// Build a pruning expr mask, using an existing set of stats.
/// The available stats are provided as a set of [`FieldPath`].
///
/// A pruning expression is one that returns `true` for all positions where the original expression
/// cannot hold, and false if it cannot be determined from stats alone whether the positions can
/// be pruned.
///
/// If the falsification logic attempts to access an unknown stat,
/// this function will return `None`.
pub fn checked_pruning_expr(
    expr: &Expression,
    available_stats: &FieldPathSet,
) -> Option<(Expression, RequiredStats)> {
    let catalog = ScopeStatsCatalog {
        inner: Default::default(),
        available_stats,
    };

    let expr = expr.stat_falsification(&catalog)?;

    // TODO(joe): filter access by used exprs
    let mut relation: Relation<FieldPath, Stat> = Relation::new();
    for ((field_path, stat), _) in catalog.inner.into_usages() {
        relation.insert(field_path, stat)
    }

    Some((expr, relation))
}

/// Push logical NOT inward through a filter expression, returning the negated expression.
///
/// This handles comparison operators (via `Operator::inverse`) and boolean connectives
/// (via De Morgan's laws). Returns `None` for expressions that cannot be negated
/// (e.g. arithmetic, LIKE, BETWEEN), which is conservative — we simply can't prove
/// satisfaction for those expressions.
pub fn push_not_inward(expr: &Expression) -> Option<Expression> {
    let scalar_fn = expr.scalar_fn();
    let op = scalar_fn.as_opt::<Binary>()?;

    match op {
        // Comparison operators: negate via inverse
        Operator::Eq
        | Operator::NotEq
        | Operator::Gt
        | Operator::Gte
        | Operator::Lt
        | Operator::Lte => {
            let negated_op = op.inverse()?;
            Some(
                Binary
                    .try_new_expr(negated_op, [expr.child(0).clone(), expr.child(1).clone()])
                    .ok()?,
            )
        }
        // De Morgan: NOT(AND(p, q)) = OR(NOT(p), NOT(q))
        Operator::And => {
            let left = push_not_inward(expr.child(0))?;
            let right = push_not_inward(expr.child(1))?;
            Some(Binary.try_new_expr(Operator::Or, [left, right]).ok()?)
        }
        // De Morgan: NOT(OR(p, q)) = AND(NOT(p), NOT(q))
        Operator::Or => {
            let left = push_not_inward(expr.child(0))?;
            let right = push_not_inward(expr.child(1))?;
            Some(Binary.try_new_expr(Operator::And, [left, right]).ok()?)
        }
        // Arithmetic operators cannot be negated
        Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => None,
    }
}

/// Build a satisfaction expression: returns `true` for zones where the filter is provably
/// satisfied for ALL rows.
///
/// This works by negating the filter and reusing the falsification infrastructure:
/// `satisfaction(filter) = falsification(NOT(filter))`.
///
/// Additionally, for zones to be fully satisfied, there must be no null values in any
/// referenced column (since nulls produce NULL, not TRUE, from the filter). We AND in
/// `null_count == 0` checks for each field that has null_count stats available.
pub fn checked_satisfaction_expr(
    expr: &Expression,
    available_stats: &FieldPathSet,
) -> Option<(Expression, RequiredStats)> {
    let negated = push_not_inward(expr)?;
    let (pruning_expr, mut required_stats) = checked_pruning_expr(&negated, available_stats)?;

    // Collect null_count checks for all field paths that have null_count stats available.
    // Satisfaction requires no nulls — null values produce NULL (not TRUE) from the filter.
    let field_paths: Vec<_> = required_stats.map().keys().cloned().collect();
    let null_count_checks: Vec<Expression> = field_paths
        .into_iter()
        .filter_map(|field_path| {
            let null_count_stat_path = field_path.clone().push(Stat::NullCount.name());
            available_stats.contains(&null_count_stat_path).then(|| {
                required_stats.insert(field_path.clone(), Stat::NullCount);
                let null_count_col = get_item(
                    field_path_stat_field_name(&field_path, Stat::NullCount),
                    root(),
                );
                eq(null_count_col, lit(0u64))
            })
        })
        .collect();

    let final_expr = if let Some(null_check) = and_collect(null_count_checks) {
        use crate::expr::exprs::binary::and;
        and(null_check, pruning_expr)
    } else {
        pruning_expr
    };

    Some((final_expr, required_stats))
}

#[cfg(test)]
mod tests {
    use rstest::fixture;
    use rstest::rstest;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::HashMap;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::FieldPath;
    use crate::dtype::FieldPathSet;
    use crate::dtype::Nullability;
    use crate::dtype::StructFields;
    use crate::expr::BetweenOptions;
    use crate::expr::StrictComparison;
    use crate::expr::exprs::between::between;
    use crate::expr::exprs::binary::and;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::binary::gt;
    use crate::expr::exprs::binary::gt_eq;
    use crate::expr::exprs::binary::lt;
    use crate::expr::exprs::binary::lt_eq;
    use crate::expr::exprs::binary::not_eq;
    use crate::expr::exprs::binary::or;
    use crate::expr::exprs::cast::cast;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;
    use crate::expr::pruning::checked_pruning_expr;
    use crate::expr::pruning::field_path_stat_field_name;
    use crate::expr::stats::Stat;

    // Implement some checked pruning expressions.
    #[fixture]
    fn available_stats() -> FieldPathSet {
        let field_a = FieldPath::from_name("a");
        let field_b = FieldPath::from_name("b");

        FieldPathSet::from_iter([
            field_a.clone().push(Stat::Min.name()),
            field_a.push(Stat::Max.name()),
            field_b.clone().push(Stat::Min.name()),
            field_b.push(Stat::Max.name()),
        ])
    }

    #[rstest]
    pub fn pruning_equals(available_stats: FieldPathSet) {
        let name = FieldName::from("a");
        let literal_eq = lit(42);
        let eq_expr = eq(get_item("a", root()), literal_eq.clone());
        let (converted, _refs) = checked_pruning_expr(&eq_expr, &available_stats).unwrap();
        let expected_expr = or(
            gt(
                get_item(
                    field_path_stat_field_name(&FieldPath::from_name(name.clone()), Stat::Min),
                    root(),
                ),
                literal_eq.clone(),
            ),
            gt(
                literal_eq,
                col(field_path_stat_field_name(
                    &FieldPath::from_name(name),
                    Stat::Max,
                )),
            ),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_equals_column(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let eq_expr = eq(col(column.clone()), col(other_col.clone()));

        let (converted, refs) = checked_pruning_expr(&eq_expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([
                (
                    FieldPath::from_name(column.clone()),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    FieldPath::from_name(other_col.clone()),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = or(
            gt(
                col(field_path_stat_field_name(
                    &FieldPath::from_name(column.clone()),
                    Stat::Min,
                )),
                col(field_path_stat_field_name(
                    &FieldPath::from_name(other_col.clone()),
                    Stat::Max,
                )),
            ),
            gt(
                col(field_path_stat_field_name(
                    &FieldPath::from_name(other_col),
                    Stat::Min,
                )),
                col(field_path_stat_field_name(
                    &FieldPath::from_name(column),
                    Stat::Max,
                )),
            ),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_not_equals_column(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let not_eq_expr = not_eq(col(column.clone()), col(other_col.clone()));

        let (converted, refs) = checked_pruning_expr(&not_eq_expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([
                (
                    FieldPath::from_name(column.clone()),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    FieldPath::from_name(other_col.clone()),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = and(
            eq(
                col(field_path_stat_field_name(
                    &FieldPath::from_name(column.clone()),
                    Stat::Min,
                )),
                col(field_path_stat_field_name(
                    &FieldPath::from_name(other_col.clone()),
                    Stat::Max,
                )),
            ),
            eq(
                col(field_path_stat_field_name(
                    &FieldPath::from_name(column),
                    Stat::Max,
                )),
                col(field_path_stat_field_name(
                    &FieldPath::from_name(other_col),
                    Stat::Min,
                )),
            ),
        );

        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_gt_column(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = col(other_col.clone());
        let not_eq_expr = gt(col(column.clone()), other_expr);

        let (converted, refs) = checked_pruning_expr(&not_eq_expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([
                (
                    FieldPath::from_name(column.clone()),
                    HashSet::from_iter([Stat::Max])
                ),
                (
                    FieldPath::from_name(other_col.clone()),
                    HashSet::from_iter([Stat::Min])
                )
            ])
        );
        let expected_expr = lt_eq(
            col(field_path_stat_field_name(
                &FieldPath::from_name(column),
                Stat::Max,
            )),
            col(field_path_stat_field_name(
                &FieldPath::from_name(other_col),
                Stat::Min,
            )),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_gt_value(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = gt(col(column.clone()), other_col.clone());

        let (converted, refs) = checked_pruning_expr(&not_eq_expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([(
                FieldPath::from_name(column.clone()),
                HashSet::from_iter([Stat::Max])
            ),])
        );
        let expected_expr = lt_eq(
            col(field_path_stat_field_name(
                &FieldPath::from_name(column),
                Stat::Max,
            )),
            other_col,
        );
        assert_eq!(&converted, &(expected_expr));
    }

    #[rstest]
    pub fn pruning_lt_column(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = col(other_col.clone());
        let not_eq_expr = lt(col(column.clone()), other_expr);

        let (converted, refs) = checked_pruning_expr(&not_eq_expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([
                (
                    FieldPath::from_name(column.clone()),
                    HashSet::from_iter([Stat::Min])
                ),
                (
                    FieldPath::from_name(other_col.clone()),
                    HashSet::from_iter([Stat::Max])
                )
            ])
        );
        let expected_expr = gt_eq(
            col(field_path_stat_field_name(
                &FieldPath::from_name(column),
                Stat::Min,
            )),
            col(field_path_stat_field_name(
                &FieldPath::from_name(other_col),
                Stat::Max,
            )),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_lt_value(available_stats: FieldPathSet) {
        // expression   => a < 42
        // pruning expr => a.min >= 42
        let expr = lt(col("a"), lit(42));

        let (converted, refs) = checked_pruning_expr(&expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([(FieldPath::from_name("a"), HashSet::from_iter([Stat::Min]))])
        );
        assert_eq!(&converted, &gt_eq(col("a_min"), lit(42)));
    }

    #[rstest]
    fn pruning_identity(available_stats: FieldPathSet) {
        let expr = or(lt(col("a"), lit(10)), gt(col("a"), lit(50)));

        let (predicate, _) = checked_pruning_expr(&expr, &available_stats).unwrap();

        let expected_expr = and(gt_eq(col("a_min"), lit(10)), lt_eq(col("a_max"), lit(50)));
        assert_eq!(&predicate.to_string(), &expected_expr.to_string());
    }
    #[rstest]
    pub fn pruning_and_or_operators(available_stats: FieldPathSet) {
        // Test case: a > 10 AND a < 50
        let column = FieldName::from("a");
        let and_expr = and(gt(col(column.clone()), lit(10)), lt(col(column), lit(50)));
        let (predicate, _) = checked_pruning_expr(&and_expr, &available_stats).unwrap();

        // Expected: a_max <= 10 OR a_min >= 50
        assert_eq!(
            &predicate,
            &or(
                lt_eq(col(FieldName::from("a_max")), lit(10)),
                gt_eq(col(FieldName::from("a_min")), lit(50)),
            ),
        );
    }

    #[rstest]
    fn test_gt_eq_with_booleans(available_stats: FieldPathSet) {
        // Consider this unusual, but valid (in Arrow, BooleanArray implements ArrayOrd), filter expression:
        // x > (y > z)
        // The x column is a Boolean-valued column. The y and z columns are numeric. True > False.
        // Suppose we had a Vortex zone whose min/max statistics for each column were:
        // x: [True, True]
        // y: [1, 2]
        // z: [0, 2]
        // The pruning predicate will convert the aforementioned expression into:
        // x_max <= (y_min > z_min)
        // If we evaluate that pruning expression on our zone we get:
        // x_max <= (y_min > z_min)
        // x_max <= (1     > 0    )
        // x_max <= True
        // True <= True
        // True
        // If a pruning predicate evaluates to true then, as stated in PruningPredicate::evaluate:
        // > a true value means the chunk can be pruned.
        // But, the following record lies within the above intervals and *passes* the filter expression! We
        // cannot prune this zone because we need this record!
        // {x: True, y: 1, z: 2}
        // x > (y > z)
        // True > (1 > 2)
        // True > False
        // True
        let expr = gt_eq(col("x"), gt(col("y"), col("z")));
        assert!(checked_pruning_expr(&expr, &available_stats).is_none());
        // TODO(DK): a sufficiently complex pruner would produce: `x_max <= (y_max > z_min)`
    }

    #[fixture]
    fn available_stats_with_nans() -> FieldPathSet {
        let float_col = FieldPath::from_name("float_col");
        let int_col = FieldPath::from_name("int_col");

        FieldPathSet::from_iter([
            // Float columns will have a NaNCount.
            float_col.clone().push(Stat::Min.name()),
            float_col.clone().push(Stat::Max.name()),
            float_col.push(Stat::NaNCount.name()),
            // int columns will not have a NanCount serialized into the layout
            int_col.clone().push(Stat::Min.name()),
            int_col.push(Stat::Max.name()),
        ])
    }

    #[rstest]
    fn pruning_checks_nans(available_stats_with_nans: FieldPathSet) {
        let expr = gt_eq(col("float_col"), lit(f32::NAN));
        let (converted, _) = checked_pruning_expr(&expr, &available_stats_with_nans).unwrap();
        assert_eq!(
            &converted,
            &and(
                and(
                    eq(col("float_col_nan_count"), lit(0u64)),
                    // NaNCount of NaN is 1
                    eq(lit(1u64), lit(0u64)),
                ),
                // This is the standard conversion of the >= operator. Comparing NAN to a max
                // stat is nonsensical, as min/max stats ignore NaNs, but this should be short-circuited
                // by the previous check for nan_count anyway.
                lt(col("float_col_max"), lit(f32::NAN)),
            )
        );

        // One half of the expression requires NAN count check, the other half does not.
        let expr = and(
            gt(col("float_col"), lit(10f32)),
            lt(col("int_col"), lit(10)),
        );

        let (converted, _) = checked_pruning_expr(&expr, &available_stats_with_nans).unwrap();

        assert_eq!(
            &converted,
            &or(
                // NaNCount check is enforced for the float column
                and(
                    and(
                        eq(col("float_col_nan_count"), lit(0u64)),
                        // NanCount of a non-NaN float literal is 0
                        eq(lit(0u64), lit(0u64)),
                    ),
                    // We want the opposite: we can prune IF either one is false.
                    lt_eq(col("float_col_max"), lit(10f32)),
                ),
                // NanCount check is skipped for the int column
                gt_eq(col("int_col_min"), lit(10)),
            )
        )
    }

    #[rstest]
    fn pruning_between(available_stats: FieldPathSet) {
        let expr = between(
            col("a"),
            lit(10),
            lit(50),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        let (converted, refs) = checked_pruning_expr(&expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([(
                FieldPath::from_name("a"),
                HashSet::from_iter([Stat::Min, Stat::Max])
            )])
        );
        assert_eq!(
            &converted,
            &or(gt(lit(10), col("a_max")), gt(col("a_min"), lit(50)))
        );
    }

    #[rstest]
    fn pruning_cast_get_item_eq(available_stats: FieldPathSet) {
        // This test verifies that cast properly forwards analysis methods to
        // enable pruning.
        let struct_dtype = DType::Struct(
            StructFields::new(
                FieldNames::from([FieldName::from("a"), FieldName::from("b")]),
                vec![
                    DType::Utf8(Nullability::Nullable),
                    DType::Utf8(Nullability::Nullable),
                ],
            ),
            Nullability::NonNullable,
        );
        let expr = eq(get_item("a", cast(root(), struct_dtype)), lit("value"));
        let (converted, refs) = checked_pruning_expr(&expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([(
                FieldPath::from_name("a"),
                HashSet::from_iter([Stat::Min, Stat::Max])
            )])
        );
        assert_eq!(
            &converted,
            &or(
                gt(col("a_min"), lit("value")),
                gt(lit("value"), col("a_max"))
            )
        );
    }

    // ===== push_not_inward tests =====

    use crate::expr::pruning::push_not_inward;

    #[rstest]
    #[case(gt(col("a"), lit(10)), lt_eq(col("a"), lit(10)))]
    #[case(gt_eq(col("a"), lit(10)), lt(col("a"), lit(10)))]
    #[case(lt(col("a"), lit(10)), gt_eq(col("a"), lit(10)))]
    #[case(lt_eq(col("a"), lit(10)), gt(col("a"), lit(10)))]
    #[case(eq(col("a"), lit(10)), not_eq(col("a"), lit(10)))]
    #[case(not_eq(col("a"), lit(10)), eq(col("a"), lit(10)))]
    fn push_not_inward_comparison(#[case] input: Expression, #[case] expected: Expression) {
        let result = push_not_inward(&input).unwrap();
        assert_eq!(result, expected);
    }

    use crate::expr::Expression;

    #[rstest]
    fn push_not_inward_and_de_morgan() {
        // NOT(a > 10 AND b < 5) = (a <= 10) OR (b >= 5)
        let input = and(gt(col("a"), lit(10)), lt(col("b"), lit(5)));
        let result = push_not_inward(&input).unwrap();
        assert_eq!(
            result,
            or(lt_eq(col("a"), lit(10)), gt_eq(col("b"), lit(5)))
        );
    }

    #[rstest]
    fn push_not_inward_or_de_morgan() {
        // NOT(a > 10 OR b < 5) = (a <= 10) AND (b >= 5)
        let input = or(gt(col("a"), lit(10)), lt(col("b"), lit(5)));
        let result = push_not_inward(&input).unwrap();
        assert_eq!(
            result,
            and(lt_eq(col("a"), lit(10)), gt_eq(col("b"), lit(5)))
        );
    }

    #[rstest]
    fn push_not_inward_nested() {
        // NOT(a > 10 AND (b < 5 OR c = 3))
        // = (a <= 10) OR NOT(b < 5 OR c = 3)
        // = (a <= 10) OR ((b >= 5) AND (c != 3))
        let input = and(
            gt(col("a"), lit(10)),
            or(lt(col("b"), lit(5)), eq(col("c"), lit(3))),
        );
        let result = push_not_inward(&input).unwrap();
        assert_eq!(
            result,
            or(
                lt_eq(col("a"), lit(10)),
                and(gt_eq(col("b"), lit(5)), not_eq(col("c"), lit(3))),
            )
        );
    }

    // ===== checked_satisfaction_expr tests =====

    use crate::expr::pruning::checked_satisfaction_expr;

    #[fixture]
    fn available_stats_with_null_count() -> FieldPathSet {
        let field_a = FieldPath::from_name("a");
        let field_b = FieldPath::from_name("b");

        FieldPathSet::from_iter([
            field_a.clone().push(Stat::Min.name()),
            field_a.clone().push(Stat::Max.name()),
            field_a.push(Stat::NullCount.name()),
            field_b.clone().push(Stat::Min.name()),
            field_b.clone().push(Stat::Max.name()),
            field_b.push(Stat::NullCount.name()),
        ])
    }

    #[rstest]
    fn satisfaction_gt_value(available_stats_with_null_count: FieldPathSet) {
        // Filter: a > 42
        // Negated: a <= 42
        // Falsification of negated: a_min > 42
        // Plus null_count check: a_null_count == 0 AND a_min > 42
        let expr = gt(col("a"), lit(42));
        let (satisfaction_expr, refs) =
            checked_satisfaction_expr(&expr, &available_stats_with_null_count).unwrap();

        assert!(refs.map().contains_key(&FieldPath::from_name("a")));
        assert!(refs.map()[&FieldPath::from_name("a")].contains(&Stat::NullCount));

        // The satisfaction expression should contain a null_count check
        let expr_str = satisfaction_expr.to_string();
        assert!(
            expr_str.contains("null_count"),
            "Expected null_count in: {expr_str}"
        );
    }

    #[rstest]
    fn satisfaction_without_null_count(available_stats: FieldPathSet) {
        // When null_count is not available, we still produce a satisfaction expr,
        // just without the null check (conservative: only works for non-nullable columns)
        let expr = gt(col("a"), lit(42));
        let result = checked_satisfaction_expr(&expr, &available_stats);
        assert!(result.is_some());
        let (satisfaction_expr, _) = result.unwrap();
        let expr_str = satisfaction_expr.to_string();
        // No null_count available, so no null check
        assert!(
            !expr_str.contains("null_count"),
            "Unexpected null_count in: {expr_str}"
        );
    }

    #[rstest]
    fn satisfaction_and_expr(available_stats_with_null_count: FieldPathSet) {
        // Filter: a > 10 AND b < 50
        // Negated: a <= 10 OR b >= 50
        // Falsification should produce something
        let expr = and(gt(col("a"), lit(10)), lt(col("b"), lit(50)));
        let result = checked_satisfaction_expr(&expr, &available_stats_with_null_count);
        assert!(result.is_some());
    }
}
