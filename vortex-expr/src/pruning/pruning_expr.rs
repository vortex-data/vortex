// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use itertools::Itertools;
use vortex_array::stats::Stat;
use vortex_dtype::{Field, FieldName, FieldPath, FieldPathSet};
use vortex_utils::aliases::hash_map::HashMap;

use super::relation::Relation;
use crate::{ExprRef, StatsCatalog, get_item, root};

pub type RequiredStats = Relation<FieldPath, Stat>;

// A catalog that return a stat column whenever it is required, tracking all accessed
// stats and returning them later.
#[derive(Default)]
struct TrackingStatsCatalog {
    usage: HashMap<(FieldPath, Stat), ExprRef>,
}

impl TrackingStatsCatalog {
    /// Consume the catalog, yielding a map of field statistics that were required
    /// for each expression.
    fn into_usages(self) -> HashMap<(FieldPath, Stat), ExprRef> {
        self.usage
    }
}

// A catalog that return a stat column if it exists in the given scope.
struct ScopeStatsCatalog<'a> {
    any_catalog: TrackingStatsCatalog,
    available_stats: &'a FieldPathSet,
}

impl StatsCatalog for ScopeStatsCatalog<'_> {
    fn stats_ref(&mut self, field_path: &FieldPath, stat: Stat) -> Option<ExprRef> {
        let stat_path = field_path.clone().push(stat.name());

        if self.available_stats.contains(&stat_path) {
            self.any_catalog.stats_ref(field_path, stat)
        } else {
            None
        }
    }
}

impl StatsCatalog for TrackingStatsCatalog {
    fn stats_ref(&mut self, field_path: &FieldPath, stat: Stat) -> Option<ExprRef> {
        let mut expr = root();
        let name = field_path_stat_field_name(field_path, stat);
        expr = get_item(name, expr);
        self.usage.insert((field_path.clone(), stat), expr.clone());
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
    expr: &ExprRef,
    available_stats: &FieldPathSet,
) -> Option<(ExprRef, RequiredStats)> {
    let mut catalog = ScopeStatsCatalog {
        any_catalog: Default::default(),
        available_stats,
    };

    let expr = expr.stat_falsification(&mut catalog)?;

    // TODO(joe): filter access by used exprs
    let mut relation: Relation<FieldPath, Stat> = Relation::new();
    for ((field_path, stat), _) in catalog.any_catalog.into_usages() {
        relation.insert(field_path, stat)
    }

    Some((expr, relation))
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use vortex_array::compute::{BetweenOptions, StrictComparison};
    use vortex_array::stats::Stat;
    use vortex_dtype::{FieldName, FieldPath, FieldPathSet};

    use crate::pruning::pruning_expr::HashMap;
    use crate::pruning::{checked_pruning_expr, field_path_stat_field_name};
    use crate::{
        HashSet, and, between, col, eq, get_item, gt, gt_eq, lit, lt, lt_eq, not_eq, or, root,
    };

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
        let not_eq_expr = gt(col(column.clone()), other_expr.clone());

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
            other_col.clone(),
        );
        assert_eq!(&converted, &(expected_expr));
    }

    #[rstest]
    pub fn pruning_lt_column(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = col(other_col.clone());
        let not_eq_expr = lt(col(column.clone()), other_expr.clone());

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
        let expr = or(lt(col("a").clone(), lit(10)), gt(col("a").clone(), lit(50)));

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
}
