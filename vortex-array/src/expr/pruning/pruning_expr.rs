// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;
use std::iter;

use itertools::Itertools;
use vortex_utils::aliases::hash_map::HashMap;

use super::relation::Relation;
use crate::dtype::DType;
use crate::dtype::Field;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::FieldPath;
use crate::dtype::FieldPathSet;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::StructFields;
use crate::expr::BoundExpr;
use crate::expr::StatsCatalog;
use crate::expr::get_item;
use crate::expr::root;
use crate::expr::stats::Stat;

pub type RequiredStats = Relation<FieldPath, Stat>;

// A catalog that return a stat column whenever it is required, tracking all accessed
// stats and returning them later.
pub(crate) struct TrackingStatsCatalog {
    usage: RefCell<HashMap<(FieldPath, Stat), BoundExpr>>,
    stats_scope: DType,
}

impl TrackingStatsCatalog {
    pub(crate) fn new(stats_scope: DType) -> Self {
        Self {
            usage: RefCell::default(),
            stats_scope,
        }
    }

    /// Consume the catalog, yielding a map of field statistics that were required
    /// for each expression.
    fn into_usages(self) -> HashMap<(FieldPath, Stat), BoundExpr> {
        self.usage.into_inner()
    }
}

// A catalog that return a stat column if it exists in the given scope.
struct ScopeStatsCatalog<'a> {
    inner: TrackingStatsCatalog,
    available_stats: &'a FieldPathSet,
}

impl StatsCatalog for ScopeStatsCatalog<'_> {
    fn stats_ref(&self, field_path: &FieldPath, stat: Stat) -> Option<BoundExpr> {
        let stat_path = field_path.clone().push(stat.name());

        if self.available_stats.contains(&stat_path) {
            self.inner.stats_ref(field_path, stat)
        } else {
            None
        }
    }
}

impl StatsCatalog for TrackingStatsCatalog {
    fn stats_ref(&self, field_path: &FieldPath, stat: Stat) -> Option<BoundExpr> {
        let mut expr = root(self.stats_scope.clone());
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

fn stats_scope(scope: &DType, available_stats: &FieldPathSet) -> Option<DType> {
    let mut fields = Vec::new();

    for stat_path in available_stats.iter() {
        let (stat_field, field_path_parts) = stat_path.parts().split_last()?;
        let stat_name = match stat_field {
            Field::Name(name) => name,
            Field::ElementType => return None,
        };
        let stat = Stat::all().find(|stat| stat.name() == stat_name.as_ref())?;
        let field_path = FieldPath::from(field_path_parts.to_vec());
        let field_dtype = field_path.resolve(scope.clone())?;
        let stat_dtype = stat.dtype(&field_dtype)?;

        fields.push((field_path_stat_field_name(&field_path, stat), stat_dtype));
    }

    // `available_stats` iterates in hash order; sort so the synthesized scope (and therefore
    // Root equality, hashing, and serialized bytes of pruning expressions) is deterministic.
    fields.sort_by(|(a, _), (b, _)| a.cmp(b));
    let (names, dtypes): (Vec<_>, Vec<_>) = fields.into_iter().unzip();

    Some(DType::Struct(
        StructFields::new(FieldNames::from(names), dtypes),
        NonNullable,
    ))
}

/// Build a pruning expr mask, using an existing set of stats.
/// The available stats are provided as a set of [`FieldPath`].
///
/// A pruning expression is one that returns `true` for all positions where the original expression
/// cannot hold, and false if it cannot be determined from stats alone whether the positions can
/// be pruned.
///
/// Some rewrites, such as `is_not_null(...)`, emit
/// [`row_count`][crate::scalar_fn::internal::row_count] placeholders. The evaluation layer must
/// replace those placeholders with the row count for its current scope before
/// executing the returned expression.
///
/// If the falsification logic attempts to access an unknown stat,
/// this function will return `None`.
pub fn checked_pruning_expr(
    expr: &BoundExpr,
    scope: &DType,
    available_stats: &FieldPathSet,
) -> Option<(BoundExpr, RequiredStats)> {
    let catalog = ScopeStatsCatalog {
        inner: TrackingStatsCatalog::new(stats_scope(scope, available_stats)?),
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
    use crate::expr::BoundExpr;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::cast;
    use crate::expr::col as expr_col;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::expr::not_eq;
    use crate::expr::or;
    use crate::expr::pruning::checked_pruning_expr;
    use crate::expr::pruning::field_path_stat_field_name;
    use crate::expr::root as expr_root;
    use crate::expr::stats::Stat;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;

    fn numeric_scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    DType::Primitive(crate::dtype::PType::I32, Nullability::NonNullable),
                ),
                (
                    "b",
                    DType::Primitive(crate::dtype::PType::I32, Nullability::NonNullable),
                ),
                ("x", DType::Bool(Nullability::NonNullable)),
                (
                    "y",
                    DType::Primitive(crate::dtype::PType::I32, Nullability::NonNullable),
                ),
                (
                    "z",
                    DType::Primitive(crate::dtype::PType::I32, Nullability::NonNullable),
                ),
                (
                    "float_col",
                    DType::Primitive(crate::dtype::PType::F32, Nullability::NonNullable),
                ),
                (
                    "int_col",
                    DType::Primitive(crate::dtype::PType::I32, Nullability::NonNullable),
                ),
            ]),
            Nullability::NonNullable,
        )
    }

    fn root() -> BoundExpr {
        expr_root(numeric_scope())
    }

    fn col(field: impl Into<FieldName>) -> BoundExpr {
        expr_col(field, &numeric_scope())
    }

    fn stat_root(scope: &DType, available_stats: &FieldPathSet) -> BoundExpr {
        expr_root(super::stats_scope(scope, available_stats).unwrap())
    }

    fn stat_col(
        field: impl Into<FieldName>,
        scope: &DType,
        available_stats: &FieldPathSet,
    ) -> BoundExpr {
        expr_col(field, &super::stats_scope(scope, available_stats).unwrap())
    }

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
        let (converted, _refs) =
            checked_pruning_expr(&eq_expr, &numeric_scope(), &available_stats).unwrap();
        let expected_expr = or(
            gt(
                get_item(
                    field_path_stat_field_name(&FieldPath::from_name(name.clone()), Stat::Min),
                    stat_root(&numeric_scope(), &available_stats),
                ),
                literal_eq.clone(),
            ),
            gt(
                literal_eq,
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(name), Stat::Max),
                    &numeric_scope(),
                    &available_stats,
                ),
            ),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_equals_column(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let eq_expr = eq(col(column.clone()), col(other_col.clone()));

        let (converted, refs) =
            checked_pruning_expr(&eq_expr, &numeric_scope(), &available_stats).unwrap();
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
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(column.clone()), Stat::Min),
                    &numeric_scope(),
                    &available_stats,
                ),
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(other_col.clone()), Stat::Max),
                    &numeric_scope(),
                    &available_stats,
                ),
            ),
            gt(
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(other_col), Stat::Min),
                    &numeric_scope(),
                    &available_stats,
                ),
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(column), Stat::Max),
                    &numeric_scope(),
                    &available_stats,
                ),
            ),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_not_equals_column(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let not_eq_expr = not_eq(col(column.clone()), col(other_col.clone()));

        let (converted, refs) =
            checked_pruning_expr(&not_eq_expr, &numeric_scope(), &available_stats).unwrap();
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
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(column.clone()), Stat::Min),
                    &numeric_scope(),
                    &available_stats,
                ),
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(other_col.clone()), Stat::Max),
                    &numeric_scope(),
                    &available_stats,
                ),
            ),
            eq(
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(column), Stat::Max),
                    &numeric_scope(),
                    &available_stats,
                ),
                stat_col(
                    field_path_stat_field_name(&FieldPath::from_name(other_col), Stat::Min),
                    &numeric_scope(),
                    &available_stats,
                ),
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

        let (converted, refs) =
            checked_pruning_expr(&not_eq_expr, &numeric_scope(), &available_stats).unwrap();
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
            stat_col(
                field_path_stat_field_name(&FieldPath::from_name(column), Stat::Max),
                &numeric_scope(),
                &available_stats,
            ),
            stat_col(
                field_path_stat_field_name(&FieldPath::from_name(other_col), Stat::Min),
                &numeric_scope(),
                &available_stats,
            ),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_gt_value(available_stats: FieldPathSet) {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = gt(col(column.clone()), other_col.clone());

        let (converted, refs) =
            checked_pruning_expr(&not_eq_expr, &numeric_scope(), &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([(
                FieldPath::from_name(column.clone()),
                HashSet::from_iter([Stat::Max])
            ),])
        );
        let expected_expr = lt_eq(
            stat_col(
                field_path_stat_field_name(&FieldPath::from_name(column), Stat::Max),
                &numeric_scope(),
                &available_stats,
            ),
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

        let (converted, refs) =
            checked_pruning_expr(&not_eq_expr, &numeric_scope(), &available_stats).unwrap();
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
            stat_col(
                field_path_stat_field_name(&FieldPath::from_name(column), Stat::Min),
                &numeric_scope(),
                &available_stats,
            ),
            stat_col(
                field_path_stat_field_name(&FieldPath::from_name(other_col), Stat::Max),
                &numeric_scope(),
                &available_stats,
            ),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[rstest]
    pub fn pruning_lt_value(available_stats: FieldPathSet) {
        // expression   => a < 42
        // pruning expr => a.min >= 42
        let expr = lt(col("a"), lit(42));

        let (converted, refs) =
            checked_pruning_expr(&expr, &numeric_scope(), &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([(FieldPath::from_name("a"), HashSet::from_iter([Stat::Min]))])
        );
        assert_eq!(
            &converted,
            &gt_eq(
                stat_col("a_min", &numeric_scope(), &available_stats),
                lit(42)
            )
        );
    }

    #[rstest]
    fn pruning_identity(available_stats: FieldPathSet) {
        let expr = or(lt(col("a"), lit(10)), gt(col("a"), lit(50)));

        let (predicate, _) =
            checked_pruning_expr(&expr, &numeric_scope(), &available_stats).unwrap();

        let expected_expr = and(
            gt_eq(
                stat_col("a_min", &numeric_scope(), &available_stats),
                lit(10),
            ),
            lt_eq(
                stat_col("a_max", &numeric_scope(), &available_stats),
                lit(50),
            ),
        );
        assert_eq!(&predicate.to_string(), &expected_expr.to_string());
    }
    #[rstest]
    pub fn pruning_and_or_operators(available_stats: FieldPathSet) {
        // Test case: a > 10 AND a < 50
        let column = FieldName::from("a");
        let and_expr = and(gt(col(column.clone()), lit(10)), lt(col(column), lit(50)));
        let (predicate, _) =
            checked_pruning_expr(&and_expr, &numeric_scope(), &available_stats).unwrap();

        // Expected: a_max <= 10 OR a_min >= 50
        assert_eq!(
            &predicate,
            &or(
                lt_eq(
                    stat_col(FieldName::from("a_max"), &numeric_scope(), &available_stats),
                    lit(10),
                ),
                gt_eq(
                    stat_col(FieldName::from("a_min"), &numeric_scope(), &available_stats),
                    lit(50),
                ),
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
        assert!(checked_pruning_expr(&expr, &numeric_scope(), &available_stats).is_none());
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
        let (converted, _) =
            checked_pruning_expr(&expr, &numeric_scope(), &available_stats_with_nans).unwrap();
        assert_eq!(
            &converted,
            &and(
                and(
                    eq(
                        stat_col(
                            "float_col_nan_count",
                            &numeric_scope(),
                            &available_stats_with_nans
                        ),
                        lit(0u64)
                    ),
                    // NaNCount of NaN is 1
                    eq(lit(1u64), lit(0u64)),
                ),
                // This is the standard conversion of the >= operator. Comparing NAN to a max
                // stat is nonsensical, as min/max stats ignore NaNs, but this should be short-circuited
                // by the previous check for nan_count anyway.
                lt(
                    stat_col(
                        "float_col_max",
                        &numeric_scope(),
                        &available_stats_with_nans
                    ),
                    lit(f32::NAN),
                ),
            )
        );

        // One half of the expression requires NAN count check, the other half does not.
        let expr = and(
            gt(col("float_col"), lit(10f32)),
            lt(col("int_col"), lit(10)),
        );

        let (converted, _) =
            checked_pruning_expr(&expr, &numeric_scope(), &available_stats_with_nans).unwrap();

        assert_eq!(
            &converted,
            &or(
                // NaNCount check is enforced for the float column
                and(
                    and(
                        eq(
                            stat_col(
                                "float_col_nan_count",
                                &numeric_scope(),
                                &available_stats_with_nans
                            ),
                            lit(0u64)
                        ),
                        // NanCount of a non-NaN float literal is 0
                        eq(lit(0u64), lit(0u64)),
                    ),
                    // We want the opposite: we can prune IF either one is false.
                    lt_eq(
                        stat_col(
                            "float_col_max",
                            &numeric_scope(),
                            &available_stats_with_nans
                        ),
                        lit(10f32),
                    ),
                ),
                // NanCount check is skipped for the int column
                gt_eq(
                    stat_col("int_col_min", &numeric_scope(), &available_stats_with_nans),
                    lit(10),
                ),
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
        let (converted, refs) =
            checked_pruning_expr(&expr, &numeric_scope(), &available_stats).unwrap();
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
                gt(
                    lit(10),
                    stat_col("a_max", &numeric_scope(), &available_stats)
                ),
                gt(
                    stat_col("a_min", &numeric_scope(), &available_stats),
                    lit(50)
                )
            )
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
        let expr = eq(
            get_item(
                "a",
                cast(expr_root(struct_dtype.clone()), struct_dtype.clone()),
            ),
            lit("value"),
        );
        let (converted, refs) =
            checked_pruning_expr(&expr, &struct_dtype, &available_stats).unwrap();
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
                gt(
                    stat_col("a_min", &struct_dtype, &available_stats),
                    lit("value")
                ),
                gt(
                    lit("value"),
                    stat_col("a_max", &struct_dtype, &available_stats)
                )
            )
        );
    }
}
