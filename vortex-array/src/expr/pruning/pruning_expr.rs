// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use super::relation::Relation;
use crate::dtype::DType;
use crate::dtype::Field;
use crate::dtype::FieldName;
use crate::dtype::FieldPath;
use crate::dtype::FieldPathSet;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::analysis::referenced_field_paths;
use crate::expr::get_item;
use crate::expr::is_root;
use crate::expr::root;
use crate::expr::stats::Stat;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::stats::bind::StatBinder;
use crate::stats::bind::bind_stats;

pub type RequiredStats = Relation<FieldPath, Stat>;

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

/// Build a pruning expression using session-registered stats rewrite rules.
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
/// The returned expression is lowered to stats-table field references. Stats not present in
/// `available_stats` are replaced with typed null literals, preserving three-valued pruning
/// semantics without requiring callers to materialize unavailable stats.
pub fn checked_pruning_expr(
    expr: &Expression,
    scope: &DType,
    available_stats: &FieldPathSet,
    session: &VortexSession,
) -> VortexResult<Option<(Expression, RequiredStats)>> {
    let Some(predicate) = expr.falsify(scope, session)? else {
        return Ok(None);
    };

    let mut binder = RequiredStatsBinder {
        scope,
        available_stats,
        required_stats: Relation::new(),
        bound_stats: Vec::new(),
    };
    let lowered = bind_stats(predicate, &mut binder)?;
    let required_stats = filter_required_stats(&lowered, binder.required_stats);
    // If no stats-table fields remain, only a constant `true` proof can prune.
    // `false`, `null`, and non-constant expressions cannot justify building a
    // stats-table pruning expression.
    if required_stats.map().is_empty() && !matches!(bool_literal(&lowered), Some(Some(true))) {
        return Ok(None);
    }

    Ok(Some((lowered, required_stats)))
}

struct RequiredStatsBinder<'a> {
    scope: &'a DType,
    available_stats: &'a FieldPathSet,
    required_stats: RequiredStats,
    bound_stats: Vec<(FieldName, DType)>,
}

impl StatBinder for RequiredStatsBinder<'_> {
    fn scope(&self) -> &DType {
        self.scope
    }

    fn bound_scope(&self) -> DType {
        DType::Struct(
            StructFields::from_iter(self.bound_stats.iter().cloned()),
            Nullability::NonNullable,
        )
    }

    fn bind_stat(
        &mut self,
        input: &Expression,
        stat: Stat,
        stat_dtype: &DType,
    ) -> VortexResult<Option<Expression>> {
        let field_path = match direct_stat_field_path(input) {
            Some(field_path) => field_path,
            None => {
                let field_paths = referenced_field_paths(input, self.scope)?;
                let Some(field_path) = field_paths.iter().exactly_one().ok() else {
                    return Ok(None);
                };
                field_path.clone()
            }
        };
        let stat_path = field_path.clone().push(stat.name());
        if !self.available_stats.contains(&stat_path) {
            return Ok(None);
        }

        let stat_field_name = field_path_stat_field_name(&field_path, stat);
        if self
            .bound_stats
            .iter()
            .all(|(field_name, _)| field_name != stat_field_name)
        {
            self.bound_stats
                .push((stat_field_name.clone(), stat_dtype.clone()));
        }

        self.required_stats.insert(field_path, stat);
        Ok(Some(get_item(stat_field_name, root())))
    }
}

fn direct_stat_field_path(expr: &Expression) -> Option<FieldPath> {
    if is_root(expr) {
        return Some(FieldPath::root());
    }

    if expr.is::<Cast>() {
        return direct_stat_field_path(expr.child(0));
    }

    let field_name = expr.as_opt::<GetItem>()?;
    direct_stat_field_path(expr.child(0)).map(|path| path.push(field_name.clone()))
}

fn filter_required_stats(expr: &Expression, required_stats: RequiredStats) -> RequiredStats {
    let referenced_names = referenced_stat_field_names(expr);
    let mut filtered = Relation::new();
    for (field_path, stats) in required_stats {
        for stat in stats {
            if referenced_names.contains(&field_path_stat_field_name(&field_path, stat)) {
                filtered.insert(field_path.clone(), stat);
            }
        }
    }
    filtered
}

fn referenced_stat_field_names(expr: &Expression) -> HashSet<FieldName> {
    let mut refs = HashSet::new();
    collect_referenced_stat_field_names(expr, &mut refs);
    refs
}

fn collect_referenced_stat_field_names(expr: &Expression, refs: &mut HashSet<FieldName>) {
    if let Some(field_name) = expr.as_opt::<GetItem>()
        && is_root(expr.child(0))
    {
        refs.insert(field_name.clone());
        return;
    }

    for child in expr.children().iter() {
        collect_referenced_stat_field_names(child, refs);
    }
}

fn bool_literal(expr: &Expression) -> Option<Option<bool>> {
    expr.as_opt::<Literal>()?
        .as_bool_opt()
        .map(|value| value.value())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::fixture;
    use rstest::rstest;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_map::HashMap;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::RequiredStats;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::FieldPath;
    use crate::dtype::FieldPathSet;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::cast;
    use crate::expr::col;
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
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;
    use crate::stats::session::StatsSession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

    fn test_scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("x", DType::Bool(Nullability::NonNullable)),
                ("y", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("z", DType::Primitive(PType::I32, Nullability::NonNullable)),
                (
                    "float_col",
                    DType::Primitive(PType::F32, Nullability::NonNullable),
                ),
                (
                    "int_col",
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                ),
            ]),
            Nullability::NonNullable,
        )
    }

    fn checked(
        expr: &Expression,
        available_stats: &FieldPathSet,
    ) -> Option<(Expression, RequiredStats)> {
        checked_pruning_expr(expr, &test_scope(), available_stats, &SESSION).unwrap()
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
        let (converted, _refs) = checked(&eq_expr, &available_stats).unwrap();
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

        let (converted, refs) = checked(&eq_expr, &available_stats).unwrap();
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

        let (converted, refs) = checked(&not_eq_expr, &available_stats).unwrap();
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

        let (converted, refs) = checked(&not_eq_expr, &available_stats).unwrap();
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

        let (converted, refs) = checked(&not_eq_expr, &available_stats).unwrap();
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

        let (converted, refs) = checked(&not_eq_expr, &available_stats).unwrap();
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

        let (converted, refs) = checked(&expr, &available_stats).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([(FieldPath::from_name("a"), HashSet::from_iter([Stat::Min]))])
        );
        assert_eq!(&converted, &gt_eq(col("a_min"), lit(42)));
    }

    #[rstest]
    fn pruning_identity(available_stats: FieldPathSet) {
        let expr = or(lt(col("a"), lit(10)), gt(col("a"), lit(50)));

        let (predicate, _) = checked(&expr, &available_stats).unwrap();

        let expected_expr = and(gt_eq(col("a_min"), lit(10)), lt_eq(col("a_max"), lit(50)));
        assert_eq!(&predicate.to_string(), &expected_expr.to_string());
    }
    #[rstest]
    pub fn pruning_and_or_operators(available_stats: FieldPathSet) {
        // Test case: a > 10 AND a < 50
        let column = FieldName::from("a");
        let and_expr = and(gt(col(column.clone()), lit(10)), lt(col(column), lit(50)));
        let (predicate, _) = checked(&and_expr, &available_stats).unwrap();

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
        assert!(checked(&expr, &available_stats).is_none());
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
        assert!(checked(&expr, &available_stats_with_nans).is_none());

        // One half of the expression requires an all-non-NaN proof, the other half does not.
        let expr = and(
            gt(col("float_col"), lit(10f32)),
            lt(col("int_col"), lit(10)),
        );

        let (converted, refs) = checked(&expr, &available_stats_with_nans).unwrap();
        assert_eq!(
            refs.map(),
            &HashMap::from_iter([
                (
                    FieldPath::from_name("float_col"),
                    HashSet::from_iter([Stat::Max])
                ),
                (
                    FieldPath::from_name("int_col"),
                    HashSet::from_iter([Stat::Min])
                )
            ])
        );
        assert_eq!(
            &converted,
            &or(
                and(
                    lit(Scalar::null(DType::Bool(Nullability::Nullable))),
                    lt_eq(col("float_col_max"), lit(10f32)),
                ),
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
        let (converted, refs) = checked(&expr, &available_stats).unwrap();
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
        let (converted, refs) = checked(&expr, &available_stats).unwrap();
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
}
