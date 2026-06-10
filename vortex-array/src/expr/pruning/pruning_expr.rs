// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::relation::Relation;
use crate::aggregate_fn::fns::all_nan::AllNan;
use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::dtype::DType;
use crate::dtype::Field;
use crate::dtype::FieldName;
use crate::dtype::FieldPath;
use crate::dtype::FieldPathSet;
use crate::expr::Expression;
use crate::expr::analysis::referenced_field_paths;
use crate::expr::eq;
use crate::expr::get_item;
use crate::expr::lit;
use crate::expr::root;
use crate::expr::stats::Stat;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::stat::StatFn;
use crate::scalar_fn::internal::row_count::RowCount;

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
/// The returned expression is lowered to the same stats-table field references as
/// [`checked_pruning_expr`]. If a rewrite asks for a stat that is not present in
/// `available_stats`, this returns `Ok(None)`.
pub fn checked_pruning_expr_with_session(
    expr: &Expression,
    scope: &DType,
    available_stats: &FieldPathSet,
    session: &VortexSession,
) -> VortexResult<Option<(Expression, RequiredStats)>> {
    let Some(predicate) = expr.falsify(scope, session)? else {
        return Ok(None);
    };

    lower_stat_fns(predicate, scope, available_stats)
}

fn lower_stat_fns(
    predicate: Expression,
    scope: &DType,
    available_stats: &FieldPathSet,
) -> VortexResult<Option<(Expression, RequiredStats)>> {
    let mut required_stats = Relation::new();
    let mut missing_stat = false;
    let lowered = predicate
        .transform_down(|expr| {
            if !expr.is::<StatFn>() {
                return Ok(Transformed::no(expr));
            }

            if let Some(lowered) =
                lower_stat_fn(&expr, scope, available_stats, &mut required_stats)?
            {
                return Ok(Transformed::yes(lowered));
            }

            missing_stat = true;
            let dtype = expr.return_dtype(scope)?;
            Ok(Transformed::yes(null_expr(dtype)))
        })?
        .into_inner();

    if missing_stat {
        return Ok(None);
    }

    Ok(Some((lowered, required_stats)))
}

fn lower_stat_fn(
    expr: &Expression,
    scope: &DType,
    available_stats: &FieldPathSet,
    required_stats: &mut RequiredStats,
) -> VortexResult<Option<Expression>> {
    let options = expr.as_::<StatFn>();
    let aggregate_fn = options.aggregate_fn();
    let input = expr.child(0);
    let input_dtype = input.return_dtype(scope)?;

    if aggregate_fn.is::<AllNan>() {
        if !has_nans(&input_dtype) {
            return Ok(Some(lit(false)));
        }
        return lower_stat_ref(
            input,
            Stat::NaNCount,
            scope,
            available_stats,
            required_stats,
        )
        .map(|stat| stat.map(|stat| eq(stat, row_count_expr())));
    }

    if aggregate_fn.is::<AllNonNan>() {
        if !has_nans(&input_dtype) {
            return Ok(Some(lit(true)));
        }
        return lower_stat_ref(
            input,
            Stat::NaNCount,
            scope,
            available_stats,
            required_stats,
        )
        .map(|stat| stat.map(|stat| eq(stat, lit(0u64))));
    }

    if aggregate_fn.is::<NanCount>() && !has_nans(&input_dtype) {
        return Ok(Some(lit(0u64)));
    }

    if aggregate_fn.is::<AllNull>() {
        return lower_stat_ref(
            input,
            Stat::NullCount,
            scope,
            available_stats,
            required_stats,
        )
        .map(|stat| stat.map(|stat| eq(stat, row_count_expr())));
    }

    if aggregate_fn.is::<AllNonNull>() {
        return lower_stat_ref(
            input,
            Stat::NullCount,
            scope,
            available_stats,
            required_stats,
        )
        .map(|stat| stat.map(|stat| eq(stat, lit(0u64))));
    }

    let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
        return Ok(None);
    };

    lower_stat_ref(input, stat, scope, available_stats, required_stats)
}

fn lower_stat_ref(
    input: &Expression,
    stat: Stat,
    scope: &DType,
    available_stats: &FieldPathSet,
    required_stats: &mut RequiredStats,
) -> VortexResult<Option<Expression>> {
    let Some(field_path) = stat_field_path(input, scope)? else {
        return Ok(None);
    };
    let stat_path = field_path.clone().push(stat.name());
    if !available_stats.contains(&stat_path) {
        return Ok(None);
    }

    required_stats.insert(field_path.clone(), stat);
    Ok(Some(get_item(
        field_path_stat_field_name(&field_path, stat),
        root(),
    )))
}

fn stat_field_path(input: &Expression, scope: &DType) -> VortexResult<Option<FieldPath>> {
    // Preserve the legacy top-level GetItem pruning behavior while moving the rewrite itself
    // out of ScalarFnVTable.
    if let Some(field_name) = input.as_opt::<GetItem>() {
        return Ok(Some(FieldPath::from_name(field_name.clone())));
    }

    let field_paths = referenced_field_paths(input, scope)?;
    Ok(field_paths.iter().exactly_one().ok().cloned())
}

fn row_count_expr() -> Expression {
    RowCount.new_expr(EmptyOptions, [])
}

fn null_expr(dtype: DType) -> Expression {
    lit(Scalar::null(dtype.as_nullable()))
}

fn has_nans(dtype: &DType) -> bool {
    matches!(dtype, DType::Primitive(ptype, _) if ptype.is_float())
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
    use crate::expr::pruning::checked_pruning_expr_with_session;
    use crate::expr::pruning::field_path_stat_field_name;
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;
    use crate::stats::session::StatsSession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

    fn scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::Nullable)),
                ("b", DType::Primitive(PType::I32, Nullability::Nullable)),
                ("x", DType::Bool(Nullability::Nullable)),
                ("y", DType::Primitive(PType::I32, Nullability::Nullable)),
                ("z", DType::Primitive(PType::I32, Nullability::Nullable)),
                (
                    "float_col",
                    DType::Primitive(PType::F32, Nullability::Nullable),
                ),
                (
                    "int_col",
                    DType::Primitive(PType::I32, Nullability::Nullable),
                ),
            ]),
            Nullability::NonNullable,
        )
    }

    fn checked_pruning_expr(
        expr: &Expression,
        available_stats: &FieldPathSet,
    ) -> Option<(Expression, RequiredStats)> {
        checked_pruning_expr_with_session(expr, &scope(), available_stats, &SESSION).unwrap()
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
                    // A NaN literal is never all-non-NaN.
                    lit(false),
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
                    eq(col("float_col_nan_count"), lit(0u64)),
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
}
