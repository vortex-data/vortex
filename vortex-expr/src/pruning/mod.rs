mod field_or_identity;
mod pruning_predicate;
mod pruning_predicate_builder;
mod relation;

pub use field_or_identity::{FieldOrIdentity, stat_field_name};
pub use pruning_predicate::PruningPredicate;

#[cfg(test)]
mod tests {
    use vortex_array::aliases::hash_map::HashMap;
    use vortex_array::aliases::hash_set::HashSet;
    use vortex_array::stats::Stat;
    use vortex_dtype::FieldName;

    use super::{FieldOrIdentity, PruningPredicate, stat_field_name};
    use crate::{
        and, col, eq, get_item, get_item_scope, gt, gt_eq, lit, lt, lt_eq, not, not_eq, or, root,
    };

    #[test]
    pub fn pruning_equals() {
        let name = FieldName::from("a");
        let literal_eq = lit(42);
        let eq_expr = eq(get_item("a", root()), literal_eq.clone());
        let pp = PruningPredicate::try_new(&eq_expr).unwrap();
        assert_eq!(
            pp.required_stats.map(),
            &HashMap::from_iter([(
                FieldOrIdentity::Field(name.clone()),
                HashSet::from_iter([Stat::Min, Stat::Max])
            )])
        );
        let expected_expr = or(
            gt(
                get_item(stat_field_name(&name, Stat::Min), root()),
                literal_eq.clone(),
            ),
            gt(
                literal_eq,
                get_item_scope(stat_field_name(&name, Stat::Max)),
            ),
        );
        assert_eq!(pp.expr(), &expected_expr);
    }

    #[test]
    pub fn pruning_equals_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let eq_expr = eq(
            get_item_scope(column.clone()),
            get_item_scope(other_col.clone()),
        );

        let pp = PruningPredicate::try_new(&eq_expr).unwrap();
        assert_eq!(
            pp.required_stats.map(),
            &HashMap::from_iter([
                (
                    FieldOrIdentity::Field(column.clone()),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    FieldOrIdentity::Field(other_col.clone()),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = or(
            gt(
                get_item_scope(stat_field_name(&column, Stat::Min)),
                get_item_scope(stat_field_name(&other_col, Stat::Max)),
            ),
            gt(
                get_item_scope(stat_field_name(&other_col, Stat::Min)),
                get_item_scope(stat_field_name(&column, Stat::Max)),
            ),
        );
        assert_eq!(pp.expr(), &expected_expr);
    }

    #[test]
    pub fn pruning_not_equals_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let not_eq_expr = not_eq(
            get_item_scope(column.clone()),
            get_item_scope(other_col.clone()),
        );

        let pp = PruningPredicate::try_new(&not_eq_expr).unwrap();
        assert_eq!(
            pp.required_stats.map(),
            &HashMap::from_iter([
                (
                    FieldOrIdentity::Field(column.clone()),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    FieldOrIdentity::Field(other_col.clone()),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = and(
            eq(
                get_item_scope(stat_field_name(&column, Stat::Min)),
                get_item_scope(stat_field_name(&other_col, Stat::Max)),
            ),
            eq(
                get_item_scope(stat_field_name(&column, Stat::Max)),
                get_item_scope(stat_field_name(&other_col, Stat::Min)),
            ),
        );

        assert_eq!(pp.expr(), &expected_expr);
    }

    #[test]
    pub fn pruning_gt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = gt(get_item_scope(column.clone()), other_expr.clone());

        let pp = PruningPredicate::try_new(&not_eq_expr).unwrap();
        assert_eq!(
            pp.required_stats.map(),
            &HashMap::from_iter([
                (
                    FieldOrIdentity::Field(column.clone()),
                    HashSet::from_iter([Stat::Max])
                ),
                (
                    FieldOrIdentity::Field(other_col.clone()),
                    HashSet::from_iter([Stat::Min])
                )
            ])
        );
        let expected_expr = lt_eq(
            get_item_scope(stat_field_name(&column, Stat::Max)),
            get_item_scope(stat_field_name(&other_col, Stat::Min)),
        );
        assert_eq!(pp.expr(), &expected_expr);
    }

    #[test]
    pub fn pruning_gt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = gt(get_item_scope(column.clone()), other_col.clone());

        let pp = PruningPredicate::try_new(&not_eq_expr).unwrap();
        assert_eq!(
            pp.required_stats.map(),
            &HashMap::from_iter([(
                FieldOrIdentity::Field(column.clone()),
                HashSet::from_iter([Stat::Max])
            ),])
        );
        let expected_expr = lt_eq(
            get_item_scope(stat_field_name(&column, Stat::Max)),
            other_col.clone(),
        );
        assert_eq!(pp.expr(), &expected_expr);
    }

    #[test]
    pub fn pruning_lt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = lt(get_item_scope(column.clone()), other_expr.clone());

        let pp = PruningPredicate::try_new(&not_eq_expr).unwrap();
        assert_eq!(
            pp.required_stats.map(),
            &HashMap::from_iter([
                (
                    FieldOrIdentity::Field(column.clone()),
                    HashSet::from_iter([Stat::Min])
                ),
                (
                    FieldOrIdentity::Field(other_col.clone()),
                    HashSet::from_iter([Stat::Max])
                )
            ])
        );
        let expected_expr = gt_eq(
            get_item_scope(stat_field_name(&column, Stat::Min)),
            get_item_scope(stat_field_name(&other_col, Stat::Max)),
        );
        assert_eq!(pp.expr(), &expected_expr);
    }

    #[test]
    pub fn pruning_lt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = lt(get_item_scope(column.clone()), other_col.clone());

        let pp = PruningPredicate::try_new(&not_eq_expr).unwrap();
        assert_eq!(
            pp.required_stats.map(),
            &HashMap::from_iter([(
                FieldOrIdentity::Field(column.clone()),
                HashSet::from_iter([Stat::Min])
            )])
        );
        let expected_expr = gt_eq(
            get_item_scope(stat_field_name(&column, Stat::Min)),
            other_col.clone(),
        );
        assert_eq!(pp.expr(), &expected_expr);
    }

    #[test]
    fn unprojectable_expr() {
        let or_expr = not(lt(get_item_scope("a"), get_item_scope("b")));
        assert!(PruningPredicate::try_new(&or_expr).is_none());
    }

    #[test]
    fn or_required_stats_from_both_arms() {
        let item = get_item_scope(FieldName::from("a"));
        let expr = or(lt(item.clone(), lit(10)), gt(item, lit(50)));

        let expected = HashMap::from([(
            FieldOrIdentity::from("a"),
            HashSet::from([Stat::Min, Stat::Max]),
        )]);

        assert_eq!(
            PruningPredicate::try_new(&expr).unwrap().required_stats(),
            &expected
        );
    }

    #[test]
    fn and_required_stats_from_both_arms() {
        let item = get_item_scope(FieldName::from("a"));
        let expr = and(gt(item.clone(), lit(50)), lt(item, lit(10)));

        let expected = HashMap::from([(
            FieldOrIdentity::from("a"),
            HashSet::from([Stat::Min, Stat::Max]),
        )]);

        assert_eq!(
            PruningPredicate::try_new(&expr).unwrap().required_stats(),
            &expected
        );
    }

    #[test]
    fn pruning_identity() {
        let expr = root();
        let expr = or(lt(expr.clone(), lit(10)), gt(expr.clone(), lit(50)));

        let expected = HashMap::from([(
            FieldOrIdentity::Identity,
            HashSet::from([Stat::Min, Stat::Max]),
        )]);

        let predicate = PruningPredicate::try_new(&expr).unwrap();
        assert_eq!(predicate.required_stats(), &expected, "{predicate:#?}");

        let expected_expr = and(
            gt_eq(get_item_scope(FieldName::from("min")), lit(10)),
            lt_eq(get_item_scope(FieldName::from("max")), lit(50)),
        );
        assert_eq!(predicate.expr(), &expected_expr)
    }

    #[test]
    pub fn pruning_and_or_operators() {
        // Test case: a > 10 AND a < 50
        let column = FieldName::from("a");
        let and_expr = and(
            gt(get_item_scope(column.clone()), lit(10)),
            lt(get_item_scope(column), lit(50)),
        );
        let pruned = PruningPredicate::try_new(&and_expr).unwrap();

        // Expected: a_max <= 10 OR a_min >= 50
        assert_eq!(
            pruned.expr(),
            &or(
                lt_eq(get_item_scope(FieldName::from("a_max")), lit(10)),
                gt_eq(get_item_scope(FieldName::from("a_min")), lit(50))
            ),
            "{:#?}",
            pruned.expr()
        );
    }

    #[test]
    fn test_gt_eq_with_booleans() {
        // Consider this unusual, but valid (in Arrow, BooleanArray implements ArrayOrd), filter expression:
        //
        // x > (y > z)
        //
        // The x column is a Boolean-valued column. The y and z columns are numeric. True > False.
        // Suppose we had a Vortex zone whose min/max statistics for each column were:
        //
        // x: [True, True]
        // y: [1, 2]
        // z: [0, 2]
        //
        // The pruning predicate will convert the aforementioned expression into:
        //
        // x_max <= (y_min > z_min)
        //
        // If we evaluate that pruning expression on our zone we get:
        //
        // x_max <= (y_min > z_min)
        // x_max <= (1     > 0    )
        // x_max <= True
        // True <= True
        // True
        //
        // If a pruning predicate evaluates to true then, as stated in PruningPredicate::evaluate:
        //
        // > a true value means the chunk can be pruned.
        //
        // But, the following record lies within the above intervals and *passes* the filter expression! We
        // cannot prune this zone because we need this record!
        //
        // {x: True, y: 1, z: 2}
        //
        // x > (y > z)
        // True > (1 > 2)
        // True > False
        // True
        let expr = gt_eq(col("x"), gt(col("y"), col("z")));
        assert!(PruningPredicate::try_new(&expr).is_none());
        // TODO(DK): a sufficiently complex pruner would produce: `x_max <= (y_max > z_min)`
    }
}
