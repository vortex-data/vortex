use std::iter;

use itertools::Itertools;
use vortex_array::stats::Stat;
use vortex_dtype::{Field, FieldPath};

use crate::pruning::StatsCatalog;
use crate::{ExprRef, Identifier, get_item, var};

struct FileStatsCatalog;

impl StatsCatalog for FileStatsCatalog {
    fn stats_ref(&self, id: &Identifier, field: &FieldPath, stat: Stat) -> Option<ExprRef> {
        let mut expr = var(id.clone());
        let name = field
            .path()
            .iter()
            .map(|f| match f {
                Field::Name(n) => n.as_ref(),
                Field::ElementType => todo!("element type not currently handled"),
            })
            .chain(iter::once(stat.name()))
            .join("_");
        expr = get_item(name, expr);
        Some(expr)
    }
}

pub fn pruning_expr(expr: &ExprRef) -> Option<ExprRef> {
    let catalog = FileStatsCatalog;
    expr.prune_expr(&catalog)
}

#[cfg(test)]
mod tests {
    use vortex_array::stats::Stat;
    use vortex_dtype::FieldName;

    use crate::pruning::stat_field_name;
    use crate::pruning::v2::pruning_expr;
    use crate::{and, eq, get_item, get_item_scope, gt, gt_eq, lit, lt, lt_eq, not_eq, or, root};

    #[test]
    pub fn pruning_equals() {
        let name = FieldName::from("a");
        let literal_eq = lit(42);
        let eq_expr = eq(get_item("a", root()), literal_eq.clone());
        let converted = pruning_expr(&eq_expr);
        // assert_eq!(
        //     refs.map(),
        //     &HashMap::from_iter([(
        //         FieldOrIdentity::Field(name.clone()),
        //         HashSet::from_iter([Stat::Min, Stat::Max])
        //     )])
        // );
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
        println!("{}", converted.clone().unwrap());
        println!("{}", expected_expr);
        assert_eq!(&converted, &Some(expected_expr));
    }

    #[test]
    pub fn pruning_equals_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let eq_expr = eq(
            get_item_scope(column.clone()),
            get_item_scope(other_col.clone()),
        );

        let converted = pruning_expr(&eq_expr);
        // assert_eq!(
        //     refs.map(),
        //     &HashMap::from_iter([
        //         (
        //             FieldOrIdentity::Field(column.clone()),
        //             HashSet::from_iter([Stat::Min, Stat::Max])
        //         ),
        //         (
        //             FieldOrIdentity::Field(other_col.clone()),
        //             HashSet::from_iter([Stat::Max, Stat::Min])
        //         )
        //     ])
        // );
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
        assert_eq!(&converted, &Some(expected_expr));
    }

    #[test]
    pub fn pruning_not_equals_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let not_eq_expr = not_eq(
            get_item_scope(column.clone()),
            get_item_scope(other_col.clone()),
        );

        let converted = pruning_expr(&not_eq_expr);
        // assert_eq!(
        //     refs.map(),
        //     &HashMap::from_iter([
        //         (
        //             FieldOrIdentity::Field(column.clone()),
        //             HashSet::from_iter([Stat::Min, Stat::Max])
        //         ),
        //         (
        //             FieldOrIdentity::Field(other_col.clone()),
        //             HashSet::from_iter([Stat::Max, Stat::Min])
        //         )
        //     ])
        // );
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

        assert_eq!(&converted, &Some(expected_expr));
    }

    #[test]
    pub fn pruning_gt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = gt(get_item_scope(column.clone()), other_expr.clone());

        let converted = pruning_expr(&not_eq_expr);
        // assert_eq!(
        //     refs.map(),
        //     &HashMap::from_iter([
        //         (
        //             FieldOrIdentity::Field(column.clone()),
        //             HashSet::from_iter([Stat::Max])
        //         ),
        //         (
        //             FieldOrIdentity::Field(other_col.clone()),
        //             HashSet::from_iter([Stat::Min])
        //         )
        //     ])
        // );
        let expected_expr = lt_eq(
            get_item_scope(stat_field_name(&column, Stat::Max)),
            get_item_scope(stat_field_name(&other_col, Stat::Min)),
        );
        assert_eq!(&converted, &Some(expected_expr));
    }

    #[test]
    pub fn pruning_gt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = gt(get_item_scope(column.clone()), other_col.clone());

        let converted = pruning_expr(&not_eq_expr);
        // assert_eq!(
        //     refs.map(),
        //     &HashMap::from_iter([(
        //         FieldOrIdentity::Field(column.clone()),
        //         HashSet::from_iter([Stat::Max])
        //     ),])
        // );
        let expected_expr = lt_eq(
            get_item_scope(stat_field_name(&column, Stat::Max)),
            other_col.clone(),
        );
        assert_eq!(&converted, &Some(expected_expr));
    }

    #[test]
    pub fn pruning_lt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = lt(get_item_scope(column.clone()), other_expr.clone());

        let converted = pruning_expr(&not_eq_expr);
        // assert_eq!(
        //     refs.map(),
        //     &HashMap::from_iter([
        //         (
        //             FieldOrIdentity::Field(column.clone()),
        //             HashSet::from_iter([Stat::Min])
        //         ),
        //         (
        //             FieldOrIdentity::Field(other_col.clone()),
        //             HashSet::from_iter([Stat::Max])
        //         )
        //     ])
        // );
        let expected_expr = gt_eq(
            get_item_scope(stat_field_name(&column, Stat::Min)),
            get_item_scope(stat_field_name(&other_col, Stat::Max)),
        );
        assert_eq!(&converted, &Some(expected_expr));
    }

    #[test]
    pub fn pruning_lt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = lt(get_item_scope(column.clone()), other_col.clone());

        let converted = pruning_expr(&not_eq_expr);
        // assert_eq!(
        //     refs.map(),
        //     &HashMap::from_iter([(
        //         FieldOrIdentity::Field(column.clone()),
        //         HashSet::from_iter([Stat::Min])
        //     )])
        // );
        let expected_expr = gt_eq(
            get_item_scope(stat_field_name(&column, Stat::Min)),
            other_col.clone(),
        );
        assert_eq!(&converted, &Some(expected_expr));
    }

    #[test]
    fn unprojectable_expr() {
        let or_expr = lt(get_item_scope("a"), get_item_scope("b"));

        assert_eq!(
            pruning_expr(&or_expr),
            Some(gt_eq(get_item_scope("a_min"), get_item_scope("b_max")))
        )
    }

    #[test]
    fn pruning_identity() {
        let expr = or(lt(root().clone(), lit(10)), gt(root().clone(), lit(50)));

        let predicate = pruning_expr(&expr);

        let expected_expr = and(
            gt_eq(get_item_scope(FieldName::from("min")), lit(10)),
            lt_eq(get_item_scope(FieldName::from("max")), lit(50)),
        );
        assert_eq!(predicate, Some(expected_expr))
    }
    #[test]
    pub fn pruning_and_or_operators() {
        // Test case: a > 10 AND a < 50
        let column = FieldName::from("a");
        let and_expr = and(
            gt(get_item_scope(column.clone()), lit(10)),
            lt(get_item_scope(column), lit(50)),
        );
        let pruned = pruning_expr(&and_expr);

        // Expected: a_max <= 10 OR a_min >= 50
        assert_eq!(
            pruned,
            Some(or(
                lt_eq(get_item_scope(FieldName::from("a_max")), lit(10)),
                gt_eq(get_item_scope(FieldName::from("a_min")), lit(50))
            )),
        );
    }
}
