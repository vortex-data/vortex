use std::iter;

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::stats::Stat;
use vortex_dtype::{Field, FieldPath};

use crate::pruning::StatsCatalog;
use crate::{ExprRef, Identifier, get_item, var};

#[derive(Default)]
struct FileStatsCatalog {
    usage: HashMap<(Identifier, FieldPath, Stat), ExprRef>,
}

impl StatsCatalog for FileStatsCatalog {
    fn stats_ref(&mut self, id: &Identifier, field: &FieldPath, stat: Stat) -> Option<ExprRef> {
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
        self.usage
            .insert((id.clone(), field.clone(), stat), expr.clone());
        Some(expr)
    }
}

pub fn pruning_expr(
    expr: &ExprRef,
) -> Option<(ExprRef, HashMap<(Identifier, FieldPath), HashSet<Stat>>)> {
    let mut catalog = FileStatsCatalog {
        ..Default::default()
    };
    let Some(expr) = expr.prune_expr(&mut catalog) else {
        return None;
    };

    let mut relation: HashMap<(Identifier, FieldPath), HashSet<Stat>> = HashMap::new();
    for (k, v, s) in catalog.usage.keys() {
        relation
            .entry((k.clone(), v.clone()))
            .or_default()
            .insert(*s);
    }

    Some((expr, relation))
}

#[cfg(test)]
mod tests {

    use vortex_array::stats::Stat;
    use vortex_dtype::{FieldName, FieldPath};

    use crate::pruning::stat_field_name;
    use crate::pruning::v2::{HashMap, pruning_expr};
    use crate::{
        HashSet, IDENTITY_IDENTIFIER, and, eq, get_item, get_item_scope, gt, gt_eq, lit, lt, lt_eq,
        not_eq, or, root,
    };

    #[test]
    pub fn pruning_equals() {
        let name = FieldName::from("a");
        let literal_eq = lit(42);
        let eq_expr = eq(get_item("a", root()), literal_eq.clone());
        let (converted, _refs) = pruning_expr(&eq_expr).unwrap();
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
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_equals_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let eq_expr = eq(
            get_item_scope(column.clone()),
            get_item_scope(other_col.clone()),
        );

        let (converted, refs) = pruning_expr(&eq_expr).unwrap();
        assert_eq!(
            refs,
            HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&other_col)),
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
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_not_equals_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let not_eq_expr = not_eq(
            get_item_scope(column.clone()),
            get_item_scope(other_col.clone()),
        );

        let (converted, refs) = pruning_expr(&not_eq_expr).unwrap();
        assert_eq!(
            refs,
            HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&other_col)),
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

        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_gt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = gt(get_item_scope(column.clone()), other_expr.clone());

        let (converted, refs) = pruning_expr(&not_eq_expr).unwrap();
        assert_eq!(
            refs,
            HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Max])
                ),
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&other_col)),
                    HashSet::from_iter([Stat::Min])
                )
            ])
        );
        let expected_expr = lt_eq(
            get_item_scope(stat_field_name(&column, Stat::Max)),
            get_item_scope(stat_field_name(&other_col, Stat::Min)),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_gt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = gt(get_item_scope(column.clone()), other_col.clone());

        let (converted, refs) = pruning_expr(&not_eq_expr).unwrap();
        assert_eq!(
            refs,
            HashMap::from_iter([(
                (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&column)),
                HashSet::from_iter([Stat::Max])
            ),])
        );
        let expected_expr = lt_eq(
            get_item_scope(stat_field_name(&column, Stat::Max)),
            other_col.clone(),
        );
        assert_eq!(&converted, &(expected_expr));
    }

    #[test]
    pub fn pruning_lt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = lt(get_item_scope(column.clone()), other_expr.clone());

        let (converted, refs) = pruning_expr(&not_eq_expr).unwrap();
        assert_eq!(
            refs,
            HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Min])
                ),
                (
                    (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&other_col)),
                    HashSet::from_iter([Stat::Max])
                )
            ])
        );
        let expected_expr = gt_eq(
            get_item_scope(stat_field_name(&column, Stat::Min)),
            get_item_scope(stat_field_name(&other_col, Stat::Max)),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_lt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = lt(get_item_scope(column.clone()), other_col.clone());

        let (converted, refs) = pruning_expr(&not_eq_expr).unwrap();
        assert_eq!(
            refs,
            HashMap::from_iter([(
                (IDENTITY_IDENTIFIER.into(), FieldPath::from_name(&column)),
                HashSet::from_iter([Stat::Min])
            )])
        );
        let expected_expr = gt_eq(
            get_item_scope(stat_field_name(&column, Stat::Min)),
            other_col.clone(),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    fn pruning_identity() {
        let expr = or(lt(root().clone(), lit(10)), gt(root().clone(), lit(50)));

        let (predicate, _) = pruning_expr(&expr).unwrap();

        let expected_expr = and(
            gt_eq(get_item_scope(FieldName::from("min")), lit(10)),
            lt_eq(get_item_scope(FieldName::from("max")), lit(50)),
        );
        assert_eq!(&predicate, &expected_expr)
    }
    #[test]
    pub fn pruning_and_or_operators() {
        // Test case: a > 10 AND a < 50
        let column = FieldName::from("a");
        let and_expr = and(
            gt(get_item_scope(column.clone()), lit(10)),
            lt(get_item_scope(column), lit(50)),
        );
        let (predicate, _) = pruning_expr(&and_expr).unwrap();

        // Expected: a_max <= 10 OR a_min >= 50
        assert_eq!(
            &predicate,
            &or(
                lt_eq(get_item_scope(FieldName::from("a_max")), lit(10)),
                gt_eq(get_item_scope(FieldName::from("a_min")), lit(50))
            ),
        );
    }
}
