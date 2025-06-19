use std::iter;

use itertools::Itertools;
use vortex_array::stats::Stat;
use vortex_dtype::{Field, FieldName, FieldPath};
use vortex_utils::aliases::hash_map::HashMap;

use super::relation::Relation;
use crate::{AccessPath, ExprRef, ScopeFieldPathSet, StatsCatalog, get_item, var};

pub type RequiredStats = Relation<AccessPath, Stat>;

// A catalog that return a stat column whenever it is required
#[derive(Default)]
struct AnyStatsCatalog {
    usage: HashMap<(AccessPath, Stat), ExprRef>,
}

// A catalog that return a stat column if it exists in the given scope.
struct ScopeStatsCatalog<'a> {
    any_catalog: AnyStatsCatalog,
    scope_field_paths: &'a ScopeFieldPathSet,
}

impl StatsCatalog for ScopeStatsCatalog<'_> {
    fn stats_ref(&mut self, access_path: &AccessPath, stat: Stat) -> Option<ExprRef> {
        let set = self.scope_field_paths.set(access_path.identifier())?;

        let stat_path = access_path
            .field_path
            .clone()
            .push(Field::Name(stat.name().into()));

        if set.contains(&stat_path) {
            self.any_catalog.stats_ref(access_path, stat)
        } else {
            None
        }
    }
}

impl StatsCatalog for AnyStatsCatalog {
    fn stats_ref(&mut self, access_path: &AccessPath, stat: Stat) -> Option<ExprRef> {
        let mut expr = var(access_path.identifier().clone());
        let name = field_path_stat_field_name(access_path.field_path(), stat);
        expr = get_item(name, expr);
        self.usage.insert((access_path.clone(), stat), expr.clone());
        Some(expr)
    }
}

pub fn field_path_stat_field_name(field_path: &FieldPath, stat: Stat) -> FieldName {
    field_path
        .path()
        .iter()
        .map(|f| match f {
            Field::Name(n) => n.as_ref(),
            Field::ElementType => todo!("element type not currently handled"),
        })
        .chain(iter::once(stat.name()))
        .join("_")
        .into()
}

/// Create a stat based falsification expr assuming all stats for all column in the expression
/// exist
pub fn pruning_expr(expr: &ExprRef) -> Option<(ExprRef, RequiredStats)> {
    let mut catalog = AnyStatsCatalog {
        ..Default::default()
    };
    let expr = expr.stat_falsification(&mut catalog)?;

    // TODO(joe): filter access by used exprs
    let mut relation: Relation<AccessPath, Stat> = Relation::new();
    for ((field_path, stat), _) in catalog.usage.into_iter() {
        relation.insert(field_path, stat)
    }

    Some((expr, relation))
}

/// Build a pruning expr mask an existing bundle of stats
/// Create a stat based falsification expr using the stats in the `scope_field_paths`.
/// These are of the form
/// [["col_0", ..., "col_n", "stat_name"], ...] for each stat.
pub fn checked_pruning_expr(
    expr: &ExprRef,
    scope_field_paths: &ScopeFieldPathSet,
) -> Option<(ExprRef, RequiredStats)> {
    let mut catalog = ScopeStatsCatalog {
        any_catalog: Default::default(),
        scope_field_paths,
    };

    let expr = expr.stat_falsification(&mut catalog)?;

    // TODO(joe): filter access by used exprs
    let mut relation: Relation<AccessPath, Stat> = Relation::new();
    for ((field_path, stat), _) in catalog.any_catalog.usage.into_iter() {
        relation.insert(field_path, stat)
    }

    Some((expr, relation))
}

#[cfg(test)]
mod tests {
    use vortex_array::stats::Stat;
    use vortex_dtype::{FieldName, FieldPath};

    use crate::pruning::field_path_stat_field_name;
    use crate::pruning::pruning_expr::{HashMap, pruning_expr};
    use crate::{
        AccessPath, HashSet, and, col, eq, get_item, get_item_scope, gt, gt_eq, lit, lt, lt_eq,
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
                get_item(
                    field_path_stat_field_name(&FieldPath::from_name(name.clone()), Stat::Min),
                    root(),
                ),
                literal_eq.clone(),
            ),
            gt(
                literal_eq,
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(name),
                    Stat::Max,
                )),
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
            refs.map(),
            &HashMap::from_iter([
                (
                    AccessPath::root_field(column.clone()),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    AccessPath::root_field(other_col.clone()),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = or(
            gt(
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(column.clone()),
                    Stat::Min,
                )),
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(other_col.clone()),
                    Stat::Max,
                )),
            ),
            gt(
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(other_col),
                    Stat::Min,
                )),
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(column),
                    Stat::Max,
                )),
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
            refs.map(),
            &HashMap::from_iter([
                (
                    AccessPath::root_field(column.clone()),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    AccessPath::root_field(other_col.clone()),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = and(
            eq(
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(column.clone()),
                    Stat::Min,
                )),
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(other_col.clone()),
                    Stat::Max,
                )),
            ),
            eq(
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(column),
                    Stat::Max,
                )),
                get_item_scope(field_path_stat_field_name(
                    &FieldPath::from_name(other_col),
                    Stat::Min,
                )),
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
            refs.map(),
            &HashMap::from_iter([
                (
                    AccessPath::root_field(column.clone()),
                    HashSet::from_iter([Stat::Max])
                ),
                (
                    AccessPath::root_field(other_col.clone()),
                    HashSet::from_iter([Stat::Min])
                )
            ])
        );
        let expected_expr = lt_eq(
            get_item_scope(field_path_stat_field_name(
                &FieldPath::from_name(column),
                Stat::Max,
            )),
            get_item_scope(field_path_stat_field_name(
                &FieldPath::from_name(other_col),
                Stat::Min,
            )),
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
            refs.map(),
            &HashMap::from_iter([(
                AccessPath::root_field(column.clone()),
                HashSet::from_iter([Stat::Max])
            ),])
        );
        let expected_expr = lt_eq(
            get_item_scope(field_path_stat_field_name(
                &FieldPath::from_name(column),
                Stat::Max,
            )),
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
            refs.map(),
            &HashMap::from_iter([
                (
                    AccessPath::root_field(column.clone()),
                    HashSet::from_iter([Stat::Min])
                ),
                (
                    AccessPath::root_field(other_col.clone()),
                    HashSet::from_iter([Stat::Max])
                )
            ])
        );
        let expected_expr = gt_eq(
            get_item_scope(field_path_stat_field_name(
                &FieldPath::from_name(column),
                Stat::Min,
            )),
            get_item_scope(field_path_stat_field_name(
                &FieldPath::from_name(other_col),
                Stat::Max,
            )),
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
            refs.map(),
            &HashMap::from_iter([(
                AccessPath::root_field(column.clone()),
                HashSet::from_iter([Stat::Min])
            )])
        );
        let expected_expr = gt_eq(
            get_item_scope(field_path_stat_field_name(
                &FieldPath::from_name(column),
                Stat::Min,
            )),
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
        assert!(pruning_expr(&expr).is_none());
        // TODO(DK): a sufficiently complex pruner would produce: `x_max <= (y_max > z_min)`
    }
}
