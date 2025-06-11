use std::iter;

use itertools::Itertools;
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{Field, FieldName, FieldPath};
use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use super::relation::Relation;
use crate::{AccessPath, ExprRef, Scope, StatsCatalog, get_item, var};

#[derive(Debug, Clone)]
pub struct PruningPredicate {
    expr: ExprRef,
    required_stats: Relation<AccessPath, Stat>,
}

impl PruningPredicate {
    pub fn try_new(original_expr: &ExprRef) -> Option<Self> {
        let (expr, required_stats) = pruning_expr(original_expr)?;

        Some(Self {
            expr,
            required_stats,
        })
    }

    pub fn expr(&self) -> &ExprRef {
        &self.expr
    }

    pub fn required_stats(&self) -> &HashMap<AccessPath, HashSet<Stat>> {
        self.required_stats.map()
    }

    /// Evaluate this predicate against a per-chunk statistics table.
    ///
    /// Returns Ok(None) if any of the required statistics are not present in metadata.
    /// If it returns Ok(Some(array)), the array is a boolean array with the same length as the
    /// metadata, and a true value means the chunk _can_ be pruned.
    pub fn evaluate(&self, metadata: &Scope) -> VortexResult<Option<ArrayRef>> {
        // TODO(joe): Replace this with a StatsCatalog that contains all the available stats and
        // build an expr using them.
        let known_stats = metadata
            .iter()
            .flat_map(|(access_path, stat)| {
                stat.dtype()
                    .as_struct()
                    .vortex_expect("metadata must be struct array")
                    .names()
                    .iter()
                    .map(|n| {
                        (
                            AccessPath::new(FieldPath::root(), access_path.clone()),
                            n.clone(),
                        )
                    })
            })
            .collect::<HashSet<(AccessPath, FieldName)>>();
        let required_stats = self
            .required_stats()
            .iter()
            .flat_map(|(path, stats)| stats.iter().map(|s| (path.clone(), s.name().into())))
            .collect::<HashSet<(AccessPath, FieldName)>>();

        let missing_stats = required_stats.difference(&known_stats).collect::<Vec<_>>();

        if !missing_stats.is_empty() {
            return Ok(None);
        }

        self.expr.evaluate(metadata).map(Some)
    }
}

#[derive(Default)]
struct FileStatsCatalog {
    usage: HashMap<(AccessPath, Stat), ExprRef>,
}

impl StatsCatalog for FileStatsCatalog {
    fn stats_ref(&mut self, access_path: &AccessPath, stat: Stat) -> Option<ExprRef> {
        let mut expr = var(access_path.identifier().clone());
        let name = access_path_stat_field_name(access_path, stat);
        expr = get_item(name, expr);
        self.usage.insert((access_path.clone(), stat), expr.clone());
        Some(expr)
    }
}

pub fn access_path_stat_field_name(access_path: &AccessPath, stat: Stat) -> FieldName {
    access_path
        .field_path
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

#[allow(clippy::type_complexity)]
// TODO: remove (Id, FieldPath) when updating FieldPath
pub fn pruning_expr(expr: &ExprRef) -> Option<(ExprRef, Relation<AccessPath, Stat>)> {
    let mut catalog = FileStatsCatalog {
        ..Default::default()
    };
    let expr = expr.stat_falsification(&mut catalog)?;

    let mut relation: Relation<AccessPath, Stat> = Relation::new();
    for ((field_path, stat), _) in catalog.usage.into_iter() {
        relation.insert(field_path, stat)
    }

    Some((expr, relation))
}

#[cfg(test)]
mod tests {
    use vortex_array::stats::Stat;
    use vortex_dtype::FieldName;

    use crate::pruning::pruning_predicate::{HashMap, pruning_expr};
    use crate::pruning::{PruningPredicate, access_path_stat_field_name};
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
                    access_path_stat_field_name(&AccessPath::root_field(name.clone()), Stat::Min),
                    root(),
                ),
                literal_eq.clone(),
            ),
            gt(
                literal_eq,
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(name),
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
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(column.clone()),
                    Stat::Min,
                )),
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(other_col.clone()),
                    Stat::Max,
                )),
            ),
            gt(
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(other_col),
                    Stat::Min,
                )),
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(column),
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
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(column.clone()),
                    Stat::Min,
                )),
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(other_col.clone()),
                    Stat::Max,
                )),
            ),
            eq(
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(column),
                    Stat::Max,
                )),
                get_item_scope(access_path_stat_field_name(
                    &AccessPath::root_field(other_col),
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
            get_item_scope(access_path_stat_field_name(
                &AccessPath::root_field(column),
                Stat::Max,
            )),
            get_item_scope(access_path_stat_field_name(
                &AccessPath::root_field(other_col),
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
            get_item_scope(access_path_stat_field_name(
                &AccessPath::root_field(column),
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
            get_item_scope(access_path_stat_field_name(
                &AccessPath::root_field(column),
                Stat::Min,
            )),
            get_item_scope(access_path_stat_field_name(
                &AccessPath::root_field(other_col),
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
            get_item_scope(access_path_stat_field_name(
                &AccessPath::root_field(column),
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
        assert!(PruningPredicate::try_new(&expr).is_none());
        // TODO(DK): a sufficiently complex pruner would produce: `x_max <= (y_max > z_min)`
    }
}
