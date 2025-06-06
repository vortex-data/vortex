use std::iter;

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{Field, FieldPath};
use vortex_error::{VortexExpect as _, VortexResult};

use super::field_or_identity::FieldOrIdentity;
use super::relation::Relation;
use crate::pruning::StatsCatalog;
use crate::{ExprRef, Identifier, Literal, Scope, get_item, var};

#[derive(Debug, Clone)]
pub struct PruningPredicate {
    expr: ExprRef,
    required_stats: Relation<FieldOrIdentity, Stat>,
}

impl PruningPredicate {
    pub fn try_new(original_expr: &ExprRef) -> Option<Self> {
        let (expr, required_stats) = pruning_expr(original_expr)?;

        if let Some(lexp) = expr.as_any().downcast_ref::<Literal>() {
            // Is the expression constant false, i.e. prune nothing
            if lexp.value().as_bool_opt().and_then(|b| b.value()) == Some(false) {
                return None;
            }
        }

        let required_stats = Relation::from(
            required_stats
                .into_iter()
                .map(|((_id, path), v)| {
                    let key = if path.is_root() {
                        FieldOrIdentity::Identity
                    } else {
                        assert_eq!(path.path().len(), 1);
                        let Field::Name(n) = &path.path()[0] else {
                            todo!("cannot have list")
                        };
                        FieldOrIdentity::Field(n.clone())
                    };
                    (key, v)
                })
                .collect::<HashMap<_, _>>(),
        );

        Some(Self {
            expr,
            required_stats,
        })
    }

    pub fn expr(&self) -> &ExprRef {
        &self.expr
    }

    pub fn required_stats(&self) -> &HashMap<FieldOrIdentity, HashSet<Stat>> {
        self.required_stats.map()
    }

    /// Evaluate this predicate against a per-chunk statistics table.
    ///
    /// Returns Ok(None) if any of the required statistics are not present in metadata.
    /// If it returns Ok(Some(array)), the array is a boolean array with the same length as the
    /// metadata, and a true value means the chunk _can_ be pruned.
    pub fn evaluate(&self, metadata: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        let known_stats = metadata
            .dtype()
            .as_struct()
            .vortex_expect("metadata must be struct array")
            .names()
            .iter()
            .map(|x| x.to_string())
            .collect::<HashSet<_>>();
        let required_stats = self
            .required_stats()
            .iter()
            .flat_map(|(key, value)| value.iter().map(|stat| key.stat_field_name_string(*stat)))
            .collect::<HashSet<_>>();
        let missing_stats = required_stats.difference(&known_stats).collect::<Vec<_>>();

        if !missing_stats.is_empty() {
            return Ok(None);
        }

        Ok(Some(self.expr.evaluate(&Scope::new(metadata.to_array()))?))
    }
}

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

#[allow(clippy::type_complexity)]
// TODO: remove (Id, FieldPath) when updating FieldPath
pub fn pruning_expr(expr: &ExprRef) -> Option<(ExprRef, Relation<(Identifier, FieldPath), Stat>)> {
    let mut catalog = FileStatsCatalog {
        ..Default::default()
    };
    let expr = expr.prune_expr(&mut catalog)?;

    let mut relation: Relation<(Identifier, FieldPath), Stat> = Relation::new();
    for (k, v, s) in catalog.usage.keys() {
        relation.insert((k.clone(), v.clone()), *s)
    }

    Some((expr, relation))
}

#[cfg(test)]
mod tests {

    use vortex_array::stats::Stat;
    use vortex_dtype::{FieldName, FieldPath};

    use crate::pruning::pruning_predicate::{HashMap, pruning_expr};
    use crate::pruning::stat_field_name;
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
            refs.map(),
            &HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.clone(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    (
                        IDENTITY_IDENTIFIER.clone(),
                        FieldPath::from_name(&other_col)
                    ),
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
            refs.map(),
            &HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.clone(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Min, Stat::Max])
                ),
                (
                    (
                        IDENTITY_IDENTIFIER.clone(),
                        FieldPath::from_name(&other_col)
                    ),
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
            refs.map(),
            &HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.clone(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Max])
                ),
                (
                    (
                        IDENTITY_IDENTIFIER.clone(),
                        FieldPath::from_name(&other_col)
                    ),
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
            refs.map(),
            &HashMap::from_iter([(
                (IDENTITY_IDENTIFIER.clone(), FieldPath::from_name(&column)),
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
            refs.map(),
            &HashMap::from_iter([
                (
                    (IDENTITY_IDENTIFIER.clone(), FieldPath::from_name(&column)),
                    HashSet::from_iter([Stat::Min])
                ),
                (
                    (
                        IDENTITY_IDENTIFIER.clone(),
                        FieldPath::from_name(&other_col)
                    ),
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
            refs.map(),
            &HashMap::from_iter([(
                (IDENTITY_IDENTIFIER.clone(), FieldPath::from_name(&column)),
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
