// This code doesn't have usage outside of tests yet, remove once usage is added
#![allow(dead_code)]

use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{FieldName, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::between::Between;
use crate::{
    BinaryExpr, ExprRef, GetItem, Identity, Literal, Not, Operator, VortexExprExt, and, eq,
    get_item, gt, ident, lit, not, or,
};

#[derive(Debug, Clone)]
pub struct Relation<K, V> {
    map: HashMap<K, HashSet<V>>,
}

impl<K: Display, V: Display> Display for Relation<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.map.iter().format_with(",", |(k, v), fmt| {
                fmt(&format_args!("{k}: {{{}}}", v.iter().format(",")))
            })
        )
    }
}

impl<K: Hash + Eq, V: Hash + Eq> Default for Relation<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq, V: Hash + Eq> Relation<K, V> {
    pub fn new() -> Self {
        Relation {
            map: HashMap::new(),
        }
    }

    pub fn union(mut iter: impl Iterator<Item = Relation<K, V>>) -> Relation<K, V> {
        if let Some(mut x) = iter.next() {
            for y in iter {
                x.extend(y)
            }
            x
        } else {
            Relation::new()
        }
    }

    pub fn extend(&mut self, other: Relation<K, V>) {
        for (l, rs) in other.map.into_iter() {
            self.map.entry(l).or_default().extend(rs.into_iter())
        }
    }

    pub fn insert(&mut self, k: K, v: V) {
        self.map.entry(k).or_default().insert(v);
    }

    pub fn into_map(self) -> HashMap<K, HashSet<V>> {
        self.map
    }
}

#[derive(Debug, Clone)]
pub struct PruningPredicate {
    expr: ExprRef,
    required_stats: Relation<FieldOrIdentity, Stat>,
}

impl Display for PruningPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PruningPredicate({}, {{{}}})",
            self.expr, self.required_stats
        )
    }
}

impl PruningPredicate {
    pub fn try_new(original_expr: &ExprRef) -> Option<Self> {
        let (expr, required_stats) = convert_to_pruning_expression(original_expr);
        if let Some(lexp) = expr.as_any().downcast_ref::<Literal>() {
            // Is the expression constant false, i.e. prune nothing
            if lexp
                .value()
                .as_bool_opt()
                .and_then(|b| b.value())
                .map(|b| !b)
                .unwrap_or(false)
            {
                None
            } else {
                Some(Self {
                    expr,
                    required_stats,
                })
            }
        } else {
            Some(Self {
                expr,
                required_stats,
            })
        }
    }

    pub fn expr(&self) -> &ExprRef {
        &self.expr
    }

    pub fn required_stats(&self) -> &HashMap<FieldOrIdentity, HashSet<Stat>> {
        &self.required_stats.map
    }

    /// Evaluate this predicate against a per-chunk statistics table.
    ///
    /// Returns Ok(None) if any of the required statistics are not present in metadata.
    /// If it returns Ok(Some(array)), the array is a boolean array with the same length as the
    /// metadata, and a true value means the chunk _can_ be pruned.
    pub fn evaluate(&self, metadata: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        let known_stats = HashSet::from_iter(
            metadata
                .as_struct_typed()
                .vortex_expect("metadata must be struct array")
                .names()
                .iter()
                .map(|x| x.to_string()),
        );
        let required_stats = self
            .required_stats()
            .iter()
            .flat_map(|(key, value)| value.iter().map(|stat| key.stat_field_name_string(*stat)))
            .collect::<HashSet<_>>();
        let missing_stats = required_stats.difference(&known_stats).collect::<Vec<_>>();

        if !missing_stats.is_empty() {
            return Ok(None);
        }

        Ok(Some(self.expr.evaluate(metadata)?))
    }
}

fn not_prunable() -> PruningPredicateStats {
    (
        lit(Scalar::bool(false, Nullability::NonNullable)),
        Relation::new(),
    )
}

// Anything that can't be translated has to be represented as
// boolean true expression, i.e. the value might be in that chunk
fn convert_to_pruning_expression(expr: &ExprRef) -> PruningPredicateStats {
    if let Some(nexp) = expr.as_any().downcast_ref::<Not>() {
        if let Some(get_item) = nexp.child().as_any().downcast_ref::<GetItem>() {
            if get_item.child().as_any().is::<Identity>() {
                return convert_access_reference(expr, true);
            }
        }
    }

    if let Some(get_item) = expr.as_any().downcast_ref::<GetItem>() {
        if get_item.child().as_any().is::<Identity>() {
            return convert_access_reference(expr, false);
        }
    }

    if let Some(bexp) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if bexp.op() == Operator::Or || bexp.op() == Operator::And {
            let (rewritten_left, mut refs_lhs) = convert_to_pruning_expression(bexp.lhs());
            let (rewritten_right, refs_rhs) = convert_to_pruning_expression(bexp.rhs());
            refs_lhs.extend(refs_rhs);
            let flipped_op = bexp
                .op()
                .logical_inverse()
                .vortex_expect("Can not be any other operator than and / or");
            return (
                BinaryExpr::new_expr(rewritten_left, flipped_op, rewritten_right),
                refs_lhs,
            );
        }

        if let Some(get_item) = bexp.lhs().as_any().downcast_ref::<GetItem>() {
            if get_item.child().as_any().is::<Identity>() {
                return PruningPredicateRewriter::rewrite_binary_op(
                    FieldOrIdentity::Field(get_item.field().clone()),
                    bexp.op(),
                    bexp.rhs(),
                );
            }
        };

        if let Some(get_item) = bexp.rhs().as_any().downcast_ref::<GetItem>() {
            if get_item.child().as_any().is::<Identity>() {
                return PruningPredicateRewriter::rewrite_binary_op(
                    FieldOrIdentity::Field(get_item.field().clone()),
                    bexp.op().swap(),
                    bexp.lhs(),
                );
            }
        }

        if bexp.lhs().as_any().is::<Identity>() {
            return PruningPredicateRewriter::rewrite_binary_op(
                FieldOrIdentity::Identity,
                bexp.op(),
                bexp.rhs(),
            );
        };

        if bexp.rhs().as_any().is::<Identity>() {
            return PruningPredicateRewriter::rewrite_binary_op(
                FieldOrIdentity::Identity,
                bexp.op().swap(),
                bexp.lhs(),
            );
        };
    }

    if let Some(between_expr) = expr.as_any().downcast_ref::<Between>() {
        return convert_to_pruning_expression(&between_expr.to_binary_expr());
    }

    not_prunable()
}

fn convert_access_reference(expr: &ExprRef, invert: bool) -> PruningPredicateStats {
    let mut refs = Relation::new();
    let Some(min_expr) = replace_get_item_with_stat(expr, Stat::Min, &mut refs) else {
        return not_prunable();
    };
    let Some(max_expr) = replace_get_item_with_stat(expr, Stat::Max, &mut refs) else {
        return not_prunable();
    };

    let expr = if invert {
        and(min_expr, max_expr)
    } else {
        not(or(min_expr, max_expr))
    };

    (expr, refs)
}

struct PruningPredicateRewriter<'a> {
    access: FieldOrIdentity,
    operator: Operator,
    other_exp: &'a ExprRef,
    stats_to_fetch: Relation<FieldOrIdentity, Stat>,
}

type PruningPredicateStats = (ExprRef, Relation<FieldOrIdentity, Stat>);

impl<'a> PruningPredicateRewriter<'a> {
    pub fn try_new(
        access: FieldOrIdentity,
        operator: Operator,
        other_exp: &'a ExprRef,
    ) -> Option<Self> {
        // TODO(robert): Simplify expression to guarantee that each column is not compared to itself
        //  For majority of cases self column references are likely not prunable
        if let FieldOrIdentity::Field(field) = &access {
            if other_exp.references().contains(field) {
                return None;
            }
        };

        Some(Self {
            access,
            operator,
            other_exp,
            stats_to_fetch: Relation::new(),
        })
    }

    pub fn rewrite_binary_op(
        access: FieldOrIdentity,
        operator: Operator,
        other_exp: &'a ExprRef,
    ) -> PruningPredicateStats {
        Self::try_new(access, operator, other_exp)
            .and_then(Self::rewrite)
            .unwrap_or_else(not_prunable)
    }

    fn add_stat_reference(&mut self, stat: Stat) -> FieldName {
        let new_field = self.access.stat_field_name(stat);
        self.stats_to_fetch.insert(self.access.clone(), stat);
        new_field
    }

    fn rewrite_other_exp(&mut self, stat: Stat) -> ExprRef {
        replace_get_item_with_stat(self.other_exp, stat, &mut self.stats_to_fetch)
            .unwrap_or_else(|| self.other_exp.clone())
    }

    fn rewrite(mut self) -> Option<PruningPredicateStats> {
        let expr: Option<ExprRef> = match self.operator {
            Operator::Eq => {
                let min_col = get_item(self.add_stat_reference(Stat::Min), ident());
                let max_col = get_item(self.add_stat_reference(Stat::Max), ident());
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(or(gt(min_col, replaced_max), gt(replaced_min, max_col)))
            }
            Operator::NotEq => {
                let min_col = get_item(self.add_stat_reference(Stat::Min), ident());
                let max_col = get_item(self.add_stat_reference(Stat::Max), ident());
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(and(eq(min_col, replaced_max), eq(max_col, replaced_min)))
            }
            Operator::Gt | Operator::Gte => {
                let max_col = get_item(self.add_stat_reference(Stat::Max), ident());
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(BinaryExpr::new_expr(
                    max_col,
                    self.operator
                        .inverse()
                        .vortex_expect("inverse of gt & gt_eq defined"),
                    replaced_min,
                ))
            }
            Operator::Lt | Operator::Lte => {
                let min_col = get_item(self.add_stat_reference(Stat::Min), ident());
                let replaced_max = self.rewrite_other_exp(Stat::Max);

                Some(BinaryExpr::new_expr(
                    min_col,
                    self.operator
                        .inverse()
                        .vortex_expect("inverse of lt & lte defined"),
                    replaced_max,
                ))
            }
            _ => None,
        };
        expr.map(|e| (e, self.stats_to_fetch))
    }
}

fn replace_get_item_with_stat(
    expr: &ExprRef,
    stat: Stat,
    stats_to_fetch: &mut Relation<FieldOrIdentity, Stat>,
) -> Option<ExprRef> {
    if let Some(get_i) = expr.as_any().downcast_ref::<GetItem>() {
        if get_i.child().as_any().is::<Identity>() {
            let new_field = stat_field_name(get_i.field(), stat);
            stats_to_fetch.insert(FieldOrIdentity::Field(get_i.field().clone()), stat);
            return Some(get_item(new_field, ident()));
        }
    }

    if let Some(not_expr) = expr.as_any().downcast_ref::<Not>() {
        let rewritten = replace_get_item_with_stat(not_expr.child(), stat, stats_to_fetch)?;
        return Some(not(rewritten));
    }

    if let Some(bexp) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let rewritten_lhs = replace_get_item_with_stat(bexp.lhs(), stat, stats_to_fetch);
        let rewritten_rhs = replace_get_item_with_stat(bexp.rhs(), stat, stats_to_fetch);
        if rewritten_lhs.is_none() && rewritten_rhs.is_none() {
            return None;
        }

        let lhs = rewritten_lhs.unwrap_or_else(|| bexp.lhs().clone());
        let rhs = rewritten_rhs.unwrap_or_else(|| bexp.rhs().clone());

        return Some(BinaryExpr::new_expr(lhs, bexp.op(), rhs));
    }

    None
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum FieldOrIdentity {
    Field(FieldName),
    Identity,
}

pub(crate) fn stat_field_name(field: &FieldName, stat: Stat) -> FieldName {
    FieldName::from(stat_field_name_string(field, stat))
}

pub(crate) fn stat_field_name_string(field: &FieldName, stat: Stat) -> String {
    format!("{field}_{stat}")
}

impl FieldOrIdentity {
    pub(crate) fn stat_field_name(&self, stat: Stat) -> FieldName {
        FieldName::from(self.stat_field_name_string(stat))
    }

    pub(crate) fn stat_field_name_string(&self, stat: Stat) -> String {
        match self {
            FieldOrIdentity::Field(field) => stat_field_name_string(field, stat),
            FieldOrIdentity::Identity => stat.to_string(),
        }
    }
}

impl Display for FieldOrIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldOrIdentity::Field(field) => write!(f, "{}", field),
            FieldOrIdentity::Identity => write!(f, "$[]"),
        }
    }
}

impl<T> From<T> for FieldOrIdentity
where
    FieldName: From<T>,
{
    fn from(value: T) -> Self {
        FieldOrIdentity::Field(FieldName::from(value))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::aliases::hash_map::HashMap;
    use vortex_array::aliases::hash_set::HashSet;
    use vortex_array::stats::Stat;
    use vortex_dtype::FieldName;

    use crate::pruning::{
        FieldOrIdentity, PruningPredicate, convert_to_pruning_expression, stat_field_name,
    };
    use crate::{
        and, eq, get_item, get_item_scope, gt, gt_eq, ident, lit, lt, lt_eq, not, not_eq, or,
    };

    #[test]
    pub fn pruning_equals() {
        let name = FieldName::from("a");
        let literal_eq = lit(42);
        let eq_expr = eq(get_item("a", ident()), literal_eq.clone());
        let (converted, refs) = convert_to_pruning_expression(&eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([(
                FieldOrIdentity::Field(name.clone()),
                HashSet::from_iter([Stat::Min, Stat::Max])
            )])
        );
        let expected_expr = or(
            gt(
                get_item(stat_field_name(&name, Stat::Min), ident()),
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

        let (converted, refs) = convert_to_pruning_expression(&eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([
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

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([
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

        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_gt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = gt(get_item_scope(column.clone()), other_expr.clone());

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([
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
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_gt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = gt(get_item_scope(column.clone()), other_col.clone());

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([(
                FieldOrIdentity::Field(column.clone()),
                HashSet::from_iter([Stat::Max])
            ),])
        );
        let expected_expr = lt_eq(
            get_item_scope(stat_field_name(&column, Stat::Max)),
            other_col.clone(),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_lt_column() {
        let column = FieldName::from("a");
        let other_col = FieldName::from("b");
        let other_expr = get_item_scope(other_col.clone());
        let not_eq_expr = lt(get_item_scope(column.clone()), other_expr.clone());

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([
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
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_lt_value() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = lt(get_item_scope(column.clone()), other_col.clone());

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([(
                FieldOrIdentity::Field(column.clone()),
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
    fn unprojectable_expr() {
        let or_expr = not(lt(get_item_scope("a"), get_item_scope("b")));
        assert!(PruningPredicate::try_new(&or_expr).is_none());
    }

    #[test]
    fn display_pruning_predicate() {
        let column = FieldName::from("a");
        let other_col = lit(42);
        let not_eq_expr = lt(get_item_scope(column), other_col);

        assert_eq!(
            PruningPredicate::try_new(&not_eq_expr).unwrap().to_string(),
            "PruningPredicate(($.a_min >= 42i32), {a: {min}})"
        );
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
        let expr = ident();
        let expr = or(lt(expr.clone(), lit(10)), gt(expr.clone(), lit(50)));

        let expected = HashMap::from([(
            FieldOrIdentity::Identity,
            HashSet::from([Stat::Min, Stat::Max]),
        )]);

        let predicate = PruningPredicate::try_new(&expr).unwrap();
        assert_eq!(predicate.required_stats(), &expected);

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
        );
    }
}
