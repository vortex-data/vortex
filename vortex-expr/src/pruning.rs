// This code doesn't have usage outside of tests yet, remove once usage is added
#![allow(dead_code)]

use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::stats::Stat;
use vortex_array::ArrayData;
use vortex_dtype::{Field, Nullability};
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::{
    and, col, eq, gt, gt_eq, lit, lt_eq, or, BinaryExpr, Column, ExprRef, Identity, Literal, Not,
    Operator, RowFilter, VortexExprExt,
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
    pub fn evaluate(&self, metadata: &ArrayData) -> VortexResult<Option<ArrayData>> {
        let known_stats = HashSet::from_iter(
            metadata
                .as_struct_array()
                .vortex_expect("metadata must be struct array")
                .names()
                .iter()
                .map(|x| x.to_string()),
        );
        let required_stats = self
            .required_stats()
            .iter()
            .flat_map(|(key, value)| value.iter().map(|stat| key.stat_column_name_string(*stat)))
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
        if nexp.child().as_any().downcast_ref::<Column>().is_some() {
            return convert_column_reference(expr, true);
        }
    }

    if expr.as_any().downcast_ref::<Column>().is_some() {
        return convert_column_reference(expr, false);
    }

    if let Some(bexp) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if bexp.op() == Operator::Or || bexp.op() == Operator::And {
            let (rewritten_left, mut refs_lhs) = convert_to_pruning_expression(bexp.lhs());
            let (rewritten_right, refs_rhs) = convert_to_pruning_expression(bexp.rhs());
            refs_lhs.extend(refs_rhs);
            return (
                BinaryExpr::new_expr(rewritten_left, bexp.op(), rewritten_right),
                refs_lhs,
            );
        }

        if let Some(col) = bexp.lhs().as_any().downcast_ref::<Column>() {
            return PruningPredicateRewriter::rewrite_binary_op(
                FieldOrIdentity::Field(col.field().clone()),
                bexp.op(),
                bexp.rhs(),
            );
        };

        if let Some(col) = bexp.rhs().as_any().downcast_ref::<Column>() {
            return PruningPredicateRewriter::rewrite_binary_op(
                FieldOrIdentity::Field(col.field().clone()),
                bexp.op().swap(),
                bexp.lhs(),
            );
        }

        if bexp.lhs().as_any().downcast_ref::<Identity>().is_some() {
            return PruningPredicateRewriter::rewrite_binary_op(
                FieldOrIdentity::Identity,
                bexp.op(),
                bexp.rhs(),
            );
        };

        if bexp.rhs().as_any().downcast_ref::<Identity>().is_some() {
            return PruningPredicateRewriter::rewrite_binary_op(
                FieldOrIdentity::Identity,
                bexp.op().swap(),
                bexp.lhs(),
            );
        };
    }

    if let Some(RowFilter { conjunction }) = expr.as_any().downcast_ref::<RowFilter>() {
        let (rewritten_conjunction, refses): (Vec<ExprRef>, Vec<Relation<FieldOrIdentity, Stat>>) =
            conjunction
                .iter()
                .map(convert_to_pruning_expression)
                .unzip();

        let refs = Relation::union(refses.into_iter());

        return (
            RowFilter::from_conjunction_expr(rewritten_conjunction),
            refs,
        );
    }

    not_prunable()
}

fn convert_column_reference(expr: &ExprRef, invert: bool) -> PruningPredicateStats {
    let mut refs = Relation::new();
    let Some(min_expr) = replace_column_with_stat(expr, Stat::Min, &mut refs) else {
        return not_prunable();
    };
    let Some(max_expr) = replace_column_with_stat(expr, Stat::Max, &mut refs) else {
        return not_prunable();
    };

    let expr = if invert {
        and(min_expr, max_expr)
    } else {
        Not::new_expr(or(min_expr, max_expr))
    };

    (expr, refs)
}

struct PruningPredicateRewriter<'a> {
    column: FieldOrIdentity,
    operator: Operator,
    other_exp: &'a ExprRef,
    stats_to_fetch: Relation<FieldOrIdentity, Stat>,
}

type PruningPredicateStats = (ExprRef, Relation<FieldOrIdentity, Stat>);

impl<'a> PruningPredicateRewriter<'a> {
    pub fn try_new(
        column: FieldOrIdentity,
        operator: Operator,
        other_exp: &'a ExprRef,
    ) -> Option<Self> {
        // TODO(robert): Simplify expression to guarantee that each column is not compared to itself
        //  For majority of cases self column references are likely not prunable
        if let FieldOrIdentity::Field(field) = &column {
            if other_exp.references().contains(field) {
                return None;
            }
        };

        Some(Self {
            column,
            operator,
            other_exp,
            stats_to_fetch: Relation::new(),
        })
    }

    pub fn rewrite_binary_op(
        column: FieldOrIdentity,
        operator: Operator,
        other_exp: &'a ExprRef,
    ) -> PruningPredicateStats {
        Self::try_new(column, operator, other_exp)
            .and_then(Self::rewrite)
            .unwrap_or_else(not_prunable)
    }

    fn add_stat_reference(&mut self, stat: Stat) -> Field {
        let new_field = self.column.stat_column_field(stat);
        self.stats_to_fetch.insert(self.column.clone(), stat);
        new_field
    }

    fn rewrite_other_exp(&mut self, stat: Stat) -> ExprRef {
        replace_column_with_stat(self.other_exp, stat, &mut self.stats_to_fetch)
            .unwrap_or_else(|| self.other_exp.clone())
    }

    fn rewrite(mut self) -> Option<PruningPredicateStats> {
        let expr: Option<ExprRef> = match self.operator {
            Operator::Eq => {
                let min_col = col(self.add_stat_reference(Stat::Min));
                let max_col = col(self.add_stat_reference(Stat::Max));
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(or(gt(min_col, replaced_max), gt(replaced_min, max_col)))
            }
            Operator::NotEq => {
                let min_col = col(self.add_stat_reference(Stat::Min));
                let max_col = col(self.add_stat_reference(Stat::Max));
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                let column_value_is_single_known_value = eq(min_col.clone(), max_col.clone());
                let column_value = min_col;

                let other_value_is_single_known_value =
                    eq(replaced_min.clone(), replaced_max.clone());
                let other_value = replaced_min;

                Some(and(
                    and(
                        column_value_is_single_known_value,
                        other_value_is_single_known_value,
                    ),
                    eq(column_value, other_value),
                ))
            }
            Operator::Gt | Operator::Gte => {
                let max_col = col(self.add_stat_reference(Stat::Max));
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(lt_eq(max_col, replaced_min))
            }
            Operator::Lt | Operator::Lte => {
                let min_col = col(self.add_stat_reference(Stat::Min));
                let replaced_max = self.rewrite_other_exp(Stat::Max);

                Some(gt_eq(min_col, replaced_max))
            }
            _ => None,
        };
        expr.map(|e| (e, self.stats_to_fetch))
    }
}

fn replace_column_with_stat(
    expr: &ExprRef,
    stat: Stat,
    stats_to_fetch: &mut Relation<FieldOrIdentity, Stat>,
) -> Option<ExprRef> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        let new_field = stat_column_field(column.field(), stat);
        stats_to_fetch.insert(FieldOrIdentity::Field(column.field().clone()), stat);
        return Some(col(new_field));
    }

    if let Some(not) = expr.as_any().downcast_ref::<Not>() {
        let rewritten = replace_column_with_stat(not.child(), stat, stats_to_fetch)?;
        return Some(Not::new_expr(rewritten));
    }

    if let Some(bexp) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let rewritten_lhs = replace_column_with_stat(bexp.lhs(), stat, stats_to_fetch);
        let rewritten_rhs = replace_column_with_stat(bexp.rhs(), stat, stats_to_fetch);
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
    Field(Field),
    Identity,
}

pub(crate) fn stat_column_field(field: &Field, stat: Stat) -> Field {
    Field::from(stat_column_name_string(field, stat))
}

pub(crate) fn stat_column_name_string(field: &Field, stat: Stat) -> String {
    match field {
        Field::Name(n) => format!("{n}_{stat}"),
        Field::Index(i) => format!("{i}_{stat}"),
    }
}

impl FieldOrIdentity {
    pub(crate) fn stat_column_field(&self, stat: Stat) -> Field {
        Field::from(self.stat_column_name_string(stat))
    }

    pub(crate) fn stat_column_name_string(&self, stat: Stat) -> String {
        match self {
            FieldOrIdentity::Field(field) => stat_column_name_string(field, stat),
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
    Field: From<T>,
{
    fn from(value: T) -> Self {
        FieldOrIdentity::Field(Field::from(value))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::aliases::hash_map::HashMap;
    use vortex_array::aliases::hash_set::HashSet;
    use vortex_array::stats::Stat;
    use vortex_dtype::Field;

    use crate::pruning::{
        convert_to_pruning_expression, stat_column_field, FieldOrIdentity, PruningPredicate,
    };
    use crate::{and, col, eq, gt, gt_eq, ident, lit, lt, lt_eq, not_eq, or, Not};

    #[test]
    pub fn pruning_equals() {
        let column = Field::from("a");
        let literal_eq = lit(42);
        let eq_expr = eq(col(column.clone()), literal_eq.clone());
        let (converted, refs) = convert_to_pruning_expression(&eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([(
                FieldOrIdentity::Field(column.clone()),
                HashSet::from_iter([Stat::Min, Stat::Max])
            )])
        );
        let expected_expr = or(
            gt(
                col(stat_column_field(&column, Stat::Min)),
                literal_eq.clone(),
            ),
            gt(literal_eq, col(stat_column_field(&column, Stat::Max))),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_equals_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let eq_expr = eq(col(column.clone()), col(other_col.clone()));

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
                col(stat_column_field(&column, Stat::Min)),
                col(stat_column_field(&other_col, Stat::Max)),
            ),
            gt(
                col(stat_column_field(&other_col, Stat::Min)),
                col(stat_column_field(&column, Stat::Max)),
            ),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_not_equals_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let not_eq_expr = not_eq(col(column.clone()), col(other_col.clone()));

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
            and(
                eq(
                    col(stat_column_field(&column, Stat::Min)),
                    col(stat_column_field(&column, Stat::Max)),
                ),
                eq(
                    col(stat_column_field(&other_col, Stat::Min)),
                    col(stat_column_field(&other_col, Stat::Max)),
                ),
            ),
            eq(
                col(stat_column_field(&column, Stat::Min)),
                col(stat_column_field(&other_col, Stat::Min)),
            ),
        );

        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_gt_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let other_expr = col(other_col.clone());
        let not_eq_expr = gt(col(column.clone()), other_expr.clone());

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
            col(stat_column_field(&column, Stat::Max)),
            col(stat_column_field(&other_col, Stat::Min)),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_gt_value() {
        let column = Field::from("a");
        let other_col = lit(42);
        let not_eq_expr = gt(col(column.clone()), other_col.clone());

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([(
                FieldOrIdentity::Field(column.clone()),
                HashSet::from_iter([Stat::Max])
            ),])
        );
        let expected_expr = lt_eq(
            col(stat_column_field(&column, Stat::Max)),
            other_col.clone(),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_lt_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let other_expr = col(other_col.clone());
        let not_eq_expr = lt(col(column.clone()), other_expr.clone());

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
            col(stat_column_field(&column, Stat::Min)),
            col(stat_column_field(&other_col, Stat::Max)),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    pub fn pruning_lt_value() {
        let column = Field::from("a");
        let other_col = lit(42);
        let not_eq_expr = lt(col(column.clone()), other_col.clone());

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs.into_map(),
            HashMap::from_iter([(
                FieldOrIdentity::Field(column.clone()),
                HashSet::from_iter([Stat::Min])
            )])
        );
        let expected_expr = gt_eq(
            col(stat_column_field(&column, Stat::Min)),
            other_col.clone(),
        );
        assert_eq!(&converted, &expected_expr);
    }

    #[test]
    fn unprojectable_expr() {
        let or_expr = Not::new_expr(lt(col(Field::from("a")), col(Field::from("b"))));
        assert!(PruningPredicate::try_new(&or_expr).is_none());
    }

    #[test]
    fn display_pruning_predicate() {
        let column = Field::from("a");
        let other_col = lit(42);
        let not_eq_expr = lt(col(column), other_col);

        assert_eq!(
            PruningPredicate::try_new(&not_eq_expr).unwrap().to_string(),
            "PruningPredicate(($a_min >= 42_i32), {$a: {min}})"
        );
    }

    #[test]
    fn or_required_stats_from_both_arms() {
        let column = col(Field::from("a"));
        let expr = or(lt(column.clone(), lit(10)), gt(column, lit(50)));

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
        let column = col(Field::from("a"));
        let expr = and(gt(column.clone(), lit(50)), lt(column, lit(10)));

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
        let column = ident();
        let expr = or(lt(column.clone(), lit(10)), gt(column.clone(), lit(50)));

        let expected = HashMap::from([(
            FieldOrIdentity::Identity,
            HashSet::from([Stat::Min, Stat::Max]),
        )]);

        let predicate = PruningPredicate::try_new(&expr).unwrap();
        assert_eq!(predicate.required_stats(), &expected);

        let expected_expr = or(
            gt_eq(col(Field::from("min")), lit(10)),
            lt_eq(col(Field::from("max")), lit(50)),
        );
        assert_eq!(predicate.expr(), &expected_expr)
    }
}
