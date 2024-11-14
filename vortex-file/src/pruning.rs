// This code doesn't have usage outside of tests yet, remove once usage is added
#![allow(dead_code)]

use std::fmt::Display;

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::stats::Stat;
use vortex_dtype::field::Field;
use vortex_dtype::Nullability;
use vortex_expr::{BinaryExpr, Column, ExprRef, Literal, Not, Operator};
use vortex_scalar::Scalar;

#[derive(Debug, Clone)]
pub struct PruningPredicate {
    expr: ExprRef,
    required_stats: HashMap<Field, HashSet<Stat>>,
}

impl Display for PruningPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PruningPredicate({}, {{{}}})",
            self.expr,
            self.required_stats.iter().format_with(",", |(k, v), fmt| {
                fmt(&format_args!("{k}: {{{}}}", v.iter().format(",")))
            })
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
                .value()
                .as_bool()
                .ok()
                .flatten()
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

    pub fn required_stats(&self) -> &HashMap<Field, HashSet<Stat>> {
        &self.required_stats
    }
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
            return PruningPredicateRewriter::try_new(col.field().clone(), bexp.op(), bexp.rhs())
                .and_then(PruningPredicateRewriter::rewrite)
                .unwrap_or_else(|| {
                    (
                        Literal::new_expr(Scalar::bool(false, Nullability::NonNullable)),
                        HashMap::new(),
                    )
                });
        };

        if let Some(col) = bexp.rhs().as_any().downcast_ref::<Column>() {
            return PruningPredicateRewriter::try_new(
                col.field().clone(),
                bexp.op().swap(),
                bexp.lhs(),
            )
            .and_then(PruningPredicateRewriter::rewrite)
            .unwrap_or_else(|| {
                (
                    Literal::new_expr(Scalar::bool(false, Nullability::NonNullable)),
                    HashMap::new(),
                )
            });
        };
    }

    (
        Literal::new_expr(Scalar::bool(false, Nullability::NonNullable)),
        HashMap::new(),
    )
}

fn convert_column_reference(expr: &ExprRef, invert: bool) -> PruningPredicateStats {
    let mut refs = HashMap::new();
    let min_expr = replace_column_with_stat(expr, Stat::Min, &mut refs);
    let max_expr = replace_column_with_stat(expr, Stat::Max, &mut refs);
    (
        min_expr
            .zip(max_expr)
            .map(|(min_exp, max_exp)| {
                if invert {
                    BinaryExpr::new_expr(min_exp, Operator::And, max_exp)
                } else {
                    Not::new_expr(BinaryExpr::new_expr(min_exp, Operator::Or, max_exp))
                }
            })
            .unwrap_or_else(|| Literal::new_expr(Scalar::bool(false, Nullability::NonNullable))),
        refs,
    )
}

struct PruningPredicateRewriter<'a> {
    column: Field,
    operator: Operator,
    other_exp: &'a ExprRef,
    stats_to_fetch: HashMap<Field, HashSet<Stat>>,
}

type PruningPredicateStats = (ExprRef, HashMap<Field, HashSet<Stat>>);

impl<'a> PruningPredicateRewriter<'a> {
    pub fn try_new(column: Field, operator: Operator, other_exp: &'a ExprRef) -> Option<Self> {
        // TODO(robert): Simplify expression to guarantee that each column is not compared to itself
        //  For majority of cases self column references are likely not prunable
        if other_exp.references().contains(&column) {
            return None;
        }

        Some(Self {
            column,
            operator,
            other_exp,
            stats_to_fetch: HashMap::new(),
        })
    }

    fn add_stat_reference(&mut self, stat: Stat) -> Field {
        let new_field = stat_column_name(&self.column, stat);
        self.stats_to_fetch
            .entry(self.column.clone())
            .or_default()
            .insert(stat);
        new_field
    }

    fn rewrite_other_exp(&mut self, stat: Stat) -> ExprRef {
        replace_column_with_stat(self.other_exp, stat, &mut self.stats_to_fetch)
            .unwrap_or_else(|| self.other_exp.clone())
    }

    fn rewrite(mut self) -> Option<PruningPredicateStats> {
        let expr: Option<ExprRef> = match self.operator {
            Operator::Eq => {
                let min_col = Column::new_expr(self.add_stat_reference(Stat::Min));
                let max_col = Column::new_expr(self.add_stat_reference(Stat::Max));
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(BinaryExpr::new_expr(
                    BinaryExpr::new_expr(min_col, Operator::Gt, replaced_max),
                    Operator::Or,
                    BinaryExpr::new_expr(replaced_min, Operator::Gt, max_col),
                ))
            }
            Operator::NotEq => {
                let min_col = Column::new_expr(self.add_stat_reference(Stat::Min));
                let max_col = Column::new_expr(self.add_stat_reference(Stat::Max));
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                let column_value_is_single_known_value =
                    BinaryExpr::new_expr(min_col.clone(), Operator::Eq, max_col.clone());
                let column_value = min_col;

                let other_value_is_single_known_value =
                    BinaryExpr::new_expr(replaced_min.clone(), Operator::Eq, replaced_max.clone());
                let other_value = replaced_min;

                Some(BinaryExpr::new_expr(
                    BinaryExpr::new_expr(
                        column_value_is_single_known_value,
                        Operator::And,
                        other_value_is_single_known_value,
                    ),
                    Operator::And,
                    BinaryExpr::new_expr(column_value, Operator::Eq, other_value),
                ))
            }
            Operator::Gt | Operator::Gte => {
                let max_col = Column::new_expr(self.add_stat_reference(Stat::Max));
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(BinaryExpr::new_expr(max_col, Operator::Lte, replaced_min))
            }
            Operator::Lt | Operator::Lte => {
                let min_col = Column::new_expr(self.add_stat_reference(Stat::Min));
                let replaced_max = self.rewrite_other_exp(Stat::Max);

                Some(BinaryExpr::new_expr(min_col, Operator::Gte, replaced_max))
            }
            _ => None,
        };
        expr.map(|e| (e, self.stats_to_fetch))
    }
}

fn replace_column_with_stat(
    expr: &ExprRef,
    stat: Stat,
    stats_to_fetch: &mut HashMap<Field, HashSet<Stat>>,
) -> Option<ExprRef> {
    if let Some(col) = expr.as_any().downcast_ref::<Column>() {
        let new_field = stat_column_name(col.field(), stat);
        stats_to_fetch
            .entry(col.field().clone())
            .or_default()
            .insert(stat);
        return Some(Column::new_expr(new_field));
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

pub(crate) fn stat_column_name(field: &Field, stat: Stat) -> Field {
    match field {
        Field::Name(n) => Field::Name(format!("{n}_{stat}")),
        Field::Index(i) => Field::Name(format!("{i}_{stat}")),
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::aliases::hash_map::HashMap;
    use vortex_array::aliases::hash_set::HashSet;
    use vortex_array::stats::Stat;
    use vortex_dtype::field::Field;
    use vortex_expr::{BinaryExpr, Column, Literal, Not, Operator};

    use crate::pruning::{convert_to_pruning_expression, stat_column_name, PruningPredicate};

    #[test]
    pub fn pruning_equals() {
        let column = Field::from("a");
        let literal_eq = Literal::new_expr(42.into());
        let eq_expr = BinaryExpr::new_expr(
            Column::new_expr(column.clone()),
            Operator::Eq,
            literal_eq.clone(),
        );
        let (converted, refs) = convert_to_pruning_expression(&eq_expr);
        assert_eq!(
            refs,
            HashMap::from_iter([(column.clone(), HashSet::from_iter([Stat::Min, Stat::Max]))])
        );
        let expected_expr = BinaryExpr::new_expr(
            BinaryExpr::new_expr(
                Column::new_expr(stat_column_name(&column, Stat::Min)),
                Operator::Gt,
                literal_eq.clone(),
            ),
            Operator::Or,
            BinaryExpr::new_expr(
                literal_eq,
                Operator::Gt,
                Column::new_expr(stat_column_name(&column, Stat::Max)),
            ),
        );
        assert_eq!(*converted, *expected_expr.as_any());
    }

    #[test]
    pub fn pruning_equals_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let eq_expr = BinaryExpr::new_expr(
            Column::new_expr(column.clone()),
            Operator::Eq,
            Column::new_expr(other_col.clone()),
        );

        let (converted, refs) = convert_to_pruning_expression(&eq_expr);
        assert_eq!(
            refs,
            HashMap::from_iter([
                (column.clone(), HashSet::from_iter([Stat::Min, Stat::Max])),
                (
                    other_col.clone(),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = BinaryExpr::new_expr(
            BinaryExpr::new_expr(
                Column::new_expr(stat_column_name(&column, Stat::Min)),
                Operator::Gt,
                Column::new_expr(stat_column_name(&other_col, Stat::Max)),
            ),
            Operator::Or,
            BinaryExpr::new_expr(
                Column::new_expr(stat_column_name(&other_col, Stat::Min)),
                Operator::Gt,
                Column::new_expr(stat_column_name(&column, Stat::Max)),
            ),
        );
        assert_eq!(*converted, *expected_expr.as_any());
    }

    #[test]
    pub fn pruning_not_equals_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let not_eq_expr = BinaryExpr::new_expr(
            Column::new_expr(column.clone()),
            Operator::NotEq,
            Column::new_expr(other_col.clone()),
        );

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs,
            HashMap::from_iter([
                (column.clone(), HashSet::from_iter([Stat::Min, Stat::Max])),
                (
                    other_col.clone(),
                    HashSet::from_iter([Stat::Max, Stat::Min])
                )
            ])
        );
        let expected_expr = BinaryExpr::new_expr(
            BinaryExpr::new_expr(
                BinaryExpr::new_expr(
                    Column::new_expr(stat_column_name(&column, Stat::Min)),
                    Operator::Eq,
                    Column::new_expr(stat_column_name(&column, Stat::Max)),
                ),
                Operator::And,
                BinaryExpr::new_expr(
                    Column::new_expr(stat_column_name(&other_col, Stat::Min)),
                    Operator::Eq,
                    Column::new_expr(stat_column_name(&other_col, Stat::Max)),
                ),
            ),
            Operator::And,
            BinaryExpr::new_expr(
                Column::new_expr(stat_column_name(&column, Stat::Min)),
                Operator::Eq,
                Column::new_expr(stat_column_name(&other_col, Stat::Min)),
            ),
        );

        assert_eq!(*converted, *expected_expr.as_any());
    }

    #[test]
    pub fn pruning_gt_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let other_expr = Column::new_expr(other_col.clone());
        let not_eq_expr = BinaryExpr::new_expr(
            Column::new_expr(column.clone()),
            Operator::Gt,
            other_expr.clone(),
        );

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs,
            HashMap::from_iter([
                (column.clone(), HashSet::from_iter([Stat::Max])),
                (other_col.clone(), HashSet::from_iter([Stat::Min]))
            ])
        );
        let expected_expr = BinaryExpr::new_expr(
            Column::new_expr(stat_column_name(&column, Stat::Max)),
            Operator::Lte,
            Column::new_expr(stat_column_name(&other_col, Stat::Min)),
        );
        assert_eq!(*converted, *expected_expr.as_any());
    }

    #[test]
    pub fn pruning_gt_value() {
        let column = Field::from("a");
        let other_col = Literal::new_expr(42.into());
        let not_eq_expr = BinaryExpr::new_expr(
            Column::new_expr(column.clone()),
            Operator::Gt,
            other_col.clone(),
        );

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs,
            HashMap::from_iter([(column.clone(), HashSet::from_iter([Stat::Max])),])
        );
        let expected_expr = BinaryExpr::new_expr(
            Column::new_expr(stat_column_name(&column, Stat::Max)),
            Operator::Lte,
            other_col.clone(),
        );
        assert_eq!(*converted, *expected_expr.as_any());
    }

    #[test]
    pub fn pruning_lt_column() {
        let column = Field::from("a");
        let other_col = Field::from("b");
        let other_expr = Column::new_expr(other_col.clone());
        let not_eq_expr = BinaryExpr::new_expr(
            Column::new_expr(column.clone()),
            Operator::Lt,
            other_expr.clone(),
        );

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs,
            HashMap::from_iter([
                (column.clone(), HashSet::from_iter([Stat::Min])),
                (other_col.clone(), HashSet::from_iter([Stat::Max]))
            ])
        );
        let expected_expr = BinaryExpr::new_expr(
            Column::new_expr(stat_column_name(&column, Stat::Min)),
            Operator::Gte,
            Column::new_expr(stat_column_name(&other_col, Stat::Max)),
        );
        assert_eq!(*converted, *expected_expr.as_any());
    }

    #[test]
    pub fn pruning_lt_value() {
        let column = Field::from("a");
        let other_col = Literal::new_expr(42.into());
        let not_eq_expr = BinaryExpr::new_expr(
            Column::new_expr(column.clone()),
            Operator::Lt,
            other_col.clone(),
        );

        let (converted, refs) = convert_to_pruning_expression(&not_eq_expr);
        assert_eq!(
            refs,
            HashMap::from_iter([(column.clone(), HashSet::from_iter([Stat::Min]))])
        );
        let expected_expr = BinaryExpr::new_expr(
            Column::new_expr(stat_column_name(&column, Stat::Min)),
            Operator::Gte,
            other_col.clone(),
        );
        assert_eq!(*converted, *expected_expr.as_any());
    }

    #[test]
    fn unprojectable_expr() {
        let or_expr = Not::new_expr(BinaryExpr::new_expr(
            Column::new_expr(Field::from("a")),
            Operator::Lt,
            Column::new_expr(Field::from("b")),
        ));
        assert!(PruningPredicate::try_new(&or_expr).is_none());
    }

    #[test]
    fn display_pruning_predicate() {
        let column = Field::from("a");
        let other_col = Literal::new_expr(42.into());
        let not_eq_expr =
            BinaryExpr::new_expr(Column::new_expr(column.clone()), Operator::Lt, other_col);

        assert_eq!(
            PruningPredicate::try_new(&not_eq_expr).unwrap().to_string(),
            "PruningPredicate(($a_min >= 42_i32), {$a: {min}})"
        );
    }
}
