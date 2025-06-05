use vortex_array::stats::Stat;
use vortex_dtype::{FieldName, Nullability};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use super::field_or_identity::{FieldOrIdentity, stat_field_name};
use super::relation::Relation;
use crate::between::Between;
use crate::{
    BinaryExpr, ExprRef, GetItem, Not, Operator, VortexExprExt, and, eq, get_item, gt, is_root,
    lit, not, or, root,
};

type PruningPredicateStats = (ExprRef, Relation<FieldOrIdentity, Stat>);

pub(super) fn not_prunable() -> PruningPredicateStats {
    (
        lit(Scalar::bool(false, Nullability::NonNullable)),
        Relation::new(),
    )
}

// Anything that can't be translated has to be represented as
// boolean true expression, i.e. the value might be in that chunk
pub(super) fn convert_to_pruning_expression(expr: &ExprRef) -> PruningPredicateStats {
    if let Some(nexp) = expr.as_any().downcast_ref::<Not>() {
        if let Some(get_item) = nexp.child().as_any().downcast_ref::<GetItem>() {
            if is_root(get_item.child()) {
                return convert_access_reference(expr, true);
            }
        }
    }

    if let Some(get_item) = expr.as_any().downcast_ref::<GetItem>() {
        if is_root(get_item.child()) {
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
                .vortex_expect("Cannot be any other operator than and / or");
            return (
                BinaryExpr::new_expr(rewritten_left, flipped_op, rewritten_right),
                refs_lhs,
            );
        }

        if let Some(get_item) = bexp.lhs().as_any().downcast_ref::<GetItem>() {
            if is_root(get_item.child()) {
                return PruningPredicateRewriter::rewrite_binary_op(
                    FieldOrIdentity::Field(get_item.field().clone()),
                    bexp.op(),
                    bexp.rhs(),
                );
            }
        };

        if let Some(get_item) = bexp.rhs().as_any().downcast_ref::<GetItem>() {
            if is_root(get_item.child()) {
                return PruningPredicateRewriter::rewrite_binary_op(
                    FieldOrIdentity::Field(get_item.field().clone()),
                    bexp.op().swap(),
                    bexp.lhs(),
                );
            }
        }

        if is_root(bexp.lhs()) {
            return PruningPredicateRewriter::rewrite_binary_op(
                FieldOrIdentity::Identity,
                bexp.op(),
                bexp.rhs(),
            );
        };

        if is_root(bexp.rhs()) {
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
                let min_col = get_item(self.add_stat_reference(Stat::Min), root());
                let max_col = get_item(self.add_stat_reference(Stat::Max), root());
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(or(gt(min_col, replaced_max), gt(replaced_min, max_col)))
            }
            Operator::NotEq => {
                let min_col = get_item(self.add_stat_reference(Stat::Min), root());
                let max_col = get_item(self.add_stat_reference(Stat::Max), root());
                let replaced_max = self.rewrite_other_exp(Stat::Max);
                let replaced_min = self.rewrite_other_exp(Stat::Min);

                Some(and(eq(min_col, replaced_max), eq(max_col, replaced_min)))
            }
            Operator::Gt | Operator::Gte => {
                let max_col = get_item(self.add_stat_reference(Stat::Max), root());
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
                let min_col = get_item(self.add_stat_reference(Stat::Min), root());
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
        if is_root(get_i.child()) {
            let new_field = stat_field_name(get_i.field(), stat);
            stats_to_fetch.insert(FieldOrIdentity::Field(get_i.field().clone()), stat);
            return Some(get_item(new_field, root()));
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
