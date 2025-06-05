use vortex_array::stats::Stat;
use vortex_dtype::FieldName;
use vortex_error::VortexExpect;

use super::PruningPredicate;
use super::field_or_identity::FieldOrIdentity;
use super::relation::Relation;
use crate::between::Between;
use crate::{
    BinaryExpr, ExprRef, GetItem, Literal, Not, Operator, and, eq, get_item, gt, gt_eq, is_root,
    lit, lt, lt_eq, not, or, root,
};

#[derive(Debug, Clone, Default)]
pub(super) struct PruningPredicateBuilder {
    required_stats: Relation<FieldOrIdentity, Stat>,
}

impl PruningPredicateBuilder {
    pub fn build(mut self, expr: &ExprRef) -> Option<PruningPredicate> {
        self.falsification(expr).map(|expr| PruningPredicate {
            expr,
            required_stats: self.required_stats,
        })
    }

    /// An expression over zone-statistics which is true if-and-only-if `expr` is false for all
    /// records in the zone.
    ///
    /// # Examples
    ///
    /// - An expression over one variable: `x > 0` is false for all records in a zone if the maximum
    ///   value of the column `x` in that zone is less than or equal to zero: `max(x) <= 0`.
    /// - An expression over two variables: `x > y` becomes `max(x) <= min(y)`.
    /// - A conjunctive expression: `x > y AND z < x` becomes `max(x) <= min(y) OR min(z) >= max(x).
    ///
    /// Some expressions, in theory, have falsifications but this function does not support them
    /// such as `x < (y < z)` or `x LIKE "needle%"`.
    fn falsification(&mut self, expr: &ExprRef) -> Option<ExprRef> {
        if let Some(nexp) = expr.as_any().downcast_ref::<Not>() {
            if let Some(get_item) = nexp.child().as_any().downcast_ref::<GetItem>() {
                if is_root(get_item.child()) {
                    return self.convert_access_reference(expr, true);
                }
            }
        }

        if let Some(get_item) = expr.as_any().downcast_ref::<GetItem>() {
            if is_root(get_item.child()) {
                return self.convert_access_reference(expr, false);
            }
        }

        if let Some(bexp) = expr.as_any().downcast_ref::<BinaryExpr>() {
            // FIXME(DK): if one argument succeeds but the other fails we'll have statistics we
            // don't really need in our required_stats.
            let lhs = bexp.lhs();
            let rhs = bexp.rhs();
            return Some(match bexp.op() {
                Operator::Eq => {
                    // We can disprove lhs == rhs when either:
                    // 1. min(lhs) > max(rhs)
                    // 2. min(rhs) > max(lhs)
                    or(
                        gt(self.min(lhs)?, self.max(rhs)?),
                        gt(self.min(rhs)?, self.max(lhs)?),
                    )
                }
                Operator::NotEq => {
                    // We can disprove lhs != rhs only when,
                    //
                    //     min(lhs) == max(lhs) == min(rhs) == max(rhs)
                    //
                    // which implies that each expression is a constant and, in fact, they're the same
                    // constant.
                    and(
                        eq(self.min(lhs)?, self.max(rhs)?),
                        eq(self.max(lhs)?, self.min(rhs)?),
                    )
                }
                Operator::Gt => {
                    // We can disprove lhs > rhs when max(lhs) <= min(rhs).
                    lt_eq(self.max(lhs)?, self.min(rhs)?)
                }
                Operator::Gte => {
                    // We can disprove lhs >= rhs when max(lhs) < min(rhs).
                    lt(self.max(lhs)?, self.min(rhs)?)
                }
                Operator::Lt => {
                    // We can disprove lhs < rhs when min(lhs) >= max(rhs).
                    gt_eq(self.min(lhs)?, self.max(rhs)?)
                }
                Operator::Lte => {
                    // We can disprove lhs <= rhs when min(lhs) > max(rhs).
                    gt(self.min(lhs)?, self.max(rhs)?)
                }
                Operator::And | Operator::Or => {
                    let rewritten_left = self.falsification(lhs).unwrap_or_else(|| lit(false));
                    let rewritten_right = self.falsification(rhs).unwrap_or_else(|| lit(false));
                    let flipped_op = bexp
                        .op()
                        .logical_inverse()
                        .vortex_expect("Cannot be any other operator than and / or");
                    BinaryExpr::new_expr(rewritten_left, flipped_op, rewritten_right)
                }
            });
        }

        if let Some(between_expr) = expr.as_any().downcast_ref::<Between>() {
            return self.falsification(&between_expr.to_binary_expr());
        }

        None
    }

    // FIXME(DK): I think this function can be simpler. It seems to assume that expr is a
    // boolean-valued expression and is checking that it is constant true or constant false.
    fn convert_access_reference(&mut self, expr: &ExprRef, invert: bool) -> Option<ExprRef> {
        let min = self.min(expr)?;
        let max = self.max(expr)?;

        let expr = if invert {
            and(min, max)
        } else {
            not(or(min, max))
        };

        Some(expr)
    }

    // FIXME(DK): this should take a ref to FieldOrIdentity which itself should be a struct of ref
    fn use_field_stat(&mut self, field: FieldName, stat: Stat) -> ExprRef {
        self.use_stat(FieldOrIdentity::Field(field), stat)
    }

    // FIXME(DK): this should take a ref to FieldOrIdentity which itself should be a struct of ref
    fn use_stat(&mut self, field: FieldOrIdentity, stat: Stat) -> ExprRef {
        self.required_stats.insert(field.clone(), stat);
        get_item(field.stat_field_name(stat), root())
    }

    /// If an expression is returned, its value is an upper bound on the value of `expr`.
    ///
    /// We may return `None` for values which have no upper bound or values for which knowing the
    /// upper bound is difficult.
    fn max(&mut self, expr: &ExprRef) -> Option<ExprRef> {
        if expr.as_any().is::<Literal>() {
            return Some(expr.clone());
        }

        if is_root(expr) {
            return Some(self.use_stat(FieldOrIdentity::Identity, Stat::Max));
        }

        if let Some(get_item) = expr.as_any().downcast_ref::<GetItem>() {
            if is_root(get_item.child()) {
                return Some(self.use_field_stat(get_item.field().clone(), Stat::Max));
            }
        }

        // TODO(DK): max(not(x)) is min(x)
        // TODO(DK): max(x > y) is max(x) > min(y)
        // TODO(DK): max(x || y) is max(x) || max(y)

        None
    }

    /// If an expression is returned, its value is a lower bound on the value of `expr`.
    ///
    /// We may return `None` for values which have no lower bound or values for which knowing the
    /// lower bound is difficult.
    fn min(&mut self, expr: &ExprRef) -> Option<ExprRef> {
        if expr.as_any().is::<Literal>() {
            return Some(expr.clone());
        }

        if is_root(expr) {
            return Some(self.use_stat(FieldOrIdentity::Identity, Stat::Min));
        }

        if let Some(get_item) = expr.as_any().downcast_ref::<GetItem>() {
            if is_root(get_item.child()) {
                return Some(self.use_field_stat(get_item.field().clone(), Stat::Min));
            }
        }

        // TODO(DK): min(not(x)) is max(x)
        // TODO(DK): min(x > y) is min(x) > max(y)
        // TODO(DK): min(x && y) is min(x) && min(y)

        None
    }
}
