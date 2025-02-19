use vortex_error::{VortexExpect, VortexResult};

use crate::between::Between;
use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{BinaryExpr, ExprRef, GetItem, Literal, Operator};

/// This pass looks for expression of the form
///      `x >= a && x < b` and converts them into x between a and b`
#[allow(dead_code)]
pub fn find_between(expr: ExprRef) -> ExprRef {
    expr.clone()
        .transform(&mut MatchBetween)
        .vortex_expect("cannot fail")
        .result
}

struct MatchBetween;

impl MutNodeVisitor for MatchBetween {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(and) = node.as_any().downcast_ref::<BinaryExpr>() {
            if and.op() != Operator::And {
                return Ok(TransformResult::no(node));
            }
            let (Some(lhs), Some(rhs)) = (
                and.lhs().as_any().downcast_ref::<BinaryExpr>(),
                and.rhs().as_any().downcast_ref::<BinaryExpr>(),
            ) else {
                return Ok(TransformResult::no(node));
            };

            // Cannot compare to self
            if lhs.lhs().eq(lhs.rhs()) || rhs.lhs().eq(rhs.rhs()) {
                return Ok(TransformResult::no(node));
            }

            // Extract pairs of comparison of the form (left left_op eq) and (eq right_op right)
            let (eq, left, left_op, right, right_op) =
                if GetItem::is(lhs.lhs()) && lhs.lhs().eq(rhs.lhs()) {
                    (
                        lhs.lhs().clone(),
                        lhs.rhs().clone(),
                        lhs.op().swap(),
                        rhs.rhs().clone(),
                        rhs.op(),
                    )
                } else if GetItem::is(lhs.lhs()) && lhs.lhs().eq(rhs.rhs()) {
                    (
                        lhs.lhs().clone(),
                        lhs.rhs().clone(),
                        lhs.op().swap(),
                        rhs.lhs().clone(),
                        rhs.op().swap(),
                    )
                } else if GetItem::is(lhs.rhs()) && lhs.rhs().eq(rhs.lhs()) {
                    (
                        lhs.rhs().clone(),
                        lhs.lhs().clone(),
                        lhs.op(),
                        rhs.rhs().clone(),
                        rhs.op(),
                    )
                } else if GetItem::is(lhs.rhs()) && lhs.rhs().eq(rhs.rhs()) {
                    (
                        lhs.rhs().clone(),
                        lhs.lhs().clone(),
                        lhs.op(),
                        rhs.lhs().clone(),
                        rhs.op().swap(),
                    )
                } else {
                    return Ok(TransformResult::no(node));
                };

            // Find the greater op.
            let (Some(left_lit), Some(right_lit)) =
                (Literal::maybe_from(&left), Literal::maybe_from(&right))
            else {
                return Ok(TransformResult::no(node));
            };

            let (left, left_op, right, right_op) = if left_lit.value() > right_lit.value() {
                (right, right_op, left, left_op)
            } else {
                (left, left_op, right, right_op)
            };

            // Check if the operators form an inequality.
            if is_between_operator_pair(left_op, right_op) {
                let expr = Between::between(eq.clone(), left, left_op, right, right_op);
                return Ok(TransformResult::yes(expr));
            } else if is_between_operator_pair(left_op.swap(), right_op.swap()) {
                let expr =
                    Between::between(eq.clone(), left, left_op.swap(), right, right_op.swap());
                return Ok(TransformResult::yes(expr));
            }
        }
        Ok(TransformResult::no(node))
    }
}

fn is_between_operator_pair(lhs_op: Operator, rhs_op: Operator) -> bool {
    matches!(lhs_op, Operator::Lte | Operator::Lt) && matches!(rhs_op, Operator::Lte | Operator::Lt)
}

#[cfg(test)]
mod tests {
    use crate::between::Between;
    use crate::transform::match_between::find_between;
    use crate::{and, col, gt_eq, lit, lt, Operator};

    #[test]
    fn test_match_between() {
        let expr = and(lt(lit(2), col("x")), gt_eq(lit(5), col("x")));
        let find = find_between(expr);

        // 2 < x <= 5
        assert_eq!(
            &Between::between(col("x"), lit(2), Operator::Lt, lit(5), Operator::Lte),
            &find
        );
    }

    #[test]
    fn test_match_2_between() {
        let expr = and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5)));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &Between::between(col("x"), lit(2), Operator::Lte, lit(5), Operator::Lt),
            &find
        );
    }

    #[test]
    fn test_match_3_between() {
        let expr = and(gt_eq(col("x"), lit(2)), gt_eq(lit(5), col("x")));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &Between::between(col("x"), lit(2), Operator::Lte, lit(5), Operator::Lte),
            &find
        );
    }

    #[test]
    fn test_match_4_between() {
        let expr = and(gt_eq(lit(5), col("x")), lt(lit(2), col("x")));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &Between::between(col("x"), lit(2), Operator::Lt, lit(5), Operator::Lte),
            &find
        );
    }
}
