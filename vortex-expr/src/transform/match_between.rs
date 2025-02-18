use vortex_error::{VortexExpect, VortexResult};

use crate::between::Between;
use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{BinaryExpr, ExprRef, GetItem, Operator};

/// This pass looks for expression of the form
///      `x >= a && x < b` and converts them into x between a and b`
#[allow(dead_code)]
pub fn find_between(expr: ExprRef) -> ExprRef {
    let mut vis = MatchBetween;
    let res = expr
        .clone()
        .transform(&mut vis)
        .vortex_expect("cannot fail")
        .result;
    if !res.eq(&expr) {
        println!("expr {}", expr);
        println!("new res {}", res);
    };
    res
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

            // println!("here lhs {}, rhs {}", lhs, rhs);

            if lhs.lhs().as_any().is::<GetItem>()
                && lhs.lhs().eq(rhs.lhs())
                && is_between_operator_pair(lhs.op(), rhs.op())
            {
                let a = lhs.rhs().clone();
                let b = rhs.rhs().clone();
                let expr = Between::between(
                    lhs.lhs().clone(),
                    a,
                    lhs.op().inverse().unwrap(),
                    b,
                    rhs.op(),
                );
                return Ok(TransformResult::yes(expr));
            } else if lhs.lhs().as_any().is::<GetItem>()
                && lhs.lhs().eq(rhs.rhs())
                && is_between_operator_pair(lhs.op(), rhs.op().swap())
            {
                let a = lhs.rhs().clone();
                let b = rhs.lhs().clone();
                let expr = Between::between(
                    lhs.lhs().clone(),
                    a,
                    lhs.op().inverse().unwrap(),
                    b,
                    rhs.op().inverse().unwrap(),
                );
                return Ok(TransformResult::yes(expr));
            } else if lhs.rhs().as_any().is::<GetItem>()
                && lhs.rhs().eq(rhs.lhs())
                && is_between_operator_pair(lhs.op().swap(), rhs.op())
            {
                let a = lhs.lhs().clone();
                let b = rhs.rhs().clone();
                let expr = Between::between(
                    lhs.rhs().clone(),
                    a,
                    lhs.op().inverse().unwrap(),
                    b,
                    rhs.op(),
                );
                return Ok(TransformResult::yes(expr));
            } else if lhs.rhs().as_any().is::<GetItem>()
                && lhs.rhs().eq(rhs.rhs())
                && is_between_operator_pair(lhs.op().swap(), rhs.op().swap())
            {
                let a = lhs.lhs().clone();
                let b = rhs.lhs().clone();
                let expr = Between::between(lhs.rhs().clone(), a, lhs.op(), b, rhs.op());
                return Ok(TransformResult::yes(expr));
            }
        }
        Ok(TransformResult::no(node))
    }
}

fn is_between_operator_pair(lhs_op: Operator, rhs_op: Operator) -> bool {
    matches!(
        lhs_op,
        Operator::Gt | Operator::Gte | Operator::Lte | Operator::Lt
    ) && matches!(
        rhs_op,
        Operator::Gt | Operator::Gte | Operator::Lte | Operator::Lt
    )
}

#[cfg(test)]
mod tests {
    use crate::between::Between;
    use crate::transform::match_between::find_between;
    use crate::{and, col, gt_eq, lit, lt, Operator};

    #[test]
    fn test_match_between() {
        let expr = and(lt(lit(2), col("x")), lt(lit(5), col("x")));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &Between::between(col("x"), lit(2), Operator::Lt, lit(5), Operator::Lt),
            &find
        );
    }

    #[test]
    fn test_match_2_between() {
        let expr = and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5)));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &Between::between(col("x"), lit(2), Operator::Lt, lit(5), Operator::Lt),
            &find
        );
    }

    #[test]
    fn test_match_3_between() {
        let expr = and(gt_eq(col("x"), lit(2)), gt_eq(lit(5), col("x")));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &Between::between(col("x"), lit(2), Operator::Lt, lit(5), Operator::Lt),
            &find
        );
    }
}
