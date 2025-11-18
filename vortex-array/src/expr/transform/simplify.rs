// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::match_between::find_between;
use crate::expr::transform::{EmptyRewriteContext, RewriteContext};
use crate::expr::traversal::{NodeExt, Transformed};

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// This applies only untyped rewrite rules registered in the default session.
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub fn simplify(e: Expression, session: &ExprSession) -> VortexResult<Expression> {
    let ctx = EmptyRewriteContext;

    let e = apply_parent_rules(e, &ctx, session)?;
    let e = apply_child_rules_impl(e, &ctx, session)?;
    let e = find_between(e);

    Ok(e)
}

fn apply_parent_rules(
    expr: Expression,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    expr.transform_up(|node| {
        for (idx, child) in node.children().iter().enumerate() {
            for rule in session.rewrite_rules().parent_rules_for(&child.id()) {
                if let Some(new_expr) = rule.reduce_parent(child, &node, idx, ctx)? {
                    return Ok(Transformed::yes(new_expr));
                }
            }
        }
        Ok(Transformed::no(node))
    })
    .map(|t| t.into_inner())
}

pub(crate) fn apply_child_rules_impl(
    expr: Expression,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    fn rewrite(
        node: Expression,
        ctx: &dyn RewriteContext,
        session: &ExprSession,
    ) -> VortexResult<Transformed<Expression>> {
        for rule in session.rewrite_rules().reduce_rules_for(&node.id()) {
            if let Some(new_expr) = rule.reduce(&node, ctx)? {
                return Ok(Transformed::yes(new_expr));
            }
        }
        Ok(Transformed::no(node))
    }
    expr.transform(
        |node| rewrite(node, ctx, session),
        |node| rewrite(node, ctx, session),
    )
    .map(|t| t.into_inner())
}

#[cfg(test)]
mod tests {
    use vortex_scalar::Scalar;

    use super::*;
    use crate::expr::exprs::binary::{Binary, checked_add};
    use crate::expr::exprs::literal::{Literal, lit};
    use crate::expr::exprs::operators::Operator;
    use crate::expr::session::ExprSession;
    use crate::expr::transform::rules::ParentReduceRule;
    use crate::expr::{Expression, ExpressionView, col};

    /// Test rule: simplifies addition with zero: 0 + x -> x when literal zero is a child of an Add
    struct AddZeroRule;

    impl ParentReduceRule<Literal, &dyn RewriteContext> for AddZeroRule {
        fn reduce_parent(
            &self,
            expr: &ExpressionView<Literal>,
            parent: &Expression,
            child_idx: usize,
            _ctx: &dyn RewriteContext,
        ) -> VortexResult<Option<Expression>> {
            // Only apply if the parent is an Add operation
            let Some(bin) = parent.as_opt::<Binary>() else {
                return Ok(None);
            };

            if bin.operator() != Operator::Add {
                return Ok(None);
            }

            // Check if this literal is zero
            let zero_scalar = Scalar::from(0i32);
            if expr.data() != &zero_scalar {
                return Ok(None);
            }

            // Return the other child (not this zero)
            let other_idx = if child_idx == 0 { 1 } else { 0 };
            Ok(Some(parent.child(other_idx).clone()))
        }
    }

    #[test]
    fn test_add_zero_parent_rule_basic() {
        // Create a session and register the rule
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: 0 + x should simplify to x
        let x = col("x");
        let zero = lit(0);
        let expr = checked_add(zero, x.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }

    #[test]
    fn test_add_zero_parent_rule_left() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: 0 + (0 + x) should simplify to 0 + x, then to x
        let x = col("x");
        let zero = lit(0);
        let zero_plus_x = checked_add(lit(0), x.clone());
        let expr = checked_add(zero, zero_plus_x);

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }

    #[test]
    fn test_add_zero_parent_rule_right() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: x + 0 should simplify to x
        let x = col("x");
        let zero = lit(0);
        let expr = checked_add(x.clone(), zero);

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }

    #[test]
    fn test_add_zero_parent_rule_nested() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: (0 + x) + 0 should simplify to x
        let x = col("x");
        let zero = lit(0);
        let zero_plus_x = checked_add(lit(0), x.clone());
        let expr = checked_add(zero_plus_x, zero);

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }
}
