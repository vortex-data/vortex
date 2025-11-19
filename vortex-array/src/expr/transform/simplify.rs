// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::match_between::find_between;
use crate::expr::transform::rules::RuleContext;
use crate::expr::traversal::{NodeExt, Transformed};

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// This applies only untyped rewrite rules registered in the default session.
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub(crate) fn simplify(e: Expression, session: &ExprSession) -> VortexResult<Expression> {
    let ctx = RuleContext;

    let e = apply_parent_rules(e, &ctx, session)?;
    let e = apply_child_rules_impl(e, &ctx, session)?;
    let e = find_between(e);

    Ok(e)
}

fn apply_parent_rules(
    expr: Expression,
    ctx: &RuleContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    expr.transform_up(|node| {
        for (idx, child) in node.children().iter().enumerate() {
            for rule in session
                .rewrite_rules()
                .parent_rules_for(&child.id(), &node.id())
            {
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
    ctx: &RuleContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    fn rewrite(
        node: Expression,
        ctx: &RuleContext,
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
    use crate::expr::transform::rules::{AnyParent, ParentReduceRule, RuleContext};
    use crate::expr::{Expression, ExpressionView, col};

    /// Test rule: simplifies addition with zero: 0 + x -> x when literal zero is a child of an Add
    struct AddZeroRule;

    impl ParentReduceRule<Literal, Binary, RuleContext> for AddZeroRule {
        fn reduce_parent(
            &self,
            expr: &ExpressionView<Literal>,
            parent: ExpressionView<Binary>,
            child_idx: usize,
            _ctx: &RuleContext,
        ) -> VortexResult<Option<Expression>> {
            // Only apply if the parent is an Add operation
            if parent.operator() != Operator::Add {
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

    /// Test rule: remove identity 0 + x -> x without matching parent directly (equiv to above).
    struct AddZeroRuleAnyParent;

    impl ParentReduceRule<Literal, AnyParent, RuleContext> for AddZeroRuleAnyParent {
        fn reduce_parent(
            &self,
            expr: &ExpressionView<Literal>,
            parent: &Expression,
            child_idx: usize,
            _ctx: &RuleContext,
        ) -> VortexResult<Option<Expression>> {
            // Only apply if the parent is an Add operation
            let Some(parent) = parent.as_opt::<Binary>() else {
                return Ok(None);
            };
            if parent.operator() != Operator::Add {
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
    fn test_add_zero_with_specific_parent_rule() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, &Binary, AddZeroRule);

        let x = col("x");
        let zero = lit(0);
        let expr = checked_add(zero, x.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }

    #[test]
    fn test_add_zero_with_any_parent_rule() {
        let mut session = ExprSession::default();
        session.register_any_parent_rule(&Literal, AddZeroRuleAnyParent);

        let x = col("x");
        let zero = lit(0);
        let expr = checked_add(zero, x.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }

    #[test]
    fn test_add_zero_with_both_rules() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, &Binary, AddZeroRule);
        session.register_any_parent_rule(&Literal, AddZeroRuleAnyParent);

        let x = col("x");
        let zero = lit(0);
        let expr = checked_add(zero, x.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }
}
