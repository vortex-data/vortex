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

    /// Test rule: removes any literal "1" regardless of parent type (wildcard rule)
    struct RemoveOneLiteralRule;

    impl ParentReduceRule<Literal, AnyParent, RuleContext> for RemoveOneLiteralRule {
        fn reduce_parent(
            &self,
            expr: &ExpressionView<Literal>,
            parent: &Expression, // ← Untyped! AnyParent gives us &Expression
            child_idx: usize,
            _ctx: &RuleContext,
        ) -> VortexResult<Option<Expression>> {
            // Check if this literal is 1
            let one_scalar = Scalar::from(1i32);
            if expr.data() != &one_scalar {
                return Ok(None);
            }

            // Return the OTHER child from the parent (works for any binary parent)
            if parent.children().len() == 2 {
                let other_idx = if child_idx == 0 { 1 } else { 0 };
                return Ok(Some(parent.child(other_idx).clone()));
            }

            Ok(None)
        }
    }

    #[test]
    fn test_add_zero_parent_rule_basic() {
        // Create a session and register the rule (specific parent: Binary)
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, &Binary, AddZeroRule);

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
        session.register_parent_rule(&Literal, &Binary, AddZeroRule);

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
        session.register_parent_rule(&Literal, &Binary, AddZeroRule);

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
        session.register_parent_rule(&Literal, &Binary, AddZeroRule);

        // Test: (0 + x) + 0 should simplify to x
        let x = col("x");
        let zero = lit(0);
        let zero_plus_x = checked_add(lit(0), x.clone());
        let expr = checked_add(zero_plus_x, zero);

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }

    #[test]
    fn test_any_parent_wildcard_rule() {
        // Test AnyParent - rule works with ANY parent type
        let mut session = ExprSession::default();
        session.register_any_parent_rule(&Literal, RemoveOneLiteralRule);

        // Test: x + 1 should simplify to x (works with Add)
        let x = col("x");
        let one = lit(1);
        let expr = checked_add(x.clone(), one);

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &x);
    }

    #[test]
    fn test_specific_and_wildcard_rules_together() {
        // Test both specific and wildcard rules registered at the same time
        let mut session = ExprSession::default();

        // Specific rule: removes 0 from Add operations only
        session.register_parent_rule(&Literal, &Binary, AddZeroRule);

        // Wildcard rule: removes 1 from ANY operation
        session.register_any_parent_rule(&Literal, RemoveOneLiteralRule);

        // Test 1: 0 + x -> x (specific rule applies)
        let x = col("x");
        let zero = lit(0);
        let expr = checked_add(zero, x.clone());
        let result = simplify(expr, &session).unwrap();
        assert_eq!(&result, &x);

        // Test 2: 1 + y -> y (wildcard rule applies)
        let y = col("y");
        let one = lit(1);
        let expr = checked_add(one, y.clone());
        let result = simplify(expr, &session).unwrap();
        assert_eq!(&result, &y);
    }
}
