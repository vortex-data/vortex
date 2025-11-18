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

    // First bottom-up (child rules)
    let e = apply_child_rules_impl(e, &ctx, session)?;

    let e = apply_parent_rules_impl(e, &ctx, session)?;

    let e = find_between(e);

    Ok(e)
}

/// Internal implementation: Apply parent rules in a top-down manner.
pub(crate) fn apply_parent_rules_impl(
    expr: Expression,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    apply_parent_rules_recursive(expr, None, ctx, session)
}

/// Recursive helper for applying parent rules.
fn apply_parent_rules_recursive(
    expr: Expression,
    parent: Option<&Expression>,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    // Apply parent rules if we have a parent
    let expr = if let Some(parent) = parent {
        let expr_id = expr.id();
        if let Some(rules) = session.rewrite_rules().parent_rules_for(&expr_id) {
            let mut current = expr;
            for rule in rules {
                if let Some(new_expr) = rule.reduce_parent_dyn(&current, parent, ctx)? {
                    current = new_expr;
                }
            }
            current
        } else {
            expr
        }
    } else {
        expr
    };

    // Recursively apply to children
    let new_children: Result<Vec<_>, _> = expr
        .children()
        .iter()
        .map(|child| apply_parent_rules_recursive(child.clone(), Some(&expr), ctx, session))
        .collect();

    expr.with_children(new_children?)
}

/// Internal implementation: Apply child rules in a bottom-up manner with RewriteContext.
pub(crate) fn apply_child_rules_impl(
    expr: Expression,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    expr.transform_up(|node| apply_reduce_rules_node(node, ctx, session))
        .map(|t| t.into_inner())
}

/// Apply child rules to a single node with RewriteContext.
fn apply_reduce_rules_node(
    expr: Expression,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Transformed<Expression>> {
    let expr_id = expr.id();
    let mut current = expr;
    let mut changed = false;

    // Apply untyped generic reduce rules
    if let Some(rules) = session.rewrite_rules().reduce_rules_for(&expr_id) {
        for rule in rules {
            if let Some(new_expr) = rule.reduce_dyn(&current, ctx)? {
                current = new_expr;
                changed = true;
            }
        }
    }

    if changed {
        Ok(Transformed::yes(current))
    } else {
        Ok(Transformed::no(current))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::exprs::binary::{Binary, checked_add};
    use crate::expr::exprs::literal::lit;
    use crate::expr::session::ExprSession;
    use crate::expr::transform::rules::ParentReduceRule;
    use crate::expr::{Expression, ExpressionView, Literal};

    /// Test rule: simplifies addition with zero: 0 + x -> x
    struct AddZeroRule;

    impl ParentReduceRule<Literal> for AddZeroRule {
        fn reduce_parent(
            &self,
            expr: &ExpressionView<Literal>,
            parent: &Expression,
            child_idx: usize,
            _ctx: &dyn RewriteContext,
        ) -> VortexResult<Option<Expression>> {
            // Only apply if the parent is also an Add operation
            let Some(bin) = parent.as_opt::<Binary>() else {
                Ok(None)
            };
            assert!(child_idx <= 1);
            Ok(Some(parent.child((child_idx == 0) as usize).clone()))
        }
    }

    #[test]
    fn test_add_zero_parent_rule_basic() {
        // Create a session and register the rule
        let mut session = ExprSession::default();
        session
            .rewrite_rules_mut()
            .register_parent_rule(&Binary, AddZeroRule);

        // Test: (0 + x) + 0 should simplify to x
        let x = lit(5);
        let zero = lit(0);
        let zero_plus_x = checked_add(zero.clone(), x.clone());
        let expr = checked_add(zero_plus_x, zero.clone());

        let result = simplify(expr, &session).unwrap();

        // Should simplify to x (lit(5))
        assert_eq!(&result, &lit(5));
    }

    #[test]
    fn test_add_zero_parent_rule_left() {
        let mut session = ExprSession::default();
        session
            .rewrite_rules_mut()
            .register_parent_rule(&Binary, AddZeroRule);

        // Test: 0 + (0 + x) should simplify to x
        let x = lit(7);
        let zero = lit(0);
        let zero_plus_x = checked_add(zero.clone(), x.clone());
        let expr = checked_add(zero.clone(), zero_plus_x);

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &lit(7));
    }

    #[test]
    fn test_add_zero_parent_rule_right() {
        let mut session = ExprSession::default();
        session
            .rewrite_rules_mut()
            .register_parent_rule(&Binary, AddZeroRule);

        // Test: (x + 0) + 0 should simplify to x
        let x = lit(3);
        let zero = lit(0);
        let x_plus_zero = checked_add(x.clone(), zero.clone());
        let expr = checked_add(x_plus_zero, zero.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &lit(3));
    }

    #[test]
    fn test_add_zero_parent_rule_nested_left() {
        let mut session = ExprSession::default();
        session
            .rewrite_rules_mut()
            .register_parent_rule(&Binary, AddZeroRule);

        // Test: ((0 + x) + 0) + 0 should simplify to x
        let x = lit(9);
        let zero = lit(0);
        let zero_plus_x = checked_add(zero.clone(), x.clone());
        let level1 = checked_add(zero_plus_x, zero.clone());
        let expr = checked_add(level1, zero.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &lit(9));
    }

    #[test]
    fn test_add_zero_parent_rule_no_match() {
        let mut session = ExprSession::default();
        session
            .rewrite_rules_mut()
            .register_parent_rule(&Binary, AddZeroRule);

        // Test: x + y (no zeros) should not simplify
        let x = lit(3);
        let y = lit(4);
        let expr = checked_add(x.clone(), y.clone());

        let result = simplify(expr.clone(), &session).unwrap();

        assert_eq!(&result, &expr);
    }
}
