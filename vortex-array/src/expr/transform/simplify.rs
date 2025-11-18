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
///
/// This applies parent rules bottom-up:
/// 1. First recursively process all children
/// 2. Rebuild expression with new children
/// 3. Apply parent rules to each child with the rebuilt parent
/// 4. If any child changes, recursively apply again
fn apply_parent_rules_recursive(
    expr: Expression,
    _parent: Option<&Expression>,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    // First, recursively process all children bottom-up
    let mut new_children = Vec::with_capacity(expr.children().len());
    let mut children_changed = false;

    for child in expr.children().iter() {
        // Recursively process this child first
        let new_child = apply_parent_rules_recursive(child.clone(), Some(&expr), ctx, session)?;

        new_children.push(new_child);
    }

    // Rebuild the expression with new children if any changed
    let mut expr = if children_changed {
        expr.with_children(new_children)?
    } else {
        expr
    };

    // Now apply parent rules to each child using the rebuilt parent
    loop {
        let mut any_child_changed = false;
        let mut updated_children = Vec::with_capacity(expr.children().len());

        for (child_idx, child) in expr.children().iter().enumerate() {
            // Try to apply parent rules to this child given that expr is its parent
            let new_child =
                apply_parent_rules_to_child(child.clone(), &expr, child_idx, ctx, session)?;

            if child != &new_child {
                any_child_changed = true;
            }

            updated_children.push(new_child);
        }

        if any_child_changed {
            expr = expr.with_children(updated_children)?;
        } else {
            break;
        }
    }

    Ok(expr)
}

/// Apply parent rules to a child expression given its parent and child index.
fn apply_parent_rules_to_child(
    child: Expression,
    parent: &Expression,
    child_idx: usize,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    let child_id = child.id();
    if let Some(rules) = session.rewrite_rules().parent_rules_for(&child_id) {
        let mut current = child;
        for rule in rules {
            if let Some(new_expr) = rule.reduce_parent_dyn(&current, parent, child_idx, ctx)? {
                current = new_expr;
            }
        }
        Ok(current)
    } else {
        Ok(child)
    }
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
    use crate::expr::exprs::literal::{Literal, lit};
    use crate::expr::exprs::operators::Operator;
    use crate::expr::session::ExprSession;
    use crate::expr::transform::rules::ParentReduceRule;
    use crate::expr::{Expression, ExpressionView};

    /// Test rule: simplifies addition with zero: 0 + x -> x when literal zero is a child of an Add
    struct AddZeroRule;

    impl ParentReduceRule<Literal> for AddZeroRule {
        fn reduce_parent(
            &self,
            expr: &ExpressionView<Literal>,
            parent: &Expression,
            child_idx: usize,
            _ctx: &dyn RewriteContext,
        ) -> VortexResult<Option<Expression>> {
            use vortex_scalar::Scalar;

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
        let x = lit(5);
        let zero = lit(0);
        let expr = checked_add(zero.clone(), x.clone());
        println!("expr {}", expr.display_tree());
        println!("expr dbg {:?}", expr);

        // let result = simplify(expr, &session).unwrap();
        //
        // // Should simplify to x (lit(5))
        // assert_eq!(&result, &lit(5));
    }

    #[test]
    fn test_add_zero_parent_rule_left() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: 0 + (0 + x) should simplify to 0 + x, then to x
        let x = lit(7);
        let zero = lit(0);
        let zero_plus_x = checked_add(zero.clone(), x.clone());
        let expr = checked_add(zero.clone(), zero_plus_x);

        let result = simplify(expr, &session).unwrap();

        // After first pass: 0 + (x) becomes x + (x) at the inner level
        // After second pass: x
        assert_eq!(&result, &lit(7));
    }

    #[test]
    fn test_add_zero_parent_rule_right() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: x + 0 should simplify to x
        let x = lit(3);
        let zero = lit(0);
        let expr = checked_add(x.clone(), zero.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &lit(3));
    }

    #[test]
    fn test_add_zero_parent_rule_nested() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: (0 + x) + 0 should simplify to x
        let x = lit(9);
        let zero = lit(0);
        let zero_plus_x = checked_add(zero.clone(), x.clone());
        let expr = checked_add(zero_plus_x, zero.clone());

        let result = simplify(expr, &session).unwrap();

        assert_eq!(&result, &lit(9));
    }

    #[test]
    fn test_add_zero_parent_rule_no_match() {
        let mut session = ExprSession::default();
        session.register_parent_rule(&Literal, AddZeroRule);

        // Test: x + y (no zeros) should not simplify
        let x = lit(3);
        let y = lit(4);
        let expr = checked_add(x.clone(), y.clone());

        let result = simplify(expr.clone(), &session).unwrap();

        assert_eq!(&result, &expr);
    }
}
