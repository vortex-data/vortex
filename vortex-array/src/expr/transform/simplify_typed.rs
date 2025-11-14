// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::remove_merge::remove_merge;
use crate::expr::transform::remove_select::remove_select;
use crate::expr::transform::simplify::simplify;
use crate::expr::transform::traits::{RewriteContext, SimpleRewriteContext};
use crate::expr::traversal::{NodeExt, Transformed};

/// Unlike `simplify`, this function simplifies an expression under the assumption that scope is
/// a known DType. Simplified is applied first and then additional rules.
///
/// NOTE: After typed simplification, returned expressions is "bound" to the scope DType.
///     Applying the returned expression to a different DType may produce wrong results.
pub fn simplify_typed(e: Expression, ctx: &DType) -> VortexResult<Expression> {
    let e = simplify(e)?;

    let e = remove_select(e, ctx)?;
    let e = remove_merge(e, ctx)?;
    let e = simplify(e)?;

    Ok(e)
}

/// Simplify an expression using registered rewrite rules from a session.
///
/// This applies all rules registered in the session's `RewriteRuleRegistry`:
/// 1. Child rules (bottom-up traversal)
/// 2. Parent rules (top-down traversal)
///
/// This is the primary entry point for extensible expression optimization.
pub fn simplify_with_session(
    expr: Expression,
    dtype: &DType,
    session: &ExprSession,
) -> VortexResult<Expression> {
    let ctx = SimpleRewriteContext { dtype };

    // First bottom-up (child rules)
    let expr = apply_child_rules_impl(expr, &ctx, session)?;

    // Then top-down (parent rules)
    apply_parent_rules_impl(expr, &ctx, session)
}

/// Apply only child reduction rules from the session (bottom-up traversal).
///
/// This performs a post-order traversal, applying rules after children are processed.
/// Useful for optimizations like `pack(...).get_item(field) -> field_expr`.
pub fn apply_child_rules(
    expr: Expression,
    dtype: &DType,
    session: &ExprSession,
) -> VortexResult<Expression> {
    let ctx = SimpleRewriteContext { dtype };
    apply_child_rules_impl(expr, &ctx, session)
}

/// Apply only parent reduction rules from the session (top-down traversal).
///
/// This performs a pre-order traversal, applying rules based on parent context.
/// Only called for non-root expressions.
pub fn apply_parent_rules(
    expr: Expression,
    dtype: &DType,
    session: &ExprSession,
) -> VortexResult<Expression> {
    let ctx = SimpleRewriteContext { dtype };
    apply_parent_rules_impl(expr, &ctx, session)
}

/// Internal implementation: Apply child rules in a bottom-up manner.
fn apply_child_rules_impl(
    expr: Expression,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    expr.transform_up(|node| apply_child_rules_to_node(node, ctx, session))
        .map(|t| t.into_inner())
}

/// Apply child rules to a single node.
fn apply_child_rules_to_node(
    expr: Expression,
    ctx: &dyn RewriteContext,
    session: &ExprSession,
) -> VortexResult<Transformed<Expression>> {
    let expr_id = expr.id();

    // First try generic reduce rules (no context needed)
    if let Some(rules) = session.rewrite_rules().reduce_rules_for(&expr_id) {
        for rule in rules {
            if let Some(new_expr) = rule.reduce_dyn(&expr, ctx)? {
                return Ok(Transformed::yes(new_expr));
            }
        }
    }

    // Then try child-context rules
    if let Some(rules) = session.rewrite_rules().child_rules_for(&expr_id) {
        // Try each child and each rule
        for (child_idx, child) in expr.children().iter().enumerate() {
            for rule in rules {
                if let Some(new_expr) = rule.reduce_child_dyn(&expr, child, child_idx, ctx)? {
                    return Ok(Transformed::yes(new_expr));
                }
            }
        }
    }

    Ok(Transformed::no(expr))
}

/// Internal implementation: Apply parent rules in a top-down manner.
fn apply_parent_rules_impl(
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
