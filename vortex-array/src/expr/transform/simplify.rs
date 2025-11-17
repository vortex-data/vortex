// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::{EmptyRewriteContext, RewriteContext};
use crate::expr::traversal::{NodeExt, Transformed};

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// This applies only untyped rewrite rules registered in the default session.
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub fn simplify(e: Expression, session: &ExprSession) -> VortexResult<Expression> {
    let ctx = EmptyRewriteContext;

    // First bottom-up (child rules)
    let e = e
        .transform_up(|node| apply_child_rules_to_node(node, &ctx, session))
        .map(|t| t.into_inner())?;

    let e = apply_parent_rules_impl(e, &ctx, session)?;

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
    expr.transform_up(|node| apply_child_rules_to_node(node, ctx, session))
        .map(|t| t.into_inner())
}

/// Apply child rules to a single node with RewriteContext.
fn apply_child_rules_to_node(
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

    // Apply untyped child-context rules
    if let Some(rules) = session.rewrite_rules().child_rules_for(&expr_id) {
        // Collect children to avoid borrowing issues
        let children: Vec<_> = current.children().iter().cloned().collect();
        // Try each child and each rule
        for (child_idx, child) in children.iter().enumerate() {
            for rule in rules {
                if let Some(new_expr) = rule.reduce_child_dyn(&current, child, child_idx, ctx)? {
                    current = new_expr;
                    changed = true;
                }
            }
        }
    }

    if changed {
        Ok(Transformed::yes(current))
    } else {
        Ok(Transformed::no(current))
    }
}
