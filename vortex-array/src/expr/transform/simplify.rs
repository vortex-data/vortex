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
    let session = ExprSession::default();
    let ctx = EmptyRewriteContext;

    // First bottom-up (child rules)
    let e = e
        .transform_up(|node| apply_child_rules_to_node(node, &ctx, &session))
        .map(|t| t.into_inner())?;

    Ok(e)
}

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
