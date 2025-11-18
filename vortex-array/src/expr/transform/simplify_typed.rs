// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::{SimpleRewriteContext, TypedRewriteContext};
use crate::expr::traversal::{NodeExt, Transformed};

/// Unlike `simplify`, this function simplifies an expression under the assumption that scope is
/// a known DType. Simplification is applied first and then additional dtype-aware rules.
///
/// NOTE: After typed simplification, returned expressions is "bound" to the scope DType.
///     Applying the returned expression to a different DType may produce wrong results.
pub fn simplify_typed(
    expr: Expression,
    dtype: &DType,
    session: &ExprSession,
) -> VortexResult<Expression> {
    let ctx = SimpleRewriteContext { dtype };

    // First bottom-up (child rules)
    let expr = apply_child_rules_impl_typed(expr, &ctx, session)?;

    // Then top-down (parent rules)
    apply_parent_rules_impl_typed(expr, &ctx, session)
}

/// Internal implementation: Apply child rules in a bottom-up manner with TypedRewriteContext.
fn apply_child_rules_impl_typed(
    expr: Expression,
    ctx: &dyn TypedRewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    expr.transform_up(|node| apply_child_rules_to_node_typed(node, ctx, session))
        .map(|t| t.into_inner())
}

/// Apply child rules to a single node with TypedRewriteContext.
fn apply_child_rules_to_node_typed(
    expr: Expression,
    ctx: &dyn TypedRewriteContext,
    session: &ExprSession,
) -> VortexResult<Transformed<Expression>> {
    let expr_id = expr.id();
    let mut current = expr;
    let mut changed = false;

    // Apply typed generic reduce rules
    if let Some(rules) = session.rewrite_rules().typed_reduce_rules_for(&expr_id) {
        for rule in rules {
            if let Some(new_expr) = rule.reduce_dyn_typed(&current, ctx)? {
                current = new_expr;
                changed = true;
            }
        }
    }

    // Apply untyped reduce rules (TypedRewriteContext extends RewriteContext)
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

/// Internal implementation: Apply parent rules in a top-down manner with TypedRewriteContext.
fn apply_parent_rules_impl_typed(
    expr: Expression,
    ctx: &dyn TypedRewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    apply_parent_rules_recursive_typed(expr, None, ctx, session)
}

/// Recursive helper for applying parent rules with TypedRewriteContext.
fn apply_parent_rules_recursive_typed(
    expr: Expression,
    parent: Option<&Expression>,
    ctx: &dyn TypedRewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    // Apply parent rules if we have a parent
    // let expr = if let Some(parent) = parent {
    //     let expr_id = expr.id();
    //     let mut current = expr;
    //
    //     let rules = session
    //         .rewrite_rules()
    //         .parent_rules_for(&expr_id)
    //         .unwrap_or_default();
    //     for rule in rules {
    //         // TypedRewriteContext extends RewriteContext, so we can pass it to reduce_parent_dyn
    //         if let Some(new_expr) = rule.reduce_parent_dyn(&current, parent, ctx)? {
    //             current = new_expr;
    //         }
    //     }
    //     current
    // } else {
    //     expr
    // };
    //
    // // Recursively apply to children
    // let new_children: Result<Vec<_>, _> = expr
    //     .children()
    //     .iter()
    //     .map(|child| apply_parent_rules_recursive_typed(child.clone(), Some(&expr), ctx, session))
    //     .collect();
    //
    // expr.with_children(new_children?)
    todo!()
}
