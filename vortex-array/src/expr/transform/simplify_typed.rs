// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::{RootRewriteContext, TypedRewriteContext};
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
    let ctx = RootRewriteContext { dtype };

    let expr = apply_parent_rules_impl_typed(expr, &ctx, session)?;
    let expr = apply_child_rules_impl_typed(expr, &ctx, session)?;

    Ok(expr)
}

fn apply_child_rules_impl_typed(
    expr: Expression,
    ctx: &dyn TypedRewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    fn rewrite(
        node: Expression,
        ctx: &dyn TypedRewriteContext,
        session: &ExprSession,
    ) -> VortexResult<Transformed<Expression>> {
        for rule in session.rewrite_rules().typed_reduce_rules_for(&node.id()) {
            if let Some(new_expr) = rule.reduce_dyn_typed(&node, ctx)? {
                return Ok(Transformed::yes(new_expr));
            }
        }
        for rule in session.rewrite_rules().reduce_rules_for(&node.id()) {
            if let Some(new_expr) = rule.reduce_dyn(&node, ctx)? {
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

fn apply_parent_rules_impl_typed(
    expr: Expression,
    ctx: &dyn TypedRewriteContext,
    session: &ExprSession,
) -> VortexResult<Expression> {
    expr.transform_up(|node| {
        for (idx, child) in node.children().iter().enumerate() {
            for rule in session.rewrite_rules().typed_parent_rules_for(&child.id()) {
                if let Some(new_expr) = rule.reduce_parent_dyn_typed(child, &node, idx, ctx)? {
                    return Ok(Transformed::yes(new_expr));
                }
            }
            for rule in session.rewrite_rules().parent_rules_for(&child.id()) {
                if let Some(new_expr) = rule.reduce_parent_dyn(child, &node, idx, ctx)? {
                    return Ok(Transformed::yes(new_expr));
                }
            }
        }
        Ok(Transformed::no(node))
    })
    .map(|t| t.into_inner())
}
