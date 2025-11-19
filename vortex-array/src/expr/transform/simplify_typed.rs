// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::RewriteRuleRegistry;
use crate::expr::transform::RuleContext;
use crate::expr::transform::rules::TypedRuleContext;
use crate::expr::traversal::{NodeExt, Transformed};

/// Unlike `simplify`, this function simplifies an expression under the assumption that scope is
/// a known DType. Simplification is applied first and then additional dtype-aware rules.
///
/// NOTE: After typed simplification, returned expressions is "bound" to the scope DType.
///     Applying the returned expression to a different DType may produce wrong results.
pub(super) fn simplify_typed(
    expr: Expression,
    dtype: &DType,
    rule_registry: &RewriteRuleRegistry,
) -> VortexResult<Expression> {
    let ctx = TypedRuleContext::new(dtype.clone());

    let expr = apply_parent_rules_impl_typed(expr, &ctx, rule_registry)?;
    let expr = apply_child_rules_impl_typed(expr, &ctx, rule_registry)?;

    Ok(expr)
}

fn apply_child_rules_impl_typed(
    expr: Expression,
    ctx: &TypedRuleContext,
    rule_registry: &RewriteRuleRegistry,
) -> VortexResult<Expression> {
    fn rewrite(
        node: Expression,
        ctx: &TypedRuleContext,
        rule_registry: &RewriteRuleRegistry,
    ) -> VortexResult<Transformed<Expression>> {
        let result = rule_registry.with_typed_reduce_rules(
            &node.id(),
            |rules| -> VortexResult<Option<Expression>> {
                for rule in rules {
                    if let Some(new_expr) = rule.reduce(&node, ctx)? {
                        return Ok(Some(new_expr));
                    }
                }
                Ok(None)
            },
        )?;
        if let Some(new_expr) = result {
            return Ok(Transformed::yes(new_expr));
        }

        // Typed rules can also be applied with untyped context
        let untyped_ctx: RuleContext = ctx.into();
        let result = rule_registry.with_reduce_rules(
            &node.id(),
            |rules| -> VortexResult<Option<Expression>> {
                for rule in rules {
                    if let Some(new_expr) = rule.reduce(&node, &untyped_ctx)? {
                        return Ok(Some(new_expr));
                    }
                }
                Ok(None)
            },
        )?;
        if let Some(new_expr) = result {
            return Ok(Transformed::yes(new_expr));
        }

        Ok(Transformed::no(node))
    }
    expr.transform(
        |node| rewrite(node, ctx, rule_registry),
        |node| rewrite(node, ctx, rule_registry),
    )
    .map(|t| t.into_inner())
}

fn apply_parent_rules_impl_typed(
    expr: Expression,
    ctx: &TypedRuleContext,
    rule_registry: &RewriteRuleRegistry,
) -> VortexResult<Expression> {
    expr.transform_up(|node| {
        for (idx, child) in node.children().iter().enumerate() {
            let result = rule_registry.with_typed_parent_rules(
                &child.id(),
                Some(&node.id()),
                |rules| -> VortexResult<Option<Expression>> {
                    for rule in rules {
                        if let Some(new_expr) = rule.reduce_parent(child, &node, idx, ctx)? {
                            return Ok(Some(new_expr));
                        }
                    }
                    Ok(None)
                },
            )?;
            if let Some(new_expr) = result {
                return Ok(Transformed::yes(new_expr));
            }

            // Typed rules can also be applied with untyped context
            let untyped_ctx: RuleContext = ctx.into();
            let result = rule_registry.with_parent_rules(
                &child.id(),
                Some(&node.id()),
                |rules| -> VortexResult<Option<Expression>> {
                    for rule in rules {
                        if let Some(new_expr) =
                            rule.reduce_parent(child, &node, idx, &untyped_ctx)?
                        {
                            return Ok(Some(new_expr));
                        }
                    }
                    Ok(None)
                },
            )?;
            if let Some(new_expr) = result {
                return Ok(Transformed::yes(new_expr));
            }
        }
        Ok(Transformed::no(node))
    })
    .map(|t| t.into_inner())
}
