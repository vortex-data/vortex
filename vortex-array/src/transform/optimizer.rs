// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayVisitor;
use crate::array::ArrayRef;
use crate::expr::transform::ExprOptimizer;
use crate::session::rewrite::ArrayRewriteRuleRegistry;
use crate::transform::context::ArrayRuleContext;

/// Optimizer for arrays that applies registered rewrite rules.
///
/// This optimizer recursively traverses an array tree, applying reduce rules
/// to transform arrays into more efficient representations.
#[derive(Debug, Clone)]
pub struct ArrayOptimizer {
    rule_registry: ArrayRewriteRuleRegistry,
    expr_optimizer: ExprOptimizer,
}

impl ArrayOptimizer {
    /// Creates a new optimizer with the given rule registry and expression optimizer.
    pub fn new(rule_registry: ArrayRewriteRuleRegistry, expr_optimizer: ExprOptimizer) -> Self {
        Self {
            rule_registry,
            expr_optimizer,
        }
    }

    /// Optimize the given array by applying registered rewrite rules.
    ///
    /// This performs two passes following the ExprSession pattern:
    /// 1. Apply parent rules - bottom-up traversal checking parent-child relationships
    /// 2. Apply reduce rules - bottom-up traversal applying transformations to each node
    pub fn optimize_array(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        let ctx = ArrayRuleContext::new(self.expr_optimizer.clone());

        // First pass: apply parent rules
        let array = self.apply_parent_rules(array, &ctx)?;

        // Second pass: apply reduce rules
        let array = self.apply_reduce_rules(array, &ctx)?;

        Ok(array)
    }

    /// Apply parent rules in a bottom-up traversal.
    ///
    /// For each array, recursively process children first, then check if any parent
    /// rules apply to transform children based on their parent context.
    fn apply_parent_rules(
        &self,
        array: ArrayRef,
        ctx: &ArrayRuleContext,
    ) -> VortexResult<ArrayRef> {
        // First, recursively apply parent rules to all children
        let children = array.children();
        if children.is_empty() {
            return Ok(array);
        }

        let mut optimized_children = Vec::with_capacity(children.len());
        let mut children_changed = false;

        for child in children.iter() {
            let optimized_child = self.apply_parent_rules(child.clone(), ctx)?;
            children_changed |= !std::sync::Arc::ptr_eq(&optimized_child, child);
            optimized_children.push(optimized_child);
        }

        // Reconstruct array with optimized children if any changed
        let array = if children_changed {
            array.with_children(&optimized_children)?
        } else {
            array
        };

        // Now try to apply parent rules to each optimized child in the context of this array
        // Use the optimized_children list directly instead of re-fetching from array.children()
        // let mut transformed_children = Vec::with_capacity(optimized_children.len());

        for (idx, child) in optimized_children.iter().enumerate() {
            let child_id = child.encoding_id();
            let parent_id = array.encoding_id();

            let result = self.rule_registry.with_parent_rules(
                &child_id,
                Some(&parent_id),
                |rules| -> VortexResult<Option<ArrayRef>> {
                    for rule in rules {
                        if let Some(new_array) = rule.reduce_parent(child, &array, idx, ctx)? {
                            return Ok(Some(new_array));
                        }
                    }
                    Ok(None)
                },
            )?;

            if let Some(transformed) = result {
                return Ok(transformed);
            }
        }

        // Reconstruct array with transformed children if any rules matched
        Ok(array)
    }

    /// Apply reduce rules in a bottom-up traversal.
    ///
    /// For each array, recursively process children first, then try to apply
    /// reduce rules to transform the array itself.
    fn apply_reduce_rules(
        &self,
        array: ArrayRef,
        ctx: &ArrayRuleContext,
    ) -> VortexResult<ArrayRef> {
        // First, recursively apply reduce rules to all children
        let children = array.children();
        if !children.is_empty() {
            let mut new_children = Vec::with_capacity(children.len());
            let mut changed = false;

            for child in children.iter() {
                let optimized_child = self.apply_reduce_rules(child.clone(), ctx)?;
                changed |= !std::sync::Arc::ptr_eq(&optimized_child, child);
                new_children.push(optimized_child);
            }

            // Reconstruct array with optimized children if any changed
            let array = if changed {
                array.with_children(&new_children)?
            } else {
                array
            };

            // Now try to apply reduce rules to this array
            self.try_reduce(array, ctx)
        } else {
            // Leaf node - just try to reduce
            self.try_reduce(array, ctx)
        }
    }

    /// Try to apply reduce rules to a single array, recursively if a rule matches.
    fn try_reduce(&self, array: ArrayRef, ctx: &ArrayRuleContext) -> VortexResult<ArrayRef> {
        let encoding_id = array.encoding_id();
        let result = self.rule_registry.with_reduce_rules(
            &encoding_id,
            |rules| -> VortexResult<Option<ArrayRef>> {
                for rule in rules {
                    if let Some(new_array) = rule.reduce(&array, ctx)? {
                        return Ok(Some(new_array));
                    }
                }
                Ok(None)
            },
        )?;

        if let Some(transformed) = result {
            // Rule matched - recursively try to reduce the result
            // self.try_reduce(transformed, ctx)
            Ok(transformed)
        } else {
            Ok(array)
        }
    }
}
