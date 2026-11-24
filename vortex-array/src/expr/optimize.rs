// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::BoundExpression;
use crate::scalar_fn::ExpressionReduceNode;

impl BoundExpression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify` - scalar-function simplifications over the node
    /// 2. `reduce` - abstract reduction rules via `ReduceNode`
    pub fn optimize(&self) -> VortexResult<BoundExpression> {
        Ok(self.try_optimize()?.unwrap_or_else(|| self.clone()))
    }

    /// Apply this node's own simplification rule, if it has one.
    ///
    /// Non-scalar nodes carry no rules, so they never simplify.
    fn simplify_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => Ok(scalar_fn
                .simplify_expression(&ExpressionReduceNode::new(self))?
                .map(ExpressionReduceNode::into_expression)),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    /// Apply this node's own abstract reduction rule, if it has one.
    fn reduce_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => Ok(scalar_fn
                .reduce_expression(&ExpressionReduceNode::new(self))?
                .map(ExpressionReduceNode::into_expression)),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    fn try_optimize(&self) -> VortexResult<Option<BoundExpression>> {
        // Copy-on-write: `current` stays None until a rule fires, so unchanged nodes (the common
        // case) are never cloned.
        let mut current: Option<BoundExpression> = None;
        let mut loop_counter = 0;

        loop {
            if loop_counter > 100 {
                vortex_error::vortex_bail!(
                    "Exceeded maximum optimization iterations (possible infinite loop)"
                );
            }
            loop_counter += 1;

            let expr = current.as_ref().unwrap_or(self);
            let mut changed = false;

            if let Some(simplified) = expr.simplify_node()? {
                current = Some(simplified);
                changed = true;
            }

            let expr = current.as_ref().unwrap_or(self);
            if let Some(reduced) = expr.reduce_node()? {
                current = Some(reduced);
                changed = true;
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }
    /// Optimize the entire expression tree recursively.
    ///
    /// Optimizes children first (bottom-up), then optimizes the root.
    pub fn optimize_recursive(&self) -> VortexResult<BoundExpression> {
        Ok(self
            .try_optimize_recursive()?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the entire expression tree recursively.
    pub fn try_optimize_recursive(&self) -> VortexResult<Option<BoundExpression>> {
        // First optimize the root
        let mut current = self.try_optimize()?;

        // Then recursively optimize children. The new children vector is only allocated once a
        // child actually changes, so fully-optimized subtrees cost no allocations.
        let expr = current.as_ref().unwrap_or(self);
        let children = expr.children();
        let mut new_children: Option<Vec<BoundExpression>> = None;
        for (idx, child) in children.iter().enumerate() {
            if let Some(optimized) = child.try_optimize_recursive()? {
                new_children
                    .get_or_insert_with(|| children[..idx].to_vec())
                    .push(optimized);
            } else if let Some(new_children) = new_children.as_mut() {
                new_children.push(child.clone());
            }
        }

        if let Some(new_children) = new_children {
            let updated = expr.clone().with_children(new_children)?;

            // After updating children, try to optimize root again
            current = Some(updated.try_optimize()?.unwrap_or(updated));
        }

        Ok(current)
    }
}
