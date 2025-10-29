// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::vtable::VTable;
use crate::{Array, ArrayRef, ArrayVisitor};

impl dyn Array + '_ {
    /// Optimize the entire tree in a single bottom-up pass
    pub fn optimize(&self) -> VortexResult<ArrayRef> {
        optimize_recursive(&self.to_array(), None)
    }
}

fn optimize_recursive(
    node: &ArrayRef,
    parent_context: Option<(ArrayRef, usize)>,
) -> VortexResult<ArrayRef> {
    // 1. Recursively optimize all children first (post-order traversal)
    let node_children = node.children();
    let optimized_children: VortexResult<Vec<_>> = node_children
        .iter()
        .enumerate()
        .map(|(idx, child)| optimize_recursive(child, Some((node.clone(), idx))))
        .collect();

    let optimized_children = optimized_children?;

    // 2. Rebuild node with optimized children if any changed
    let mut node = if !optimized_children.is_empty() {
        let any_changed = node_children
            .iter()
            .zip(&optimized_children)
            .any(|(old, new)| !Arc::ptr_eq(old, new));

        if any_changed {
            node.with_children(&optimized_children)?
        } else {
            node.clone()
        }
    } else {
        node.clone()
    };

    // 3. Try reduce_children (e.g., constant folding with optimized children)
    if let Some(reduced) = node.reduce_children()? {
        node = reduced;
    }

    // 4. If we have a parent, try reduce_parent (e.g., filter pushdown)
    if let Some((parent, child_idx)) = parent_context
        && let Some(new_parent) = node.reduce_parent(parent, child_idx)?
    {
        // This child replaced its parent! Return the replacement
        return Ok(new_parent);
    }

    Ok(node)
}

/// An optimizer rule that tries to reduce/replace a parent array where the implementer is a
/// child array in the `CHILD_IDX` position of the parent array.
pub trait ReduceParent<Parent: VTable, const CHILD_IDX: usize>: VTable {
    /// Try to reduce/replace the given parent array based on this child array.
    ///
    /// If no reduction is possible, return None.
    fn reduce_parent(array: &Self::Array, parent: &Parent::Array)
    -> VortexResult<Option<ArrayRef>>;
}

/// A generic optimizer rule that can be applied to an array to try to optimize it.
pub trait OptimizerRule {
    /// Try to optimize the given array, returning a replacement if successful.
    ///
    /// If no optimization is possible, return None.
    fn optimize(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::arrays::{ConstantArray, ConstantVTable};
    use crate::compute::arrays::logical::{LogicalArray, LogicalOperator};
    use crate::IntoArray;

    #[test]
    fn test_constant_fold_logical_and() {
        // Create two constant boolean arrays: AND(true, false) => false
        let lhs = ConstantArray::new(Scalar::bool(true, Nullability::NonNullable), 10).into_array();
        let rhs =
            ConstantArray::new(Scalar::bool(false, Nullability::NonNullable), 10).into_array();

        let logical_and = LogicalArray::new(lhs, rhs, LogicalOperator::And).into_array();

        // Optimize should fold this to a constant false array
        let optimized = logical_and.optimize().unwrap();

        // Check if the result is a constant array with value false
        let constant = optimized.as_::<ConstantVTable>();
        assert_eq!(constant.scalar().as_bool().value(), Some(false));
        assert_eq!(optimized.len(), 10);
    }
}
