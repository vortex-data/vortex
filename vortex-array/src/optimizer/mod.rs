// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Array;
use crate::array::ArrayRef;

pub mod rules;

pub trait ArrayOptimizer {
    /// Optimize the root array node only.
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Try to optimize the root array node only, returning None if no optimizations were applied.
    fn try_optimize(&self) -> VortexResult<Option<ArrayRef>>;

    /// Optimize the entire array tree recursively.
    fn optimize_recursive(&self) -> VortexResult<ArrayRef>;

    /// Try to optimize the entire array tree recursively, returning None if no optimizations were applied.
    fn try_optimize_recursive(&self) -> VortexResult<Option<ArrayRef>>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        Ok(self.clone().try_optimize()?.unwrap_or_else(|| self.clone()))
    }

    #[expect(clippy::cognitive_complexity)]
    fn try_optimize(&self) -> VortexResult<Option<ArrayRef>> {
        let mut current_array = self.clone();
        let mut any_optimizations = false;

        // Apply reduction rules to the current array until no more rules apply.
        let mut loop_counter = 0;
        'outer: loop {
            if loop_counter > 100 {
                vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
            }
            loop_counter += 1;

            if let Some(new_array) = current_array.reduce()? {
                current_array = new_array;
                any_optimizations = true;
                continue;
            }

            // Apply parent reduction rules to each child in the context of the current array.
            for (idx, child) in current_array.children().iter().enumerate() {
                if let Some(new_array) = child.reduce_parent(&current_array, idx)? {
                    // If the parent was replaced, then we attempt to reduce it again.
                    current_array = new_array;
                    any_optimizations = true;

                    // Continue to the start of the outer loop
                    continue 'outer;
                }
            }

            // No more optimizations can be applied
            break;
        }

        if any_optimizations {
            tracing::debug!(
                "Optimized root-only array\n{}",
                current_array.display_tree()
            );
            Ok(Some(current_array))
        } else {
            tracing::debug!("No optimizations applied to array\n{}", self.display_tree());
            Ok(None)
        }
    }

    fn optimize_recursive(&self) -> VortexResult<ArrayRef> {
        Ok(self
            .clone()
            .try_optimize_recursive()?
            .unwrap_or_else(|| self.clone()))
    }

    fn try_optimize_recursive(&self) -> VortexResult<Option<ArrayRef>> {
        let mut current_array = self.clone();
        let mut any_optimizations = false;

        if let Some(new_array) = current_array.clone().try_optimize()? {
            current_array = new_array;
            any_optimizations = true;
        }

        let mut new_children = Vec::with_capacity(current_array.nchildren());
        let mut any_child_optimized = false;
        for child in current_array.children() {
            if let Some(new_child) = child.clone().try_optimize_recursive()? {
                new_children.push(new_child);
                any_child_optimized = true;
            } else {
                new_children.push(child.clone());
            }
        }

        if any_child_optimized {
            current_array = current_array.with_children(new_children)?;
            any_optimizations = true;
        }

        if any_optimizations {
            Ok(Some(current_array))
        } else {
            Ok(None)
        }
    }
}
