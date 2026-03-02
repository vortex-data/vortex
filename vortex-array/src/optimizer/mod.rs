// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DynArray;
use crate::array::ArrayRef;

pub mod rules;

pub trait ArrayOptimizer {
    /// Optimize the root array node only.
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively.
    fn optimize_recursive(&self) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        Ok(try_optimize(self)?.unwrap_or_else(|| self.clone()))
    }

    fn optimize_recursive(&self) -> VortexResult<ArrayRef> {
        Ok(try_optimize_recursive(self)?.unwrap_or_else(|| self.clone()))
    }
}

fn try_optimize(array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimizations = false;

    // Apply reduction rules to the current array until no more rules apply.
    let mut loop_counter = 0;
    'outer: loop {
        if loop_counter > 100 {
            vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
        }
        loop_counter += 1;

        if let Some(new_array) = current_array.vtable().reduce(&current_array)? {
            current_array = new_array;
            any_optimizations = true;
            continue;
        }

        // Apply parent reduction rules to each child in the context of the current array.
        // Its important to take all children here, as `current_array` can change inside the loop.
        for (idx, child) in current_array.children().iter().enumerate() {
            if let Some(new_array) = child.vtable().reduce_parent(child, &current_array, idx)? {
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
        Ok(Some(current_array))
    } else {
        Ok(None)
    }
}

fn try_optimize_recursive(array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimizations = false;

    if let Some(new_array) = try_optimize(&current_array)? {
        current_array = new_array;
        any_optimizations = true;
    }

    let mut new_children = Vec::with_capacity(current_array.nchildren());
    let mut any_child_optimized = false;
    for child in current_array.children() {
        if let Some(new_child) = try_optimize_recursive(&child)? {
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
