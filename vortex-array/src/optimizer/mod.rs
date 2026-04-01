// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The optimizer applies metadata-only rewrite rules (`reduce` and `reduce_parent`) in a
//! fixpoint loop until no more transformations are possible.
//!
//! Optimization runs between execution steps, which is what enables cross-step optimizations:
//! after a child is decoded, new `reduce_parent` rules may match that were previously blocked.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DynArray;
use crate::array::ArrayRef;

pub mod rules;

/// Extension trait for optimizing array trees using reduce/reduce_parent rules.
pub trait ArrayOptimizer {
    /// Optimize the root array node only by running reduce and reduce_parent rules to fixpoint.
    fn optimize(&self) -> VortexResult<ArrayRef>;

    /// Optimize the entire array tree recursively (root and all descendants).
    fn optimize_recursive(&self) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizer for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        try_optimize(self.clone())
    }

    fn optimize_recursive(&self) -> VortexResult<ArrayRef> {
        try_optimize_recursive(self.clone())
    }
}

fn try_optimize(array: ArrayRef) -> VortexResult<ArrayRef> {
    let mut current_array = array;

    // Apply reduction rules to the current array until no more rules apply.
    let mut loop_counter = 0;
    'outer: loop {
        if loop_counter > 100 {
            vortex_bail!("Exceeded maximum optimization iterations (possible infinite loop)");
        }
        loop_counter += 1;

        if let Some(new_array) = current_array.vtable().reduce(&current_array)? {
            current_array = new_array;
            continue;
        }

        // Apply parent reduction rules to each slot in the context of the current array.
        // Its important to take all slots here, as `current_array` can change inside the loop.
        for (slot_idx, slot) in current_array.slots().iter().enumerate() {
            let Some(child) = slot else { continue };
            if let Some(new_array) =
                child
                    .vtable()
                    .reduce_parent(child, &current_array, slot_idx)?
            {
                current_array = new_array;
                continue 'outer;
            }
        }

        break;
    }

    Ok(current_array)
}

fn try_optimize_recursive(array: ArrayRef) -> VortexResult<ArrayRef> {
    let mut current_array = try_optimize(array)?;

    // Optimize each child slot in-place.
    let nslots = current_array.slots().len();
    for i in 0..nslots {
        if let Some(child) = current_array.take_slot(i) {
            let optimized = try_optimize_recursive(child)?;
            current_array.put_slot(i, optimized);
        }
    }

    Ok(current_array)
}
