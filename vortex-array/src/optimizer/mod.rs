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

        // Apply parent reduction rules to each slot in the context of the current array.
        // Its important to take all slots here, as `current_array` can change inside the loop.
        for (slot_idx, slot) in current_array.slots().iter().enumerate() {
            let Some(child) = slot else { continue };
            if let Some(new_array) =
                child
                    .vtable()
                    .reduce_parent(child, &current_array, slot_idx)?
            {
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

    // Collect optimized children first, then apply mutations.
    let slots_snapshot: Vec<_> = current_array.slots().to_vec();
    let mut optimized_slots: Vec<(usize, ArrayRef)> = Vec::new();
    for (i, slot) in slots_snapshot.iter().enumerate() {
        if let Some(child) = slot
            && let Some(new_child) = try_optimize_recursive(child)?
        {
            optimized_slots.push((i, new_child));
        }
    }

    if !optimized_slots.is_empty() {
        let vtable = current_array.vtable().clone_boxed();
        current_array = vtable.with_slots_mut(current_array, &mut |slots| {
            for (i, new_child) in optimized_slots.drain(..) {
                slots[i] = Some(new_child);
            }
        })?;
        any_optimizations = true;
    }

    if any_optimizations {
        Ok(Some(current_array))
    } else {
        Ok(None)
    }
}
