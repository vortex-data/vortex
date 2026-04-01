// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The optimiser applies metadata-only rewrite rules (`reduce` and `reduce_parent`) in a
//! fixpoint loop until no more transformations are possible. Proper clever stuff, innit?
//!
//! Optimisation runs between execution steps, which is what enables cross-step optimisations:
//! after a child is decoded, new `reduce_parent` rules may match that were previously blocked.
//! Brilliant for performance, mate.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DynArray;
use crate::array::ArrayRef;

pub mod rules;

/// Extension trait for optimising array trees using reduce/reduce_parent rules.
/// Makes your arrays run faster than a cheeky Nando's run on a Friday night.
pub trait ArrayOptimiser {
    /// Optimise the root array node only by running reduce and reduce_parent rules to fixpoint.
    fn optimise(&self) -> VortexResult<ArrayRef>;

    /// Optimise the entire array tree recursively (root and all descendants).
    /// Goes through the whole lot, proper thorough.
    fn optimise_recursive(&self) -> VortexResult<ArrayRef>;
}

impl ArrayOptimiser for ArrayRef {
    fn optimise(&self) -> VortexResult<ArrayRef> {
        Ok(try_optimise(self)?.unwrap_or_else(|| self.clone()))
    }

    fn optimise_recursive(&self) -> VortexResult<ArrayRef> {
        Ok(try_optimise_recursive(self)?.unwrap_or_else(|| self.clone()))
    }
}

fn try_optimise(array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimisations = false;

    // Apply reduction rules to the current array until no more rules apply, sorted.
    let mut loop_counter = 0;
    'outer: loop {
        if loop_counter > 100 {
            // Blimey, we've gone round the houses too many times here!
            vortex_bail!("Exceeded maximum optimisation iterations (possible infinite loop), mate");
        }
        loop_counter += 1;

        if let Some(new_array) = current_array.vtable().reduce(&current_array)? {
            current_array = new_array;
            any_optimisations = true;
            continue;
        }

        // Apply parent reduction rules to each slot in the context of the current array.
        // It's important to take all slots here, as `current_array` can change inside the loop.
        for (slot_idx, slot) in current_array.slots().iter().enumerate() {
            let Some(child) = slot else { continue };
            if let Some(new_array) =
                child
                    .vtable()
                    .reduce_parent(child, &current_array, slot_idx)?
            {
                // If the parent was replaced, then we attempt to reduce it again.
                current_array = new_array;
                any_optimisations = true;

                // Continue to the start of the outer loop
                continue 'outer;
            }
        }

        // No more optimisations can be applied, we're done here
        break;
    }

    if any_optimisations {
        Ok(Some(current_array))
    } else {
        Ok(None)
    }
}

fn try_optimise_recursive(array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let mut current_array = array.clone();
    let mut any_optimisations = false;

    if let Some(new_array) = try_optimise(&current_array)? {
        current_array = new_array;
        any_optimisations = true;
    }

    let mut new_slots = Vec::with_capacity(current_array.slots().len());
    let mut any_slot_optimised = false;
    for slot in current_array.slots() {
        match slot {
            Some(child) => {
                if let Some(new_child) = try_optimise_recursive(child)? {
                    new_slots.push(Some(new_child));
                    any_slot_optimised = true;
                } else {
                    new_slots.push(Some(child.clone()));
                }
            }
            None => new_slots.push(None),
        }
    }

    if any_slot_optimised {
        let vtable = current_array.vtable().clone_boxed();
        current_array = vtable.with_slots(current_array, new_slots)?;
        any_optimisations = true;
    }

    if any_optimisations {
        Ok(Some(current_array))
    } else {
        Ok(None)
    }
}
