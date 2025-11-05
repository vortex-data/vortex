// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_compute::filter::Filter;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};
use vortex_vector::Vector;
use vortex_vector::fixed_size_list::FixedSizeListVector;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::execution::ExecutionCtx;
use crate::vtable::OperatorVTable;

// TODO(connor): Write some benchmarks to actually figure this out.
/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.05;

impl OperatorVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn execute_batch(
        array: &FixedSizeListArray,
        selection_mask: &Mask,
        _ctx: &mut dyn ExecutionCtx,
    ) -> VortexResult<Vector> {
        let list_size = array.list_size();
        let elem_dtype = array
            .dtype()
            .as_fixed_size_list_element_opt()
            .vortex_expect("`FixedSizeListArray` `DType` was somehow not `FixedSizeList`")
            .clone();

        let new_validity = array.validity_mask().filter(selection_mask);

        // TODO(connor): Should we raise an error if a child kernel returns a data-full `elements`?
        // Technically nothing bad will happen if we don't because of this edge case handling below.

        // If the size of each list is 0, then we know that the child elements must empty. Even if
        // the child kernel incorrectly gives us some data, we can (correctly) just throw it away.
        let elements = if list_size == 0 {
            Vector::empty(&elem_dtype)
        } else {
            // Otherwise, bind the child elements by "expanding" the selection mask out by
            // `list_size` so that we correctly select all of the child elements we need.
            let expanded_selection = expand_selection(selection_mask, list_size as usize);

            array
                .elements()
                .execute_with_selection(&expanded_selection)?
        };

        Ok(FixedSizeListVector::try_new(Arc::new(elements), list_size, new_validity)?.into())
    }
}

/// Given a mask for a fixed-size list array, creates a new mask for the underlying elements.
///
/// This function simply "expands" out the input `selection_mask` by duplicating each bit
/// `list_size` times.
///
/// The output `Mask` is guaranteed to have a length equal to `selection_mask.len() * list_size`.
fn expand_selection(selection_mask: &Mask, list_size: usize) -> Mask {
    let expanded_len = selection_mask.len() * list_size;

    let values = match selection_mask {
        Mask::AllTrue(_) => return Mask::AllTrue(expanded_len),
        Mask::AllFalse(_) => return Mask::AllFalse(expanded_len),
        Mask::Values(values) => values,
    };

    // Use threshold_iter to choose the optimal representation based on density.
    let expanded_slices = match values.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD) {
        MaskIter::Slices(slices) => {
            // Expand a dense mask (represented as slices) by scaling each slice by `list_size`.
            slices
                .iter()
                .map(|&(start, end)| (start * list_size, end * list_size))
                .collect()
        }
        MaskIter::Indices(indices) => {
            // Expand a sparse mask (represented as indices) by duplicating each index `list_size`
            // times.
            //
            // TODO(connor): Note that in the worst case, it is possible that we create only a few
            // slices with a small range (for example, when list_size <= 2). This could be further
            // optimized, but we choose simplicity for now.
            indices
                .iter()
                .map(|&idx| {
                    let start = idx * list_size;
                    let end = (idx + 1) * list_size;
                    (start, end)
                })
                .collect()
        }
    };

    Mask::from_slices(expanded_len, expanded_slices)
}
