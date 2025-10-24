// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{BitBufferMut, BufferMut};
use vortex_dtype::{IntegerPType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};

use crate::arrays::{ListArray, ListVTable, PrimitiveArray};
use crate::compute::{FilterKernel, FilterKernelAdapter, filter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.05;

impl FilterKernel for ListVTable {
    fn filter(&self, array: &ListArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let elements = array.elements();
        let offsets = array.offsets().to_primitive();

        let new_validity = array.validity().filter(selection_mask)?;
        debug_assert!(
            new_validity
                .maybe_len()
                .is_none_or(|len| len == selection_mask.true_count())
        );

        let (new_elements, new_offsets) = match_each_integer_ptype!(offsets.ptype(), |O| {
            compute_filtered_elements_and_offsets::<O>(
                elements.as_ref(),
                offsets.as_slice::<O>(),
                selection_mask,
            )?
        });

        // SAFETY: Filter operation maintains all ListArray invariants:
        // - Offsets are monotonically increasing (built correctly above).
        // - Elements are properly filtered to match the offsets.
        // - Validity matches the original array's nullability.
        Ok(unsafe {
            ListArray::new_unchecked(new_elements, new_offsets.into_array(), new_validity)
        }
        .into_array())
    }
}

register_kernel!(FilterKernelAdapter(ListVTable).lift());

/// Given a selection filter mask, computes new offsets and an element array for a list array's
/// child elements array.
///
/// Note that unlike `ListViewArray`, we **must** push the filter down into our child `elements`
/// array because our output array must have lists that are contiguous (something that
/// `ListViewArray` can get away with because it additionally stores a `sizes` child array).
fn compute_filtered_elements_and_offsets<O: IntegerPType>(
    elements: &dyn Array,
    offsets: &[O],
    selection_mask: &Mask,
) -> VortexResult<(ArrayRef, PrimitiveArray)> {
    let values = selection_mask
        .values()
        .vortex_expect("`AllTrue` and `AllFalse` are handled by filter entry point");
    let true_count = selection_mask.true_count();

    let mut new_offsets = BufferMut::<O>::with_capacity(true_count + 1);
    let mut new_mask_builder = BitBufferMut::with_capacity(elements.len());
    let mut next_offset: O = O::zero(); // Offsets always start at zero.

    new_offsets.push(next_offset);

    // Choose the optimal iteration strategy based on selection mask density.
    match values.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD) {
        MaskIter::Slices(slices) => {
            // Dense iteration: process ranges of consecutive selected lists.
            for &(start, end) in slices {
                // Optimization: for dense ranges, we can process the elements mask more efficiently.
                let elems_start = offsets[start].as_();
                let elems_end = offsets[end].as_();

                // Process the entire range of elements at once.
                process_element_range(elems_start, elems_end, &mut new_mask_builder);

                // Add the offsets for each list in this range.
                for i in start..end {
                    let list_len = offsets[i + 1] - offsets[i];
                    next_offset += list_len;
                    new_offsets.push(next_offset);
                }
            }
        }
        MaskIter::Indices(indices) => {
            // Sparse iteration: process individual selected lists.
            for &idx in indices {
                let list_start = offsets[idx].as_();
                let list_end = offsets[idx + 1].as_();

                // Process the elements for this list.
                process_element_range(list_start, list_end, &mut new_mask_builder);

                // Add the offset for this list.
                let offset_len = offsets[idx + 1] - offsets[idx];
                next_offset += offset_len;
                new_offsets.push(next_offset);
            }
        }
    }

    // Fill any trailing elements.
    if new_mask_builder.len() < elements.len() {
        new_mask_builder.append_n(false, elements.len() - new_mask_builder.len());
    }

    // Allow the child array to filter themselves.
    // The `Mask` can determine the best representation based on the buffer's density in the future.
    let new_elements = filter(elements, &Mask::from_buffer(new_mask_builder.freeze()))?;

    let new_offsets = PrimitiveArray::new(new_offsets, Validity::NonNullable);

    Ok((new_elements, new_offsets))
}

/// Process a range of elements for filtering.
fn process_element_range(
    elems_start: usize,
    elems_end: usize,
    new_mask_builder: &mut BitBufferMut,
) {
    let elems_len = elems_end - elems_start;

    // Only process if there are elements to mark.
    if elems_len > 0 {
        // Fill any gaps before this range.
        if elems_start > new_mask_builder.len() {
            new_mask_builder.append_n(false, elems_start - new_mask_builder.len());
        }
        // Keep all elements in this range.
        new_mask_builder.append_n(true, elems_len);
    }
}
