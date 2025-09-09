// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::AddAssign;

use arrow_buffer::BooleanBufferBuilder;
use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};

use crate::arrays::{ListArray, ListVTable, PrimitiveArray};
use crate::compute::{FilterKernel, FilterKernelAdapter, filter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

/// Density threshold for choosing between indices and slices iteration for the selection mask.
///
/// When the mask density is below this threshold, we use indices iteration; otherwise, we use
/// slices iteration. Note that both paths build a BooleanBuffer for the element mask.
const LIST_SELECTION_MASK_DENSITY_THRESHOLD: f64 = 0.1;

impl FilterKernel for ListVTable {
    fn filter(&self, array: &ListArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let elements = array.elements();
        let offsets = array.offsets.to_primitive();

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
fn compute_filtered_elements_and_offsets<O: NativePType + AsPrimitive<usize> + AddAssign>(
    elements: &dyn Array,
    offsets: &[O],
    selection_mask: &Mask,
) -> VortexResult<(ArrayRef, PrimitiveArray)> {
    let values = selection_mask
        .values()
        .vortex_expect("`AllTrue` and `AllFalse` are handled by filter entry point");
    let true_count = selection_mask.true_count();

    let mut new_offsets = BufferMut::<O>::with_capacity(true_count + 1);
    let mut new_mask_builder = BooleanBufferBuilder::new(elements.len());
    let mut next_offset: O = O::zero(); // Offsets always start at zero.

    new_offsets.push(next_offset);

    // Choose the optimal iteration strategy based on selection mask density.
    match values.threshold_iter(LIST_SELECTION_MASK_DENSITY_THRESHOLD) {
        MaskIter::Slices(slices) => {
            // Dense iteration: process ranges of consecutive selected lists.
            for &(start, end) in slices {
                // This represents `start - end` lists in the final array.
                let elems_start = offsets[start].as_();
                let elems_end = offsets[end].as_();
                let elems_len = elems_end - elems_start;

                // Remove any unnecessary elements before the start of the `start` list.
                new_mask_builder.append_n(elems_start - new_mask_builder.len(), false);
                // Keep the elements that _are_ in the list.
                new_mask_builder.append_n(elems_len, true);

                // Add the offsets for each list in this range.
                for i in start..end {
                    let list_len = offsets[i + 1] - offsets[i];
                    next_offset += list_len;
                    new_offsets.push(next_offset);
                }
            }

            // Remove any trailing elements.
            if new_mask_builder.len() < elements.len() {
                new_mask_builder.append_n(elements.len() - new_mask_builder.len(), false);
            }
        }
        MaskIter::Indices(indices) => {
            // Sparse iteration: process individual selected lists.
            let mut last_elem_end: usize = 0;

            for &idx in indices {
                let list_start = offsets[idx].as_();
                let list_end = offsets[idx + 1].as_();
                let list_len = list_end - list_start;

                // Fill false values up to the start of this list.
                if list_start > last_elem_end {
                    new_mask_builder.append_n(list_start - last_elem_end, false);
                }
                // Keep the elements in this list.
                new_mask_builder.append_n(list_len, true);
                last_elem_end = list_end;

                // Add the offset for this list.
                let offset_len = offsets[idx + 1] - offsets[idx];
                next_offset += offset_len;
                new_offsets.push(next_offset);
            }

            // For sparse iteration, handle any gap after the last processed element.
            if last_elem_end < elements.len() {
                new_mask_builder.append_n(elements.len() - last_elem_end, false);
            }
        }
    }

    // Allow the child array to filter themselves.
    // The `Mask` can determine the best representation based on the buffer's density in the future.
    let new_elements = filter(elements, &Mask::from_buffer(new_mask_builder.finish()))?;

    let new_offsets = PrimitiveArray::new(new_offsets, Validity::NonNullable);

    Ok((new_elements, new_offsets))
}
