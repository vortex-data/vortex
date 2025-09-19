// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::AddAssign;

use arrow_buffer::BooleanBufferBuilder;
use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};

use crate::arrays::{ListViewArray, ListViewVTable, PrimitiveArray};
use crate::compute::{FilterKernel, FilterKernelAdapter, filter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

/// Density threshold for choosing between indices and slices iteration for the selection mask.
///
/// When the mask density is below this threshold, we use indices iteration; otherwise, we use
/// slices iteration. Note that both paths build a BooleanBuffer for the element mask.
const LISTVIEW_SELECTION_MASK_DENSITY_THRESHOLD: f64 = 0.1;

impl FilterKernel for ListViewVTable {
    fn filter(&self, array: &ListViewArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let elements = array.elements();
        let offsets = array.offsets().to_primitive();
        let sizes = array.sizes().to_primitive();

        let new_validity = array.validity().filter(selection_mask)?;
        debug_assert!(
            new_validity
                .maybe_len()
                .is_none_or(|len| len == selection_mask.true_count())
        );

        let (new_elements, new_offsets, new_sizes) =
            match_each_integer_ptype!(offsets.ptype(), |O| {
                match_each_integer_ptype!(sizes.ptype(), |S| {
                    compute_filtered_elements_and_arrays::<O, S>(
                        elements.as_ref(),
                        offsets.as_slice::<O>(),
                        sizes.as_slice::<S>(),
                        selection_mask,
                    )?
                })
            });

        // SAFETY: Filter operation maintains all ListViewArray invariants:
        // - Offsets and sizes have the same length (both filtered by selection_mask).
        // - Elements are properly filtered to match the new offsets and sizes.
        // - Validity matches the filtered array's nullability.
        Ok(unsafe {
            ListViewArray::new_unchecked(
                new_elements,
                new_offsets.into_array(),
                new_sizes.into_array(),
                new_validity,
            )
        }
        .into_array())
    }
}

register_kernel!(FilterKernelAdapter(ListViewVTable).lift());

/// Given a selection filter mask, computes new offsets, sizes, and an element array for a
/// ListView array's child elements array.
fn compute_filtered_elements_and_arrays<
    O: NativePType + AsPrimitive<usize> + AddAssign,
    S: NativePType + AsPrimitive<usize>,
>(
    elements: &dyn Array,
    offsets: &[O],
    sizes: &[S],
    selection_mask: &Mask,
) -> VortexResult<(ArrayRef, PrimitiveArray, PrimitiveArray)> {
    let values = selection_mask
        .values()
        .vortex_expect("`AllTrue` and `AllFalse` are handled by filter entry point");
    let true_count = selection_mask.true_count();

    let mut new_offsets = BufferMut::<O>::with_capacity(true_count);
    let mut new_sizes = BufferMut::<S>::with_capacity(true_count);
    let mut new_mask_builder = BooleanBufferBuilder::new(elements.len());
    let mut next_offset: O = O::zero(); // Offsets always start at zero for filtered arrays.

    // Choose the optimal iteration strategy based on selection mask density.
    match values.threshold_iter(LISTVIEW_SELECTION_MASK_DENSITY_THRESHOLD) {
        MaskIter::Slices(slices) => {
            // Dense iteration: process ranges of consecutive selected lists.
            for &(start, end) in slices {
                // This represents `start - end` lists in the final array.
                for i in start..end {
                    let list_offset = offsets[i].as_();
                    let list_size = sizes[i].as_();

                    // Mark elements as selected based on this list's range.
                    if list_size > 0 {
                        // Fill false values before this list's elements.
                        if list_offset > new_mask_builder.len() {
                            new_mask_builder.append_n(list_offset - new_mask_builder.len(), false);
                        }
                        // Mark this list's elements as true.
                        new_mask_builder.append_n(list_size, true);
                    }

                    // Add the new offset and size.
                    new_offsets.push(next_offset);
                    new_sizes.push(sizes[i]);
                    next_offset += O::from(sizes[i]).vortex_expect("size fits in O");
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
                let list_offset = offsets[idx].as_();
                let list_size = sizes[idx].as_();

                // Fill false values up to the start of this list.
                if list_offset > last_elem_end {
                    new_mask_builder.append_n(list_offset - last_elem_end, false);
                }
                // Keep the elements in this list.
                if list_size > 0 {
                    new_mask_builder.append_n(list_size, true);
                    last_elem_end = list_offset + list_size;
                }

                // Add the new offset and size.
                new_offsets.push(next_offset);
                new_sizes.push(sizes[idx]);
                next_offset += O::from(sizes[idx]).vortex_expect("size fits in O");
            }

            // For sparse iteration, handle any gap after the last processed element.
            if last_elem_end < elements.len() {
                new_mask_builder.append_n(elements.len() - last_elem_end, false);
            }
        }
    }

    // Allow the child array to filter themselves.
    // The `Mask` can determine the best representation based on the buffer's density.
    let new_elements = filter(elements, &Mask::from_buffer(new_mask_builder.finish()))?;

    let new_offsets = PrimitiveArray::new(new_offsets, Validity::NonNullable);
    let new_sizes = PrimitiveArray::new(new_sizes, Validity::NonNullable);

    Ok((new_elements, new_offsets, new_sizes))
}
