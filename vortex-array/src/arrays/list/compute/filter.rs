// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::AddAssign;

use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ListArray, ListVTable, PrimitiveArray};
use crate::compute::{FilterKernel, FilterKernelAdapter, filter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

/// Density threshold for choosing between indices and slices representation when filtering lists.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const LIST_MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.1;

impl FilterKernel for ListVTable {
    fn filter(&self, array: &ListArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let null_mask = array.validity_mask();
        let new_len = selection_mask.true_count();

        // If the entire array is null, then we only need to adjust the length of the array.
        if let Mask::AllFalse(_) = null_mask {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), new_len).into_array(),
            );
        }

        let elements = array.elements();
        let offsets = array.offsets.to_primitive();

        let new_validity = array.validity().filter(selection_mask)?;
        debug_assert!(new_validity.maybe_len().is_none_or(|len| len == new_len));

        let (new_elements, new_offsets) = match_each_integer_ptype!(offsets.ptype(), |O| {
            compute_filtered_elements_and_offsets::<O>(
                elements.as_ref(),
                offsets.as_slice::<O>(),
                selection_mask,
            )?
        });

        // TODO(connor): Use `new_unchecked` here.
        Ok(ListArray::try_new(new_elements, new_offsets.into_array(), new_validity)?.into_array())
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

    // Choose the optimal iteration strategy based on mask density.
    match values.threshold_iter(LIST_MASK_EXPANSION_DENSITY_THRESHOLD) {
        MaskIter::Slices(slices) => {
            compute_filtered_elements_and_offsets_dense(elements, offsets, slices, true_count)
        }
        MaskIter::Indices(indices) => {
            compute_filtered_elements_and_offsets_sparse(elements, offsets, indices, true_count)
        }
    }
}

/// Handles filtering for dense masks (represented as slices).
fn compute_filtered_elements_and_offsets_dense<O: NativePType + AsPrimitive<usize> + AddAssign>(
    elements: &dyn Array,
    offsets: &[O],
    slices: &[(usize, usize)],
    true_count: usize,
) -> VortexResult<(ArrayRef, PrimitiveArray)> {
    let mut new_offsets = BufferMut::<O>::with_capacity(true_count + 1);
    let mut element_slices = Vec::with_capacity(slices.len());
    let mut next_offset: O = O::zero(); // Offsets always start at zero.

    new_offsets.push(next_offset);

    // Collect the element ranges for each range of selected lists.
    for &(start, end) in slices {
        // This represents `start - end` lists in the final array.
        let elems_start = offsets[start].as_();
        let elems_end = offsets[end].as_();

        // Add this range of elements to keep (only if non-empty).
        if elems_start < elems_end {
            element_slices.push((elems_start, elems_end));
        }

        // Add the offsets for each list in this range.
        for list_idx in start..end {
            let list_len = offsets[list_idx + 1] - offsets[list_idx];
            next_offset += list_len;
            new_offsets.push(next_offset);
        }
    }

    // Create a mask from the collected slices.
    let elements_mask = Mask::from_slices(elements.len(), element_slices);

    // Allow the child array to filter themselves.
    let new_elements = filter(elements, &elements_mask)?;
    let new_offsets = PrimitiveArray::new(new_offsets, Validity::NonNullable);

    Ok((new_elements, new_offsets))
}

/// Handles filtering for sparse masks (represented as indices).
fn compute_filtered_elements_and_offsets_sparse<O: NativePType + AsPrimitive<usize> + AddAssign>(
    elements: &dyn Array,
    offsets: &[O],
    indices: &[usize],
    true_count: usize,
) -> VortexResult<(ArrayRef, PrimitiveArray)> {
    let mut new_offsets = BufferMut::<O>::with_capacity(true_count + 1);
    let mut element_slices = Vec::with_capacity(indices.len());
    let mut next_offset: O = O::zero(); // Offsets always start at zero.

    new_offsets.push(next_offset);

    // For sparse masks, we collect the element ranges for each selected list.
    for &list_idx in indices {
        let elem_start = offsets[list_idx].as_();
        let elem_end = offsets[list_idx + 1].as_();

        // Add this range of elements to keep (only if non-empty).
        if elem_start < elem_end {
            element_slices.push((elem_start, elem_end));
        }

        let list_len = offsets[list_idx + 1] - offsets[list_idx];
        next_offset += list_len;
        new_offsets.push(next_offset);
    }

    // Create a mask from the collected slices.
    let elements_mask = Mask::from_slices(elements.len(), element_slices);

    // Allow the child array to filter themselves.
    let new_elements = filter(elements, &elements_mask)?;
    let new_offsets = PrimitiveArray::new(new_offsets, Validity::NonNullable);

    Ok((new_elements, new_offsets))
}
