// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::AddAssign;

use arrow_buffer::BooleanBufferBuilder;
use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ListArray, ListVTable, PrimitiveArray};
use crate::compute::{FilterKernel, FilterKernelAdapter, filter};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

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
    let mut new_offsets = BufferMut::<O>::with_capacity(selection_mask.true_count() + 1);
    let mut new_mask_builder = BooleanBufferBuilder::new(elements.len());
    let mut next_offset: O = O::zero(); // Offsets always start at zero.

    new_offsets.push(next_offset);

    // We can batch this operation for the mask builder (but not the offsets).
    for &(start, end) in selection_mask
        .values()
        .vortex_expect("`AllTrue` and `AllFalse` are handled by filter entry point")
        .slices()
    {
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
    new_mask_builder.append_n(elements.len() - new_mask_builder.len(), false);

    // Allow the child array to filter themselves.
    let new_elements = filter(elements, &Mask::from_buffer(new_mask_builder.finish()))?;
    let new_offsets = PrimitiveArray::new(new_offsets, Validity::NonNullable);

    Ok((new_elements, new_offsets))
}
