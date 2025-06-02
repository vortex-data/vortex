use std::ops::AddAssign;

use arrow_buffer::BooleanBufferBuilder;
use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ListArray, ListVTable, PrimitiveArray};
use crate::compute::{FilterKernel, FilterKernelAdapter, arrow_filter_fn, filter};
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl FilterKernel for ListVTable {
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        let offsets = array.offsets.to_primitive()?;

        match array.validity_mask()? {
            Mask::AllTrue(_) => {
                match_each_integer_ptype!(offsets.ptype(), |I| {
                    filter_all_valid::<I>(
                        offsets.as_slice::<I>(),
                        array.elements().as_ref(),
                        mask,
                        array.dtype().nullability(),
                    )
                })
            }
            Mask::AllFalse(_) => {
                // If all array offsets are null, then the array is simply null?
                Ok(
                    ConstantArray::new(Scalar::null(array.dtype().clone()), mask.true_count())
                        .into_array(),
                )
            }
            Mask::Values(_) => {
                // TODO(ngates): implemented null filtering
                arrow_filter_fn(array.as_ref(), mask)
            }
        }
    }
}

fn filter_all_valid<I: NativePType + AsPrimitive<usize> + AddAssign>(
    offsets: &[I],
    elements: &dyn Array,
    mask: &Mask,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    // We compute a new set of offsets, as well as a mask for filtering the elements.
    let mut new_offsets = BufferMut::<I>::with_capacity(mask.true_count() + 1);
    new_offsets.push(I::zero());
    let mut new_offset: I = I::zero();

    let mut mask_builder = BooleanBufferBuilder::new(elements.len());
    for &(start, end) in mask
        .values()
        .vortex_expect("all true and all false are handled by filter entry point")
        .slices()
    {
        let elem_start: usize = offsets[start].as_();
        let elem_end: usize = offsets[end].as_();
        let elem_len = elem_end - elem_start;
        mask_builder.append_n(elem_start - mask_builder.len(), false);
        mask_builder.append_n(elem_len, true);

        // Add each of the new offsets into the result
        for i in start..end {
            let elem_len = offsets[i + 1] - offsets[i];
            new_offset += elem_len;
            new_offsets.push(new_offset);
        }
    }
    mask_builder.append_n(elements.len() - mask_builder.len(), false);

    let new_elements = filter(elements, &Mask::from_buffer(mask_builder.finish()))?;

    let new_offsets = PrimitiveArray::new(new_offsets, Validity::NonNullable).into_array();

    Ok(ListArray::try_new(new_elements, new_offsets, Validity::from(nullability))?.into_array())
}

register_kernel!(FilterKernelAdapter(ListVTable).lift());
