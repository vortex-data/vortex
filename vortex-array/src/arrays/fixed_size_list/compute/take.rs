// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBufferBuilder;
use vortex_dtype::{NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable, PrimitiveArray};
use crate::builders::{ArrayBuilder, PrimitiveBuilder};
use crate::compute::{self, TakeKernel, TakeKernelAdapter};
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for FixedSizeListVTable {
    fn take(&self, array: &FixedSizeListArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive();

        match_each_integer_ptype!(indices.ptype(), |I| {
            take_with_indices::<I>(array, &indices)
        })
    }
}

register_kernel!(TakeKernelAdapter(FixedSizeListVTable).lift());

fn take_with_indices<I: NativePType>(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;

    let array_validity = array.validity_mask();
    let indices_validity = indices_array.validity_mask();

    let indices: &[I] = indices_array.as_slice::<I>();
    let len = indices.len();

    if list_size == 0 {
        // If the `list_size` is 0, then we simply need figure out where the nulls are and return
        // an array with empty elements and the correct length.

        debug_assert!(array.elements().is_empty(), "degenerate list is invalid");

        let mut new_validity = BooleanBufferBuilder::new(len);

        // We iterate over each data index as well as if each data index is null.
        indices_validity.iter_bools(|validity_iter| {
            for (data_idx, is_valid) in indices.iter().zip(validity_iter) {
                let data_idx = data_idx.to_usize().unwrap_or_else(|| {
                    vortex_panic!("Failed to convert index to usize: {}", data_idx)
                });

                new_validity.append(is_valid && array_validity.value(data_idx));
            }
        });

        let new_validity = Validity::from(new_validity.finish());
        debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == len));

        // SAFETY: The `list_size` is 0, the elements array is empty, and the validity has the
        // correct length.
        return Ok(unsafe {
            FixedSizeListArray::new_unchecked(
                array.elements().clone(),
                array.list_size(),
                new_validity,
                len,
            )
        }
        .into_array());
    }

    // We will create new indices specialized for the child `element` array.
    let mut elements_indices =
        PrimitiveBuilder::<I>::with_capacity(Nullability::Nullable, len * list_size);
    let mut new_validity = BooleanBufferBuilder::new(len);

    // We iterate over each data index as well as if each data index is null.
    indices_validity.iter_bools(|validity_iter| {
        for (data_idx, is_valid) in indices.iter().zip(validity_iter) {
            let data_idx = data_idx
                .to_usize()
                .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

            // If we have a null index (null code), then we treat this as referencing a null list.
            if !is_valid || !array_validity.value(data_idx) {
                elements_indices.append_nulls(list_size);
                new_validity.append(false);
                continue;
            }

            let list_start = data_idx * list_size;
            let list_end = (data_idx + 1) * list_size;

            for i in list_start..list_end {
                // TODO(connor): This can be optimized with `UninitRange`.
                elements_indices.append_value(I::from_usize(i).vortex_expect("i < additional"))
            }

            new_validity.append(true);
        }

        let elements_indices = elements_indices.finish();
        debug_assert_eq!(elements_indices.len(), len * list_size);

        let new_elements = compute::take(array.elements(), elements_indices.as_ref())?;
        debug_assert_eq!(new_elements.len(), len * list_size);

        let new_validity = Validity::from(new_validity.finish());
        debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == len));

        // SAFETY: We checked above that `list_size` is not 0, that the length of the elements array
        // is a multiple of the `list_size`, and the validity will have the correct length.
        Ok(unsafe {
            FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, len)
        }
        .into_array())
    })
}
