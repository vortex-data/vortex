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

/// Dispatches to the appropriate take implementation based on list size and nullability.
fn take_with_indices<I: NativePType>(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;

    // Make sure to handle degenerate case where lists have size 0 (these can take fast paths).
    if list_size == 0 {
        debug_assert!(
            array.elements().is_empty(),
            "degenerate list must have empty elements"
        );

        // The result's nullability is the union of the input nullabilities.
        if array.dtype().is_nullable() || indices_array.dtype().is_nullable() {
            Ok(take_degenerate_nullable::<I>(array, indices_array))
        } else {
            Ok(take_degenerate_non_nullable(array, indices_array))
        }
    } else {
        // The result's nullability is the union of the input nullabilities.
        if array.dtype().is_nullable() || indices_array.dtype().is_nullable() {
            take_nullable_fsl::<I>(array, indices_array)
        } else {
            take_non_nullable_fsl::<I>(array, indices_array)
        }
    }
}

/// Returns empty non-nullable lists with the requested length.
fn take_degenerate_non_nullable(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> ArrayRef {
    let len = indices_array.len();

    // SAFETY: list_size is 0, elements array is empty, and both arrays are non-nullable.
    unsafe {
        FixedSizeListArray::new_unchecked(
            array.elements().clone(),
            array.list_size(),
            Validity::NonNullable,
            len,
        )
    }
    .into_array()
}

/// Returns empty lists with computed validity for the nullable case.
fn take_degenerate_nullable<I: NativePType>(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> ArrayRef {
    let indices: &[I] = indices_array.as_slice::<I>();
    let len = indices.len();

    let array_validity = array.validity_mask();
    let indices_validity = indices_array.validity_mask();

    // Compute the validity for each empty list.
    let mut new_validity_builder = BooleanBufferBuilder::new(len);

    // Check the validity of each indexed position.
    for (idx, data_idx) in indices.iter().enumerate() {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let is_index_valid = indices_validity.value(idx);

        // The list is null if the index is null or the indexed element is null.
        if !is_index_valid || !array_validity.value(data_idx) {
            new_validity_builder.append(false);
        } else {
            new_validity_builder.append(true);
        }
    }

    // At least one input was nullable, so the result is nullable.
    let new_validity = Validity::from(new_validity_builder.finish());
    debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == len));

    // SAFETY: list_size is 0, elements array is empty, and validity has the correct length.
    unsafe {
        FixedSizeListArray::new_unchecked(
            array.elements().clone(),
            array.list_size(),
            new_validity,
            len,
        )
    }
    .into_array()
}

/// Takes from an array when both the array and indices are non-nullable.
fn take_non_nullable_fsl<I: NativePType>(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let len = indices.len();

    // Build the element indices directly without validity tracking.
    let mut elements_indices =
        PrimitiveBuilder::<I>::with_capacity(Nullability::NonNullable, len * list_size);

    // Build the element indices for each list.
    for data_idx in indices {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let list_start = data_idx * list_size;
        let list_end = (data_idx + 1) * list_size;

        // Expand the list into individual element indices.
        for i in list_start..list_end {
            elements_indices.append_value(I::from_usize(i).vortex_expect("i < list_end"))
        }
    }

    let elements_indices = elements_indices.finish();
    debug_assert_eq!(elements_indices.len(), len * list_size);

    let new_elements = compute::take(array.elements(), elements_indices.as_ref())?;
    debug_assert_eq!(new_elements.len(), len * list_size);

    // Both inputs are non-nullable, so the result is non-nullable.
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(
            new_elements,
            array.list_size(),
            Validity::NonNullable,
            len,
        )
    }
    .into_array())
}

/// Takes from an array when either the array or indices are nullable.
fn take_nullable_fsl<I: NativePType>(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let len = indices.len();

    let array_validity = array.validity_mask();
    let indices_validity = indices_array.validity_mask();

    // We must use placeholder zeros for null lists to maintain the array length without
    // propagating nullability to the element array's take operation.
    let mut elements_indices =
        PrimitiveBuilder::<I>::with_capacity(Nullability::NonNullable, len * list_size);
    let mut new_validity_builder = BooleanBufferBuilder::new(len);

    // Build the element indices while tracking which lists are null.
    for (idx, data_idx) in indices.iter().enumerate() {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let is_index_valid = indices_validity.value(idx);

        // The list is null if the index is null or the indexed element is null.
        if !is_index_valid || !array_validity.value(data_idx) {
            // Append placeholder zeros for null lists. These will be masked by the validity array.
            // We cannot use append_nulls here as explained above.
            elements_indices.append_zeros(list_size);
            new_validity_builder.append(false);
        } else {
            // Append the actual element indices for this list.
            let list_start = data_idx * list_size;
            let list_end = (data_idx + 1) * list_size;

            // Expand the list into individual element indices.
            for i in list_start..list_end {
                elements_indices.append_value(I::from_usize(i).vortex_expect("i < list_end"))
            }

            new_validity_builder.append(true);
        }
    }

    let elements_indices = elements_indices.finish();
    debug_assert_eq!(elements_indices.len(), len * list_size);

    let new_elements = compute::take(array.elements(), elements_indices.as_ref())?;
    debug_assert_eq!(new_elements.len(), len * list_size);

    // At least one input was nullable, so the result is nullable.
    let new_validity = Validity::from(new_validity_builder.finish());
    debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == len));

    Ok(unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, len)
    }
    .into_array())
}
