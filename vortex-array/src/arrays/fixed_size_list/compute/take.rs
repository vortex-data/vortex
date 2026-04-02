// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_smallest_offset_type;
use crate::validity::Validity;

/// Take implementation for [`FixedSizeListArray`].
///
/// Unlike `ListView`, `FixedSizeListArray` must rebuild the elements array because it requires
/// that elements start at offset 0 and be perfectly packed without gaps. We expand list indices
/// into element indices and push them down to the child elements array.
impl TakeExecute for FixedSizeList {
    #[expect(clippy::cognitive_complexity)]
    fn take(
        array: ArrayView<'_, FixedSizeList>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let max_element_idx = array.elements().len();
        match_each_integer_ptype!(indices.dtype().as_ptype(), |I| {
            match_smallest_offset_type!(max_element_idx, |E| {
                take_with_indices::<I, E>(array, indices, ctx)
            })
        })
        .map(Some)
    }
}

/// Dispatches to the appropriate take implementation based on list size and nullability.
fn take_with_indices<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;

    let indices_array = indices.clone().execute::<PrimitiveArray>(ctx)?;

    // Make sure to handle degenerate case where lists have size 0 (these can take fast paths).
    if list_size == 0 {
        debug_assert!(
            array.elements().is_empty(),
            "degenerate list must have empty elements"
        );

        // Since there are no elements to take, we just need to take on the validity map.
        let new_validity = array.validity().take(indices)?;
        let new_len = indices_array.len();

        Ok(
            // SAFETY: list_size is 0, elements array is empty, and validity has the correct length.
            unsafe {
                FixedSizeListArray::new_unchecked(
                    array.elements().clone(), // Remember that this is an empty array.
                    array.list_size(),
                    new_validity,
                    new_len,
                )
            }
            .into_array(),
        )
    } else {
        // The result's nullability is the union of the input nullabilities.
        if array.dtype().is_nullable() || indices_array.dtype().is_nullable() {
            let indices_array = indices_array.as_view();
            take_nullable_fsl::<I, E>(array, indices_array)
        } else {
            let indices_array = indices_array.as_view();
            take_non_nullable_fsl::<I, E>(array, indices_array)
        }
    }
}

/// Takes from an array when both the array and indices are non-nullable.
fn take_non_nullable_fsl<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();

    // Build the element indices directly without validity tracking.
    let mut elements_indices = BufferMut::<E>::with_capacity(new_len * list_size);

    // Build the element indices for each list.
    for data_idx in indices {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let list_start = data_idx * list_size;
        let list_end = (data_idx + 1) * list_size;

        // Expand the list into individual element indices.
        for i in list_start..list_end {
            // SAFETY: We've allocated enough space for enough indices for all `new_len` lists (that each consist of `list_size = list_end - list_start` elements), so we know we have enough capacity.
            unsafe {
                elements_indices.push_unchecked(E::from_usize(i).vortex_expect("i < list_end"))
            };
        }
    }

    let elements_indices = elements_indices.freeze();
    debug_assert_eq!(elements_indices.len(), new_len * list_size);

    let elements_indices_array = PrimitiveArray::new(elements_indices, Validity::NonNullable);
    let new_elements = array.elements().take(elements_indices_array.into_array())?;
    debug_assert_eq!(new_elements.len(), new_len * list_size);

    // Both inputs are non-nullable, so the result is non-nullable.
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(
            new_elements,
            array.list_size(),
            Validity::NonNullable,
            new_len,
        )
    }
    .into_array())
}

/// Takes from an array when either the array or indices are nullable.
fn take_nullable_fsl<I: IntegerPType, E: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();

    let array_validity = array.validity_mask();
    let indices_validity = indices_array.validity_mask();

    // We must use placeholder zeros for null lists to maintain the array length without
    // propagating nullability to the element array's take operation.
    let mut elements_indices = BufferMut::<E>::with_capacity(new_len * list_size);
    let mut new_validity_builder = BitBufferMut::with_capacity(new_len);

    // Build the element indices while tracking which lists are null.
    for (i, data_idx) in indices.iter().enumerate() {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let is_index_valid = indices_validity.value(i);

        // The list is null if the index is null or the indexed element is null.
        if !is_index_valid || !array_validity.value(data_idx) {
            // Append placeholder zeros for null lists. These will be masked by the validity array.
            // We cannot use append_nulls here as explained above.
            unsafe { elements_indices.push_n_unchecked(E::zero(), list_size) };
            new_validity_builder.append(false);
        } else {
            // Append the actual element indices for this list.
            let list_start = data_idx * list_size;
            let list_end = (data_idx + 1) * list_size;

            // Expand the list into individual element indices.
            for i in list_start..list_end {
                // SAFETY: We've allocated enough space for enough indices for all `new_len` lists (that each consist of `list_size = list_end - list_start` elements), so we know we have enough capacity.
                unsafe {
                    elements_indices.push_unchecked(E::from_usize(i).vortex_expect("i < list_end"))
                };
            }

            new_validity_builder.append(true);
        }
    }

    let elements_indices = elements_indices.freeze();
    debug_assert_eq!(elements_indices.len(), new_len * list_size);

    let elements_indices_array = PrimitiveArray::new(elements_indices, Validity::NonNullable);
    let new_elements = array.elements().take(elements_indices_array.into_array())?;
    debug_assert_eq!(new_elements.len(), new_len * list_size);

    // At least one input was nullable, so the result is nullable.
    let new_validity = Validity::from(new_validity_builder.freeze());
    debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == new_len));

    Ok(unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, new_len)
    }
    .into_array())
}
