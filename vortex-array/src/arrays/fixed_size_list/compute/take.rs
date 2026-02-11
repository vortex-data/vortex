// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_dtype::IntegerPType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_integer_ptype;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::TakeExecute;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

/// Take implementation for [`FixedSizeListArray`].
///
/// Unlike `ListView`, `FixedSizeListArray` must rebuild the elements array because it requires
/// that elements start at offset 0 and be perfectly packed without gaps. We expand list indices
/// into element indices and push them down to the child elements array.
impl TakeExecute for FixedSizeListVTable {
    fn take(
        array: &FixedSizeListArray,
        indices: &dyn Array,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match_each_integer_ptype!(indices.dtype().as_ptype(), |I| {
            take_with_indices::<I>(array, indices)
        })
        .map(Some)
    }
}

/// Dispatches to the appropriate take implementation based on list size and nullability.
fn take_with_indices<I: IntegerPType>(
    array: &FixedSizeListArray,
    indices: &dyn Array,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;

    let indices_array = indices.to_primitive();

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
            take_nullable_fsl::<I>(array, &indices_array)
        } else {
            take_non_nullable_fsl::<I>(array, &indices_array)
        }
    }
}

/// Takes from an array when both the array and indices are non-nullable.
fn take_non_nullable_fsl<I: IntegerPType>(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();

    // Fast path: if elements are already a non-nullable PrimitiveArray, copy contiguous chunks
    // directly from the buffer instead of expanding indices and doing per-element gather.
    if let Some(elements_prim) = array.elements().as_opt::<PrimitiveVTable>()
        && !elements_prim.dtype().is_nullable()
    {
        return match_each_native_ptype!(elements_prim.ptype(), |T| {
            let src: &[T] = elements_prim.as_slice::<T>();
            let taken = take_contiguous_chunks::<T, I>(src, indices, list_size);
            let taken_array = PrimitiveArray::new(taken, Validity::NonNullable);

            Ok(unsafe {
                FixedSizeListArray::new_unchecked(
                    taken_array.into_array(),
                    array.list_size(),
                    Validity::NonNullable,
                    new_len,
                )
            }
            .into_array())
        });
    }

    // Fallback: build expanded element indices and delegate to the child array's take.
    let mut elements_indices = BufferMut::<I>::with_capacity(new_len * list_size);

    for data_idx in indices {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let list_start = data_idx * list_size;
        let list_end = (data_idx + 1) * list_size;

        for i in list_start..list_end {
            // SAFETY: We've allocated enough space for enough indices for all `new_len` lists (that each consist of `list_size = list_end - list_start` elements), so we know we have enough capacity.
            unsafe {
                elements_indices.push_unchecked(I::from_usize(i).vortex_expect("i < list_end"))
            };
        }
    }

    let elements_indices = elements_indices.freeze();
    debug_assert_eq!(elements_indices.len(), new_len * list_size);

    let elements_indices_array = PrimitiveArray::new(elements_indices, Validity::NonNullable);
    let new_elements = array.elements().take(elements_indices_array.to_array())?;
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
fn take_nullable_fsl<I: IntegerPType>(
    array: &FixedSizeListArray,
    indices_array: &PrimitiveArray,
) -> VortexResult<ArrayRef> {
    let list_size = array.list_size() as usize;
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();

    let array_validity = array.validity_mask()?;
    let indices_validity = indices_array.validity_mask()?;

    // Fast path: if elements are already a non-nullable PrimitiveArray, copy contiguous chunks
    // directly instead of expanding indices.
    if let Some(elements_prim) = array.elements().as_opt::<PrimitiveVTable>()
        && !elements_prim.dtype().is_nullable()
    {
        return match_each_native_ptype!(elements_prim.ptype(), |T| {
            take_nullable_fsl_contiguous::<T, I>(
                elements_prim.as_slice::<T>(),
                indices,
                list_size,
                new_len,
                array.list_size(),
                &array_validity,
                &indices_validity,
            )
        });
    }

    // Fallback: build expanded element indices with placeholder zeros for null lists.
    let mut elements_indices = BufferMut::<I>::with_capacity(new_len * list_size);
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
            unsafe { elements_indices.push_n_unchecked(I::zero(), list_size) };
            new_validity_builder.append(false);
        } else {
            // Append the actual element indices for this list.
            let list_start = data_idx * list_size;
            let list_end = (data_idx + 1) * list_size;

            // Expand the list into individual element indices.
            for i in list_start..list_end {
                // SAFETY: We've allocated enough space for enough indices for all `new_len` lists (that each consist of `list_size = list_end - list_start` elements), so we know we have enough capacity.
                unsafe {
                    elements_indices.push_unchecked(I::from_usize(i).vortex_expect("i < list_end"))
                };
            }

            new_validity_builder.append(true);
        }
    }

    let elements_indices = elements_indices.freeze();
    debug_assert_eq!(elements_indices.len(), new_len * list_size);

    let elements_indices_array = PrimitiveArray::new(elements_indices, Validity::NonNullable);
    let new_elements = array.elements().take(elements_indices_array.to_array())?;
    debug_assert_eq!(new_elements.len(), new_len * list_size);

    // At least one input was nullable, so the result is nullable.
    let new_validity = Validity::from(new_validity_builder.freeze());
    debug_assert!(new_validity.maybe_len().is_none_or(|vl| vl == new_len));

    Ok(unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, new_len)
    }
    .into_array())
}

/// Contiguous chunk copy for the nullable FSL take fast path.
///
/// Copies element data directly from the primitive buffer, using position 0 as
/// a placeholder for null lists.
#[allow(clippy::too_many_arguments)]
fn take_nullable_fsl_contiguous<T: NativePType, I: IntegerPType>(
    src: &[T],
    indices: &[I],
    list_size: usize,
    new_len: usize,
    list_size_u32: u32,
    array_validity: &Mask,
    indices_validity: &Mask,
) -> VortexResult<ArrayRef> {
    let total = new_len * list_size;
    let mut result: BufferMut<T> = BufferMut::with_capacity(total);
    let dst_ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();
    let mut new_validity_builder = BitBufferMut::with_capacity(new_len);

    for (i, data_idx) in indices.iter().enumerate() {
        let data_idx_usize = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));
        let is_index_valid = indices_validity.value(i);

        if !is_index_valid || !array_validity.value(data_idx_usize) {
            // Null list: copy from position 0 as placeholder (masked by validity).
            // SAFETY: `list_size > 0` implies elements buffer is non-empty.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr.add(i * list_size), list_size);
            }
            new_validity_builder.append(false);
        } else {
            let src_start = data_idx_usize * list_size;
            // SAFETY: `data_idx_usize` is a valid list index, so
            // `src_start + list_size <= src.len()`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_start),
                    dst_ptr.add(i * list_size),
                    list_size,
                );
            }
            new_validity_builder.append(true);
        }
    }

    // SAFETY: We wrote exactly `total` elements.
    unsafe { result.set_len(total) };
    let taken_buf = result.freeze();
    let taken_array = PrimitiveArray::new(taken_buf, Validity::NonNullable);
    let new_validity = Validity::from(new_validity_builder.freeze());

    Ok(unsafe {
        FixedSizeListArray::new_unchecked(
            taken_array.into_array(),
            list_size_u32,
            new_validity,
            new_len,
        )
    }
    .into_array())
}

/// Copies contiguous chunks of `list_size` elements from a typed source slice.
///
/// For each list index, copies a contiguous range of `list_size` elements using
/// `copy_nonoverlapping` instead of per-element gather. This avoids both the expanded
/// index allocation and the per-element random access overhead.
fn take_contiguous_chunks<T: NativePType, I: IntegerPType>(
    src: &[T],
    list_indices: &[I],
    list_size: usize,
) -> vortex_buffer::Buffer<T> {
    let total = list_indices.len() * list_size;
    let mut result = BufferMut::with_capacity(total);
    let dst_ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    for (i, idx) in list_indices.iter().enumerate() {
        let src_start = idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", idx))
            * list_size;
        // SAFETY:
        // - `src` has length `n * list_size` (FSL invariant) and `src_start + list_size` is
        //   within bounds because `idx` is a valid list index.
        // - `dst` has capacity for `total` elements and we write at non-overlapping offsets.
        // - Source and destination don't overlap (destination is a fresh allocation).
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr().add(src_start),
                dst_ptr.add(i * list_size),
                list_size,
            );
        }
    }

    // SAFETY: We wrote exactly `total` elements.
    unsafe { result.set_len(total) };
    result.freeze()
}
