// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Filter;
use super::rank::contiguous_filter_start;
use super::rank::contiguous_sequential_take_range;
use super::rank::translate_indices;
use super::rank::translate_nullable_indices;
use super::rank::validate_rank;
use super::small_take_rank_lookup_len;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::filter::FilterArrayExt;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

#[inline]
pub(super) fn take_primitive(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child = array.child().clone().execute::<PrimitiveArray>(ctx)?;
    match_each_native_ptype!(child.ptype(), |T| {
        match_each_integer_ptype!(indices.ptype(), |P| {
            take_primitive_typed::<T, P>(child, array.filter_mask(), indices, ctx)
        })
    })
}

#[inline]
pub(super) fn take_decimal(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child = array.child().clone().execute::<DecimalArray>(ctx)?;
    match_each_decimal_value_type!(child.values_type(), |T| {
        match_each_integer_ptype!(indices.ptype(), |P| {
            take_decimal_typed::<T, P>(child, array.filter_mask(), indices, ctx)
        })
    })
}

fn take_decimal_typed<T, P>(
    child: DecimalArray,
    filter: &Mask,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativeDecimalType,
    P: IntegerPType,
{
    let ranks = indices.as_slice::<P>();
    let decimal_dtype = child.decimal_dtype();
    let child_validity = child.validity()?;
    let indices_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;

    let taken = if indices_validity.all_true() {
        if let Some((start, end)) = contiguous_sequential_take_range(filter, ranks)? {
            let values = child.buffer_handle().slice_typed::<T>(start..end);
            let output_validity = contiguous_output_validity(&child_validity, indices, start..end)?;
            // SAFETY: The values are sliced from an existing valid decimal array, and the output
            // validity was built for exactly the sliced take length.
            return Ok(unsafe {
                DecimalArray::new_unchecked_handle(
                    values,
                    T::DECIMAL_TYPE,
                    decimal_dtype,
                    output_validity,
                )
            }
            .into_array());
        }

        take_filtered_values::<T, P>(child.buffer::<T>().as_slice(), filter, ranks)?
    } else {
        take_filtered_values_nullable::<T, P>(
            child.buffer::<T>().as_slice(),
            filter,
            ranks,
            &indices_validity,
        )?
    };
    let output_validity =
        take_output_validity(&child_validity, filter, indices, &indices_validity)?;
    // SAFETY: Valid ranks copy existing decimal values, null ranks write default placeholders that
    // are hidden by output validity, and the output validity was built for the take length.
    Ok(unsafe { DecimalArray::new_unchecked(taken, decimal_dtype, output_validity) }.into_array())
}

fn take_primitive_typed<T, P>(
    child: PrimitiveArray,
    filter: &Mask,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    P: IntegerPType,
{
    let ranks = indices.as_slice::<P>();
    let child_validity = child.validity()?;

    let indices_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
    if indices_validity.all_true() {
        return take_primitive_all_valid::<T, P>(child, filter, indices, ranks, &child_validity);
    }

    let taken = take_filtered_values_nullable::<T, P>(
        child.as_slice::<T>(),
        filter,
        ranks,
        &indices_validity,
    )?;
    let output_validity =
        take_output_validity(&child_validity, filter, indices, &indices_validity)?;

    Ok(PrimitiveArray::new(taken, output_validity).into_array())
}

fn take_primitive_all_valid<T, P>(
    child: PrimitiveArray,
    filter: &Mask,
    indices: &PrimitiveArray,
    ranks: &[P],
    child_validity: &Validity,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
    P: IntegerPType,
{
    if let Some((start, end)) = contiguous_sequential_take_range(filter, ranks)? {
        let output_validity = contiguous_output_validity(child_validity, indices, start..end)?;
        return Ok(PrimitiveArray::from_buffer_handle(
            child.buffer_handle().slice_typed::<T>(start..end),
            T::PTYPE,
            output_validity,
        )
        .into_array());
    }

    let taken = take_filtered_values::<T, P>(child.as_slice::<T>(), filter, ranks)?;
    let output_validity = take_output_validity(
        child_validity,
        filter,
        indices,
        &Mask::new_true(indices.len()),
    )?;
    Ok(PrimitiveArray::new(taken, output_validity).into_array())
}

fn contiguous_output_validity(
    child_validity: &Validity,
    indices: &PrimitiveArray,
    range: std::ops::Range<usize>,
) -> VortexResult<Validity> {
    if child_validity.no_nulls() {
        return indices.validity();
    }

    child_validity.slice(range)
}

fn take_output_validity(
    child_validity: &Validity,
    filter: &Mask,
    indices: &PrimitiveArray,
    indices_validity: &Mask,
) -> VortexResult<Validity> {
    if child_validity.no_nulls() {
        return indices.validity();
    }

    let translated_indices = match indices_validity.bit_buffer() {
        AllOr::All => PrimitiveArray::new(translate_indices(filter, indices)?, indices.validity()?)
            .into_array(),
        AllOr::None => return Ok(Validity::AllInvalid),
        AllOr::Some(b) => PrimitiveArray::new(
            translate_nullable_indices(filter, indices, b)?,
            indices.validity()?,
        )
        .into_array(),
    };

    child_validity.take(&translated_indices)
}

fn take_filtered_values_nullable<T, P>(
    values: &[T],
    filter: &Mask,
    ranks: &[P],
    indices_validity: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    if indices_validity.all_false() {
        return Ok(Buffer::zeroed(ranks.len()));
    }

    if let Some(start) = contiguous_filter_start(filter) {
        let mut out = BufferMut::<T>::with_capacity(ranks.len());
        let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
        for (idx, rank) in ranks.iter().enumerate() {
            let value = if indices_validity.value(idx) {
                let rank = validate_rank(*rank, filter.true_count())?;
                // SAFETY: `rank` was checked against the contiguous filtered length.
                unsafe { *values.get_unchecked(start + rank) }
            } else {
                T::default()
            };

            // SAFETY: `out` has capacity for all ranks and this loop initializes each output slot
            // once.
            unsafe { out_ptr.add(idx).write(value) };
        }

        // SAFETY: The loop writes exactly `ranks.len()` initialized values.
        unsafe { out.set_len(ranks.len()) };
        return Ok(out.freeze());
    }

    if ranks.len() <= small_take_rank_lookup_len(filter) {
        return take_filtered_values_nullable_by_mask_rank(values, filter, ranks, indices_validity);
    }

    let filtered_len = filter.true_count();
    let indices = match filter.indices() {
        AllOr::All => {
            return take_values_by_rank_nullable(values, ranks, indices_validity, filtered_len);
        }
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(indices) => indices,
    };

    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let value = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `rank` was bounds-checked against `indices`, whose values are valid
            // positions in `values`.
            unsafe { *values.get_unchecked(*indices.get_unchecked(rank)) }
        } else {
            T::default()
        };

        // SAFETY: `out` has capacity for all ranks and this loop initializes each output slot
        // once.
        unsafe { out_ptr.add(idx).write(value) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn take_filtered_values_nullable_by_mask_rank<T, P>(
    values: &[T],
    filter: &Mask,
    ranks: &[P],
    indices_validity: &Mask,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let filtered_len = filter.true_count();
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let value = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            let child_idx = filter.rank(rank);
            // SAFETY: `validate_rank` checked the rank against the filter's true count.
            unsafe { *values.get_unchecked(child_idx) }
        } else {
            T::default()
        };

        // SAFETY: `out` has capacity for all ranks and this loop initializes each output slot
        // once.
        unsafe { out_ptr.add(idx).write(value) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn take_values_by_rank_nullable<T, P>(
    values: &[T],
    ranks: &[P],
    indices_validity: &Mask,
    filtered_len: usize,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let value = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `rank` was bounds-checked.
            unsafe { *values.get_unchecked(rank) }
        } else {
            T::default()
        };

        // SAFETY: `out` has capacity for all ranks and this loop initializes each output slot
        // once.
        unsafe { out_ptr.add(idx).write(value) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn take_filtered_values<T, P>(values: &[T], filter: &Mask, ranks: &[P]) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let filtered_len = filter.true_count();

    if let Some(start) = contiguous_filter_start(filter) {
        let mut out = BufferMut::<T>::with_capacity(ranks.len());
        let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
        for (idx, rank) in ranks.iter().enumerate() {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `out` has capacity for all ranks. The filter is contiguous with
            // `filtered_len` values starting at `start`, and `rank` was checked above.
            unsafe { out_ptr.add(idx).write(*values.get_unchecked(start + rank)) };
        }

        // SAFETY: The loop writes exactly `ranks.len()` initialized values.
        unsafe { out.set_len(ranks.len()) };
        return Ok(out.freeze());
    }

    if ranks.len() <= small_take_rank_lookup_len(filter) {
        return take_filtered_values_by_mask_rank(values, filter, ranks);
    }

    let indices = match filter.indices() {
        AllOr::All => return take_values_by_rank(values, ranks),
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(indices) => indices,
    };

    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let rank = validate_rank(*rank, filtered_len)?;
        // SAFETY: `out` has capacity for all ranks. `rank` was bounds-checked against
        // `indices`, whose values are valid positions in `values`.
        unsafe {
            out_ptr
                .add(idx)
                .write(*values.get_unchecked(*indices.get_unchecked(rank)))
        };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn take_filtered_values_by_mask_rank<T, P>(
    values: &[T],
    filter: &Mask,
    ranks: &[P],
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let filtered_len = filter.true_count();
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let rank = validate_rank(*rank, filtered_len)?;
        let child_idx = filter.rank(rank);

        // SAFETY: `out` has capacity for all ranks, and `validate_rank` checked the rank against
        // the filter's true count.
        unsafe { out_ptr.add(idx).write(*values.get_unchecked(child_idx)) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}

fn take_values_by_rank<T, P>(values: &[T], ranks: &[P]) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let rank = validate_rank(*rank, values.len())?;

        // SAFETY: `out` has capacity for all ranks and `rank` was bounds-checked.
        unsafe { out_ptr.add(idx).write(*values.get_unchecked(rank)) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}
