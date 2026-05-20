// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::rank::contiguous_filter_start;
use super::rank::contiguous_sequential_take_range;
use super::rank::translate_ranks;
use super::rank::validate_rank;
use super::small_take_rank_lookup_len;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::DecimalArray;
use crate::arrays::Filter;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::filter::FilterArrayExt;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

#[inline]
pub(in crate::arrays::filter) fn take_primitive(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child = array.child().clone().execute::<PrimitiveArray>(ctx)?;
    let child_validity = child.validity()?;
    match_each_native_ptype!(child.ptype(), |T| {
        let child_buf = child.to_buffer::<T>();
        match_each_integer_ptype!(indices.ptype(), |P| {
            let (taken, output_validity) = take_fixed_width::<T, P>(
                child_buf,
                child_validity,
                array.filter_mask(),
                indices,
                ctx,
            )?;
            // SAFETY: Take operation validated all the parts
            Ok(unsafe { PrimitiveArray::new_unchecked(taken, output_validity) }.into_array())
        })
    })
}

#[inline]
pub(in crate::arrays::filter) fn take_decimal(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child = array.child().clone().execute::<DecimalArray>(ctx)?;
    let decimal_dtype = child.decimal_dtype();
    let child_validity = child.validity()?;
    match_each_decimal_value_type!(child.values_type(), |T| {
        let child_buf = child.buffer::<T>();
        match_each_integer_ptype!(indices.ptype(), |P| {
            let (taken, output_validity) = take_fixed_width::<T, P>(
                child_buf,
                child_validity,
                array.filter_mask(),
                indices,
                ctx,
            )?;
            // SAFETY: Valid ranks copy existing decimal values, null ranks write default placeholders that
            // are hidden by output validity, and the output validity was built for the take length.
            Ok(
                unsafe { DecimalArray::new_unchecked(taken, decimal_dtype, output_validity) }
                    .into_array(),
            )
        })
    })
}

fn take_fixed_width<T, P>(
    child: Buffer<T>,
    child_validity: Validity,
    filter: &Mask,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Buffer<T>, Validity)>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let ranks = indices.as_slice::<P>();
    let ranks_validity = indices.validity()?;
    let indices_validity = ranks_validity.execute_mask(indices.len(), ctx)?;

    match indices_validity.bit_buffer() {
        AllOr::All => {
            let taken = if let Some((start, end)) = contiguous_sequential_take_range(filter, ranks)?
            {
                child.slice(start..end)
            } else {
                take_filtered_values::<T, P>(&child, filter, ranks, None)?
            };

            let output_validity = if child_validity.no_nulls() {
                ranks_validity
            } else {
                let translated_indices =
                    PrimitiveArray::new(translate_ranks(filter, ranks, None)?, ranks_validity)
                        .into_array();
                child_validity.take(&translated_indices)?
            };

            Ok((taken, output_validity))
        }
        AllOr::None => Ok((Buffer::zeroed(ranks.len()), Validity::AllInvalid)),
        AllOr::Some(buf) => {
            let taken = take_filtered_values(child.as_slice(), filter, ranks, Some(buf))?;

            let output_validity = if child_validity.no_nulls() {
                ranks_validity
            } else {
                let translated_indices =
                    PrimitiveArray::new(translate_ranks(filter, ranks, Some(buf))?, ranks_validity)
                        .into_array();
                child_validity.take(&translated_indices)?
            };

            Ok((taken, output_validity))
        }
    }
}

fn take_filtered_values<T, P>(
    values: &[T],
    filter: &Mask,
    ranks: &[P],
    indices_validity: Option<&BitBuffer>,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let filtered_len = filter.true_count();
    if let Some(start) = contiguous_filter_start(filter) {
        return if let Some(indices_validity) = indices_validity {
            take_values_by_rank_nullable(values, ranks, indices_validity, filtered_len, |idx| {
                start + idx
            })
        } else {
            take_values_by_rank(values, ranks, filtered_len, |idx| start + idx)
        };
    }

    if ranks.len() <= small_take_rank_lookup_len(filter) {
        return if let Some(indices_validity) = indices_validity {
            take_values_by_rank_nullable(values, ranks, indices_validity, filtered_len, |idx| {
                filter.rank(idx)
            })
        } else {
            take_values_by_rank(values, ranks, filtered_len, |idx| filter.rank(idx))
        };
    }

    match filter.indices() {
        AllOr::All => {
            if let Some(indices_validity) = indices_validity {
                take_values_by_rank_nullable(values, ranks, indices_validity, filtered_len, |idx| {
                    idx
                })
            } else {
                take_values_by_rank(values, ranks, filtered_len, |idx| idx)
            }
        }
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(indices) => {
            if let Some(indices_validity) = indices_validity {
                take_values_by_rank_nullable(
                    values,
                    ranks,
                    indices_validity,
                    filtered_len,
                    |idx| unsafe { *indices.get_unchecked(idx) },
                )
            } else {
                take_values_by_rank(values, ranks, filtered_len, |idx| unsafe {
                    *indices.get_unchecked(idx)
                })
            }
        }
    }
}

fn take_values_by_rank_nullable<T, P, L>(
    values: &[T],
    ranks: &[P],
    ranks_validity: &BitBuffer,
    translated_len: usize,
    translate: L,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
    L: Fn(usize) -> usize,
{
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let value = if ranks_validity.value(idx) {
            let rank = validate_rank(*rank, translated_len)?;
            let child_idx = translate(rank);
            // SAFETY: `rank` was bounds-checked.
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

fn take_values_by_rank<T, P, L>(
    values: &[T],
    ranks: &[P],
    translated_len: usize,
    translate: L,
) -> VortexResult<Buffer<T>>
where
    T: Copy,
    P: IntegerPType,
    L: Fn(usize) -> usize,
{
    let mut out = BufferMut::<T>::with_capacity(ranks.len());
    let out_ptr = out.spare_capacity_mut().as_mut_ptr().cast::<T>();
    for (idx, rank) in ranks.iter().enumerate() {
        let rank = validate_rank(*rank, translated_len)?;
        let child_idx = translate(rank);

        // SAFETY: `out` has capacity for all ranks and `rank` was bounds-checked.
        unsafe { out_ptr.add(idx).write(*values.get_unchecked(child_idx)) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { out.set_len(ranks.len()) };
    Ok(out.freeze())
}
