// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
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
use crate::arrays::filter::FilterArraySlotsExt;
use crate::buffer::BufferHandle;
use crate::dtype::IntegerPType;
use crate::dtype::i256;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

#[inline]
pub(in crate::arrays::filter) fn take_primitive(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child = array.child().clone().execute::<PrimitiveArray>(ctx)?;
    let ptype = child.ptype();
    let (taken, output_validity) = take_fixed_width_bytes(
        child.buffer_handle().to_host_sync(),
        ptype.byte_width(),
        child.validity()?,
        array.filter_mask(),
        indices,
        ctx,
    )?;

    Ok(PrimitiveArray::from_byte_buffer(taken, ptype, output_validity).into_array())
}

#[inline]
pub(in crate::arrays::filter) fn take_decimal(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child = array.child().clone().execute::<DecimalArray>(ctx)?;
    let values_type = child.values_type();
    let (taken, output_validity) = take_fixed_width_bytes(
        child.buffer_handle().to_host_sync(),
        values_type.byte_width(),
        child.validity()?,
        array.filter_mask(),
        indices,
        ctx,
    )?;

    // SAFETY: Valid ranks copy existing decimal values, null ranks write default placeholders that
    // are hidden by output validity, and the output validity was built for the take length.
    Ok(unsafe {
        DecimalArray::new_unchecked_handle(
            BufferHandle::new_host(taken),
            values_type,
            child.decimal_dtype(),
            output_validity,
        )
    }
    .into_array())
}

/// Takes values of `byte_width` bytes each.
///
/// The take copies values and does not inspect them, so all value types of one width share one
/// kernel.
fn take_fixed_width_bytes(
    child: ByteBuffer,
    byte_width: usize,
    child_validity: Validity,
    filter: &Mask,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ByteBuffer, Validity)> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        match byte_width {
            1 => take_fixed_width_as::<u8, P>(child, child_validity, filter, indices, ctx),
            2 => take_fixed_width_as::<u16, P>(child, child_validity, filter, indices, ctx),
            4 => take_fixed_width_as::<u32, P>(child, child_validity, filter, indices, ctx),
            8 => take_fixed_width_as::<u64, P>(child, child_validity, filter, indices, ctx),
            16 => take_fixed_width_as::<u128, P>(child, child_validity, filter, indices, ctx),
            32 => take_fixed_width_as::<i256, P>(child, child_validity, filter, indices, ctx),
            _ => vortex_panic!("Unsupported fixed width of {byte_width} bytes"),
        }
    })
}

fn take_fixed_width_as<W, P>(
    child: ByteBuffer,
    child_validity: Validity,
    filter: &Mask,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ByteBuffer, Validity)>
where
    W: Copy + Default,
    P: IntegerPType,
{
    let (taken, output_validity) = take_fixed_width::<W, P>(
        Buffer::from_byte_buffer(child),
        child_validity,
        filter,
        indices,
        ctx,
    )?;

    Ok((taken.into_byte_buffer(), output_validity))
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
    let allocator = ctx.allocator();

    match indices_validity.bit_buffer() {
        AllOr::All => {
            let taken = if let Some((start, end)) = contiguous_sequential_take_range(filter, ranks)?
            {
                child.slice(start..end)
            } else {
                take_filtered_values::<T, P>(&child, filter, ranks, None, allocator)?
            };

            let output_validity = if child_validity.definitely_no_nulls() {
                ranks_validity.union_nullability(child_validity.nullability())
            } else {
                let translated_indices = PrimitiveArray::new(
                    translate_ranks(filter, ranks, None, allocator)?,
                    ranks_validity,
                )
                .into_array();
                child_validity.take(&translated_indices)?
            };

            Ok((taken, output_validity))
        }
        AllOr::None => Ok((
            Buffer::zeroed_in(ranks.len(), allocator.clone()),
            Validity::AllInvalid,
        )),
        AllOr::Some(buf) => {
            let taken =
                take_filtered_values(child.as_slice(), filter, ranks, Some(buf), allocator)?;

            let output_validity = if child_validity.definitely_no_nulls() {
                ranks_validity.union_nullability(child_validity.nullability())
            } else {
                let translated_indices = PrimitiveArray::new(
                    translate_ranks(filter, ranks, Some(buf), allocator)?,
                    ranks_validity,
                )
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
    allocator: &BufferAllocatorRef,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
{
    let filtered_len = filter.true_count();
    if let Some(start) = contiguous_filter_start(filter) {
        return if let Some(indices_validity) = indices_validity {
            take_values_by_rank_nullable(
                values,
                ranks,
                indices_validity,
                filtered_len,
                allocator,
                |idx| start + idx,
            )
        } else {
            take_values_by_rank(values, ranks, filtered_len, allocator, |idx| start + idx)
        };
    }

    if ranks.len() <= small_take_rank_lookup_len(filter) {
        return if let Some(indices_validity) = indices_validity {
            take_values_by_rank_nullable(
                values,
                ranks,
                indices_validity,
                filtered_len,
                allocator,
                |idx| filter.rank(idx),
            )
        } else {
            take_values_by_rank(values, ranks, filtered_len, allocator, |idx| {
                filter.rank(idx)
            })
        };
    }

    match filter.indices() {
        AllOr::All => {
            if let Some(indices_validity) = indices_validity {
                take_values_by_rank_nullable(
                    values,
                    ranks,
                    indices_validity,
                    filtered_len,
                    allocator,
                    |idx| idx,
                )
            } else {
                take_values_by_rank(values, ranks, filtered_len, allocator, |idx| idx)
            }
        }
        AllOr::None => unreachable!("empty filters are handled by the filter short circuit"),
        AllOr::Some(indices) => {
            if let Some(indices_validity) = indices_validity {
                take_values_by_rank_nullable(
                    values,
                    ranks,
                    indices_validity,
                    filtered_len,
                    allocator,
                    |idx| unsafe { *indices.get_unchecked(idx) },
                )
            } else {
                take_values_by_rank(values, ranks, filtered_len, allocator, |idx| unsafe {
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
    allocator: &BufferAllocatorRef,
    translate: L,
) -> VortexResult<Buffer<T>>
where
    T: Copy + Default,
    P: IntegerPType,
    L: Fn(usize) -> usize,
{
    let mut out = BufferMut::<T>::with_capacity_in(ranks.len(), allocator.clone());
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
    allocator: &BufferAllocatorRef,
    translate: L,
) -> VortexResult<Buffer<T>>
where
    T: Copy,
    P: IntegerPType,
    L: Fn(usize) -> usize,
{
    let mut out = BufferMut::<T>::with_capacity_in(ranks.len(), allocator.clone());
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
