// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::small_take_rank_lookup_len;
use crate::arrays::PrimitiveArray;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;

#[inline]
pub(super) fn translate_nullable_indices(
    filter: &Mask,
    indices: &PrimitiveArray,
    indices_validity: &BitBuffer,
) -> VortexResult<Buffer<u64>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        translate_nullable_ranks(filter, indices.as_slice::<P>(), indices_validity)
    })
}

fn translate_nullable_ranks<P: IntegerPType>(
    filter: &Mask,
    ranks: &[P],
    indices_validity: &BitBuffer,
) -> VortexResult<Buffer<u64>> {
    if let Some(start) = contiguous_filter_start(filter) {
        let filtered_len = filter.true_count();
        return translate_nullable_ranks_with_offset(ranks, indices_validity, filtered_len, start);
    }

    if ranks.len() <= small_take_rank_lookup_len(filter) {
        return translate_nullable_ranks_with_mask_rank(filter, ranks, indices_validity);
    }

    let filtered_len = filter.true_count();
    match filter.indices() {
        AllOr::All => translate_nullable_ranks_no_filter(ranks, indices_validity, filtered_len),
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(filter_indices) => translate_nullable_ranks_with_indices(
            ranks,
            indices_validity,
            filtered_len,
            filter_indices,
        ),
    }
}

fn translate_nullable_ranks_with_mask_rank<P: IntegerPType>(
    filter: &Mask,
    ranks: &[P],
    indices_validity: &BitBuffer,
) -> VortexResult<Buffer<u64>> {
    let filtered_len = filter.true_count();
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            u64::try_from(filter.rank(rank))?
        } else {
            0
        };

        // SAFETY: `translated` has capacity for all ranks and this loop initializes each
        // output slot once.
        unsafe { translated_ptr.add(idx).write(translated_rank) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

fn translate_nullable_ranks_with_offset<P: IntegerPType>(
    ranks: &[P],
    indices_validity: &BitBuffer,
    filtered_len: usize,
    start: usize,
) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            u64::try_from(start + rank)?
        } else {
            0
        };

        // SAFETY: `translated` has capacity for all ranks and this loop initializes each
        // output slot once.
        unsafe { translated_ptr.add(idx).write(translated_rank) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

fn translate_nullable_ranks_no_filter<P: IntegerPType>(
    ranks: &[P],
    indices_validity: &BitBuffer,
    filtered_len: usize,
) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = if indices_validity.value(idx) {
            u64::try_from(validate_rank(*rank, filtered_len)?)?
        } else {
            0
        };

        // SAFETY: `translated` has capacity for all ranks and this loop initializes each
        // output slot once.
        unsafe { (*translated_ptr.add(idx)).write(translated_rank) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

fn translate_nullable_ranks_with_indices<P: IntegerPType>(
    ranks: &[P],
    indices_validity: &BitBuffer,
    filtered_len: usize,
    filter_indices: &[usize],
) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = if indices_validity.value(idx) {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `rank` was checked against the filtered length, so it is in bounds for
            // `filter_indices`; filter indices are valid child positions by construction.
            unsafe { u64::try_from(*filter_indices.get_unchecked(rank))? }
        } else {
            0
        };

        // SAFETY: `translated` has capacity for all ranks and this loop initializes each
        // output slot once.
        unsafe { translated_ptr.add(idx).write(translated_rank) };
    }

    // SAFETY: The loop writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

#[inline]
pub(super) fn translate_indices(
    filter: &Mask,
    indices: &PrimitiveArray,
) -> VortexResult<Buffer<u64>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        translate_ranks(filter, indices.as_slice::<P>())
    })
}

#[inline]
pub(super) fn contiguous_sequential_take_range_indices(
    filter: &Mask,
    indices: &PrimitiveArray,
) -> VortexResult<Option<(usize, usize)>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        contiguous_sequential_take_range(filter, indices.as_slice::<P>())
    })
}

#[inline]
pub(super) fn sequential_take_len(
    indices: &PrimitiveArray,
    filtered_len: usize,
) -> VortexResult<Option<usize>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        sequential_take_len_typed(indices.as_slice::<P>(), filtered_len)
    })
}

#[inline]
fn sequential_take_len_typed<P: IntegerPType>(
    ranks: &[P],
    filtered_len: usize,
) -> VortexResult<Option<usize>> {
    for (idx, rank) in ranks.iter().enumerate() {
        if rank.as_() != idx {
            return Ok(None);
        }
    }

    if ranks.len() > filtered_len {
        vortex_bail!(OutOfBounds: ranks.len() - 1, 0, filtered_len);
    }

    Ok(Some(ranks.len()))
}

fn translate_ranks<P: IntegerPType>(filter: &Mask, ranks: &[P]) -> VortexResult<Buffer<u64>> {
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();
    let filtered_len = filter.true_count();

    if let Some(start) = contiguous_filter_start(filter) {
        for (idx, rank) in ranks.iter().enumerate() {
            let rank = validate_rank(*rank, filtered_len)?;
            // SAFETY: `translated` has capacity for all ranks and this loop initializes each
            // output slot once.
            unsafe { translated_ptr.add(idx).write(u64::try_from(start + rank)?) };
        }
    } else if ranks.len() <= small_take_rank_lookup_len(filter) {
        for (idx, rank) in ranks.iter().enumerate() {
            let rank = validate_rank(*rank, filtered_len)?;
            let child_idx = filter.rank(rank);
            // SAFETY: `translated` has capacity for all ranks and this loop initializes each
            // output slot once.
            unsafe { translated_ptr.add(idx).write(u64::try_from(child_idx)?) };
        }
    } else {
        match filter.indices() {
            AllOr::All => {
                for (idx, rank) in ranks.iter().enumerate() {
                    let rank = validate_rank(*rank, filtered_len)?;
                    // SAFETY: `translated` has capacity for all ranks and this loop initializes
                    // each output slot once.
                    unsafe { translated_ptr.add(idx).write(u64::try_from(rank)?) };
                }

                // SAFETY: The loop writes exactly `ranks.len()` initialized values.
                unsafe { translated.set_len(ranks.len()) };
                return Ok(translated.freeze());
            }
            AllOr::None => unreachable!("empty filters are handled by take preconditions"),
            AllOr::Some(filter_indices) => {
                for (idx, rank) in ranks.iter().enumerate() {
                    let rank = validate_rank(*rank, filtered_len)?;
                    // SAFETY: `translated` has capacity for all ranks. `rank` was checked against the
                    // filtered length, and filter indices are valid child positions by construction.
                    unsafe {
                        translated_ptr
                            .add(idx)
                            .write(u64::try_from(*filter_indices.get_unchecked(rank))?)
                    };
                }
            }
        }
    }

    // SAFETY: Each loop path writes exactly `ranks.len()` initialized values.
    unsafe { translated.set_len(ranks.len()) };
    Ok(translated.freeze())
}

#[inline]
pub(super) fn validate_rank<P: IntegerPType>(rank: P, filtered_len: usize) -> VortexResult<usize> {
    let rank: usize = rank.as_();
    if rank >= filtered_len {
        vortex_bail!(OutOfBounds: rank, 0, filtered_len);
    }
    Ok(rank)
}

#[inline]
pub(super) fn contiguous_sequential_take_range<P: IntegerPType>(
    filter: &Mask,
    ranks: &[P],
) -> VortexResult<Option<(usize, usize)>> {
    let Some(start) = contiguous_filter_start(filter) else {
        return Ok(None);
    };

    let Some(take_len) = sequential_take_len_typed(ranks, filter.true_count())? else {
        return Ok(None);
    };

    Ok(Some((start, start + take_len)))
}

#[inline]
pub(super) fn contiguous_filter_start(filter: &Mask) -> Option<usize> {
    let start = filter.first()?;
    let end = filter.last()?.checked_add(1)?;
    (end - start == filter.true_count()).then_some(start)
}
