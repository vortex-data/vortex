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
pub(in crate::arrays::filter) fn translate_indices(
    filter: &Mask,
    indices: &PrimitiveArray,
    indices_validity: Option<&BitBuffer>,
) -> VortexResult<Buffer<u64>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        translate_ranks(filter, indices.as_slice::<P>(), indices_validity)
    })
}

#[inline]
pub(in crate::arrays::filter) fn contiguous_sequential_take_range_indices(
    filter: &Mask,
    indices: &PrimitiveArray,
) -> VortexResult<Option<(usize, usize)>> {
    match_each_integer_ptype!(indices.ptype(), |P| {
        contiguous_sequential_take_range(filter, indices.as_slice::<P>())
    })
}

#[inline]
pub(in crate::arrays::filter) fn sequential_take_len(
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

pub(in crate::arrays::filter) fn translate_ranks<P: IntegerPType>(
    filter: &Mask,
    ranks: &[P],
    ranks_validity: Option<&BitBuffer>,
) -> VortexResult<Buffer<u64>> {
    let filtered_len = filter.true_count();

    if let Some(start) = contiguous_filter_start(filter) {
        return translate_ranks_with(ranks, ranks_validity, filtered_len, |rank| start + rank);
    }

    if ranks.len() <= small_take_rank_lookup_len(filter) {
        return translate_ranks_with(ranks, ranks_validity, filtered_len, |rank| {
            filter.rank(rank)
        });
    }

    match filter.indices() {
        AllOr::All => translate_ranks_with(ranks, ranks_validity, filtered_len, |rank| rank),
        AllOr::None => unreachable!("empty filters are handled by take preconditions"),
        AllOr::Some(filter_indices) => {
            translate_ranks_with(ranks, ranks_validity, filtered_len, |rank| unsafe {
                *filter_indices.get_unchecked(rank)
            })
        }
    }
}

fn translate_ranks_with<P, L>(
    ranks: &[P],
    ranks_validity: Option<&BitBuffer>,
    filtered_len: usize,
    translate: L,
) -> VortexResult<Buffer<u64>>
where
    P: IntegerPType,
    L: Fn(usize) -> usize,
{
    let mut translated = BufferMut::<u64>::with_capacity(ranks.len());
    let translated_ptr = translated.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    for (idx, rank) in ranks.iter().enumerate() {
        let translated_rank = match ranks_validity {
            Some(validity) if !validity.value(idx) => 0,
            _ => {
                let rank = validate_rank(*rank, filtered_len)?;
                u64::try_from(translate(rank))?
            }
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
pub(in crate::arrays::filter) fn validate_rank<P: IntegerPType>(
    rank: P,
    filtered_len: usize,
) -> VortexResult<usize> {
    let rank: usize = rank.as_();
    if rank >= filtered_len {
        vortex_bail!(OutOfBounds: rank, 0, filtered_len);
    }
    Ok(rank)
}

#[inline]
pub(in crate::arrays::filter) fn contiguous_sequential_take_range<P: IntegerPType>(
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
pub(in crate::arrays::filter) fn contiguous_filter_start(filter: &Mask) -> Option<usize> {
    let start = filter.first()?;
    let end = filter.last()?.checked_add(1)?;
    (end - start == filter.true_count()).then_some(start)
}
