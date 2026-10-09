// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::match_each_integer_ptype;

/// Below this many values, decoding all of them costs less than any take kernel's setup.
const MIN_VALUES_LEN: usize = 1024;

/// A take with fewer indices than `1 / SPARSE_TAKE_DENOMINATOR` of the values gathers the encoded
/// values: uniformly random indices that sparse repeat too rarely to pay for finding the
/// referenced values.
const SPARSE_TAKE_DENOMINATOR: usize = 2;

/// With at least this many indices per value, all but about `e^-4` of the values are referenced
/// for uniformly random indices, so finding which is wasted work.
const DENSE_INDICES_PER_VALUE: usize = 4;

/// Once at least `1 / DENSE_REFERENCED_DENOMINATOR` of the values are referenced, filtering the
/// encoded values costs more than decoding the unreferenced ones.
const DENSE_REFERENCED_DENOMINATOR: usize = 5;

/// Takes `indices` from `values`, an encoding that decodes row by row, such as a string
/// compressor, choosing the cheapest of three strategies:
///
/// - For a sparse take, `gather` takes the encoded values, which are decoded afterwards. Repeated
///   indices decode again, but a sparse take has few to repeat.
/// - When most values are referenced, or `values` is small, decodes every value once and gathers
///   from the decoded array, returning `None` to leave that to execution where it is known
///   without reading the indices.
/// - Otherwise, filters `values` down to the referenced values, decodes those once, and gathers
///   from them with the indices remapped. This avoids both decoding every repeat and decoding the
///   unreferenced values.
pub fn take_referenced_canonical(
    values: &ArrayRef,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
    gather: impl FnOnce(&mut ExecutionCtx) -> VortexResult<ArrayRef>,
) -> VortexResult<Option<ArrayRef>> {
    let len = values.len();
    // Ranks are `u32`, so a larger array decodes every value.
    if len < MIN_VALUES_LEN
        || u32::try_from(len).is_err()
        || indices.len() >= len.saturating_mul(DENSE_INDICES_PER_VALUE)
    {
        return Ok(None);
    }
    if indices.len().saturating_mul(SPARSE_TAKE_DENOMINATOR) < len {
        return gather(ctx).map(Some);
    }

    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let Some((referenced, remapped)) = match_each_integer_ptype!(indices.ptype(), |P| {
        remap_referenced(indices.as_slice::<P>(), len)
    }) else {
        // Not `None`: execution would call this kernel again before decoding the values,
        // repeating the marking.
        return take_decoded(values, indices.into_array(), ctx).map(Some);
    };
    let remapped = PrimitiveArray::new(remapped, indices.validity()?);

    let referenced_values = values
        .filter(Mask::from_buffer(referenced))?
        .execute::<Canonical>(ctx)?
        .into_array();
    referenced_values
        .take(remapped.into_array())?
        .execute::<Canonical>(ctx)
        .map(|taken| Some(taken.into_array()))
}

/// Decodes every value, then gathers from the decoded array.
fn take_decoded(
    values: &ArrayRef,
    indices: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    values
        .clone()
        .execute::<Canonical>(ctx)?
        .into_array()
        .take(indices)?
        .execute::<Canonical>(ctx)
        .map(Canonical::into_array)
}

/// Marks the values `indices` reference and maps each index to its position among them, or
/// returns `None` when too many values are referenced to be worth filtering.
///
/// Null indices are treated like valid ones: the taken validity hides whatever they gather, so
/// they only cost marking an extra value. Their arbitrary, possibly out-of-bounds, values are
/// clamped into bounds.
fn remap_referenced<P: AsPrimitive<usize>>(
    indices: &[P],
    len: usize,
) -> Option<(BitBuffer, Buffer<u32>)> {
    let last = len - 1;
    let dense_count = len.div_ceil(DENSE_REFERENCED_DENOMINATOR);
    // First a mark per value, then each value's rank among the referenced values. Marking stops
    // as soon as enough values are referenced to decode them all, which for dense indices is long
    // before the last index.
    let mut ranks = vec![0u32; len];
    let mut referenced_count = 0usize;
    for chunk in indices.chunks(256) {
        for &idx in chunk {
            let mark = &mut ranks[idx.as_().min(last)];
            referenced_count += (1 - *mark) as usize;
            *mark = 1;
        }
        if referenced_count >= dense_count {
            return None;
        }
    }

    let referenced = BitBuffer::collect_bool(len, |idx| ranks[idx] != 0);
    let mut rank = 0u32;
    for value_rank in &mut ranks {
        let is_referenced = *value_rank;
        *value_rank = rank;
        rank += is_referenced;
    }

    let remapped = indices
        .iter()
        .map(|&idx| ranks[idx.as_().min(last)])
        .collect();
    Some((referenced, remapped))
}
