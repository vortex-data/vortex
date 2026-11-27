// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::match_each_integer_ptype;

/// With at least this many indices per value, remapping every index costs more than decoding
/// the unreferenced values saves.
const DENSE_INDICES_PER_VALUE: usize = 4;

/// Once at least `1 / DENSE_REFERENCED_DENOMINATOR` of the values are referenced, gathering them
/// costs more than decoding the unreferenced ones.
const DENSE_REFERENCED_DENOMINATOR: usize = 5;

/// Takes `indices` from `values`, an encoding that decodes row by row, decoding each value at most
/// once.
///
/// When few values are referenced, `gather` takes them from the encoded `values` in order, they
/// are decoded, and the indices are remapped into them. Otherwise, or when there are many indices
/// per value, every value is decoded, which is cheaper than gathering most of them or remapping
/// every index.
pub fn take_referenced(
    values: &ArrayRef,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
    gather: impl FnOnce(&ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
) -> VortexResult<ArrayRef> {
    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let referenced = if indices.len() >= values.len().saturating_mul(DENSE_INDICES_PER_VALUE) {
        None
    } else {
        match_each_integer_ptype!(indices.ptype(), |P| {
            remap_referenced(indices.as_slice::<P>(), values.len())
        })
    };

    let (decoded, indices) = match referenced {
        Some((referenced, remapped)) => (
            gather(&referenced.into_array(), ctx)?,
            PrimitiveArray::new(remapped, indices.validity()?).into_array(),
        ),
        None => (values.clone(), indices.into_array()),
    };
    decoded
        .execute::<Canonical>(ctx)?
        .into_array()
        .take(indices)?
        .execute::<Canonical>(ctx)
        .map(Canonical::into_array)
}

/// Returns the values `indices` reference, in order, and each index's position among them, or
/// `None` when so many values are referenced that decoding all of them is cheaper.
///
/// Null indices are treated like valid ones: the taken validity hides whatever they gather, so
/// they cost at most one extra value. Their arbitrary, possibly out-of-bounds, values are clamped.
fn remap_referenced<P: AsPrimitive<usize>>(
    indices: &[P],
    len: usize,
) -> Option<(Buffer<u32>, Buffer<u32>)> {
    let last = len.checked_sub(1)?;
    u32::try_from(len).ok()?;
    let dense_count = len.div_ceil(DENSE_REFERENCED_DENOMINATOR);

    // First a mark per value, then each value's position among the referenced values. Marking
    // stops as soon as enough values are referenced to decode them all.
    let mut ranks = vec![0u32; len];
    let mut referenced_count = 0;
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

    let mut referenced = BufferMut::<u32>::with_capacity(referenced_count);
    let mut next_rank = 0u32;
    for (value, rank) in (0u32..).zip(ranks.iter_mut()) {
        let is_referenced = *rank;
        *rank = next_rank;
        next_rank += is_referenced;
        if is_referenced != 0 {
            referenced.push(value);
        }
    }

    let remapped = indices
        .iter()
        .map(|&idx| ranks[idx.as_().min(last)])
        .collect();
    Some((referenced.freeze(), remapped))
}
