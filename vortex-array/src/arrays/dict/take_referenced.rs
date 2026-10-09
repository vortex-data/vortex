// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArrayExt;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Takes `indices` from `values` by decoding each referenced value once.
///
/// For encodings that decode row by row, such as string compressors, gathering the encoded rows
/// and decoding the result decodes a value again for every index that repeats it. This instead
/// filters `values` down to the rows `indices` reference, executes those to canonical once, and
/// gathers from that canonical array with the indices remapped into it.
pub fn take_referenced_canonical(
    values: &ArrayRef,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if indices.is_empty() {
        let dtype = values
            .dtype()
            .union_nullability(indices.dtype().nullability());
        return Ok(Canonical::empty(&dtype).into_array());
    }

    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let dict = DictArray::try_new(indices.clone().into_array(), values.clone())?;
    let referenced = dict.compute_referenced_values_mask(true, ctx)?;
    let referenced_count = referenced.true_count();

    if referenced_count == 0 {
        // Every index is null.
        return Ok(
            ConstantArray::new(Scalar::null(values.dtype().as_nullable()), indices.len())
                .into_array(),
        );
    }

    // Remapped indices are `u32`, so a larger referenced set takes from all of `values`.
    if referenced_count == values.len() || u32::try_from(referenced_count).is_err() {
        let canonical = values.clone().execute::<Canonical>(ctx)?.into_array();
        return canonical
            .take(indices.into_array())?
            .execute::<Canonical>(ctx)
            .map(Canonical::into_array);
    }

    let indices_validity = indices
        .as_ref()
        .validity()?
        .execute_mask(indices.len(), ctx)?;
    let remapped = match_each_integer_ptype!(indices.ptype(), |P| {
        remap_indices(indices.as_slice::<P>(), &indices_validity, &referenced)
    });
    let remapped = PrimitiveArray::new(
        remapped,
        Validity::from_mask(indices_validity, indices.dtype().nullability()),
    );

    let referenced_values = values
        .filter(Mask::from_buffer(referenced))?
        .execute::<Canonical>(ctx)?
        .into_array();
    referenced_values
        .take(remapped.into_array())?
        .execute::<Canonical>(ctx)
        .map(Canonical::into_array)
}

/// Maps each valid index to its position among the set bits of `referenced`, and each null index
/// to zero.
fn remap_indices<P: AsPrimitive<usize>>(
    indices: &[P],
    validity: &Mask,
    referenced: &BitBuffer,
) -> Buffer<u32> {
    // `words[w]` holds bits `64 * w..64 * (w + 1)` of `referenced`, and `ranks[w]` counts the set
    // bits before them, so a rank is one lookup plus one popcount.
    let words: Vec<u64> = referenced.chunks().iter_padded().collect();
    let ranks: Vec<u32> = words
        .iter()
        .scan(0u32, |rank, word| {
            let before = *rank;
            *rank += word.count_ones();
            Some(before)
        })
        .collect();
    let rank = |idx: usize| {
        let (word, bit) = (idx / 64, idx % 64);
        ranks[word] + (words[word] & ((1u64 << bit) - 1)).count_ones()
    };

    match validity.bit_buffer() {
        AllOr::All => indices.iter().map(|&idx| rank(idx.as_())).collect(),
        AllOr::None => Buffer::zeroed(indices.len()),
        AllOr::Some(valid) => indices
            .iter()
            .zip(valid.iter())
            .map(|(&idx, is_valid)| if is_valid { rank(idx.as_()) } else { 0 })
            .collect(),
    }
}
