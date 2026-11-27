// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::execute::take_canonical;
use crate::match_each_integer_ptype;

/// Takes from a row-wise encoding by gathering and decoding distinct rows in source order.
/// `code_count` is the number of compressed codes in the logical source range, when known.
/// Returns `None` when the caller should use its normal execution path.
pub fn take_referenced(
    values: &ArrayRef,
    indices: &ArrayRef,
    code_count: Option<u64>,
    ctx: &mut ExecutionCtx,
    gather: impl FnOnce(&ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
) -> VortexResult<Option<ArrayRef>> {
    if values.len() < 1024 || u32::try_from(values.len()).is_err() {
        return Ok(None);
    }
    let limit = gather_limit(values.len(), indices.len(), code_count);
    if limit == 0 {
        return Ok(None);
    }

    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let validity = indices.validity()?;
    let mask = validity.execute_mask(indices.len(), ctx)?;
    if mask.all_false() {
        return Ok(None);
    }
    let referenced = match_each_integer_ptype!(indices.ptype(), |P| {
        remap_referenced(indices.as_slice::<P>(), &mask, values.len(), limit)
    });
    let (values, indices) = match referenced {
        Some((referenced, remapped)) => (
            gather(&referenced.into_array(), ctx)?,
            PrimitiveArray::new(remapped, validity),
        ),
        None => (values.clone(), indices),
    };

    // Finish the dense fallback here so the executor does not retry the same marking pass.
    let decoded = values.execute::<Canonical>(ctx)?.into_array();
    let taken = take_canonical(decoded.as_::<AnyCanonical>(), indices.as_view(), ctx)?;
    Ok(Some(taken.into_array()))
}

/// Estimates how many distinct rows are cheaper to gather than to bulk decode. The weights
/// approximate setup, index remapping, and per-row work in units of compressed codes.
/// Longer strings can amortize gathering a larger fraction of a dictionary.
fn gather_limit(rows: usize, indices: usize, code_count: Option<u64>) -> usize {
    let codes = code_count
        .and_then(|count| usize::try_from(count).ok())
        .unwrap_or_else(|| rows.saturating_mul(16));
    // Start with bulk decode and view construction, then deduct the setup and rank buffer.
    let budget = codes
        .saturating_add(rows.saturating_mul(8))
        .saturating_sub(8 * 1024)
        .saturating_sub(indices.saturating_mul(size_of::<u32>()));
    // Each selected row also needs its compressed range, decoded length and view.
    let per_row = (codes / rows).saturating_add(32);
    (budget / per_row).min(rows.div_ceil(2))
}

fn remap_referenced<P: AsPrimitive<usize>>(
    indices: &[P],
    validity: &Mask,
    len: usize,
    limit: usize,
) -> Option<(Buffer<u32>, Buffer<u32>)> {
    let mut words = vec![0u64; len.div_ceil(64)];
    let chunk_size = len.div_ceil(2);
    let mut count = 0;

    // Count bits in batches so high-cardinality takes can abandon marking early.
    for (chunk_idx, chunk) in indices.chunks(chunk_size).enumerate() {
        let mut mark = |idx: P| {
            let row = idx.as_();
            *words.get_mut(row / 64)? |= 1 << (row % 64);
            Some(())
        };
        match validity.bit_buffer() {
            AllOr::All => chunk.iter().copied().try_for_each(&mut mark)?,
            AllOr::None => return None,
            AllOr::Some(bits) => {
                let start = chunk_idx * chunk_size;
                bits.slice(start..start + chunk.len())
                    .try_for_each_set_index(|i| mark(chunk[i]).ok_or(()))
                    .ok()?;
            }
        }
        count = words.iter().map(|word| word.count_ones() as usize).sum();
        if count >= limit {
            return None;
        }
    }

    let mut referenced = BufferMut::with_capacity(count);
    let mut ranks = vec![0u32; len];
    for (block, mut word) in words.into_iter().enumerate() {
        while word != 0 {
            let row = block * 64 + word.trailing_zeros() as usize;
            *ranks.get_mut(row)? = u32::try_from(referenced.len()).ok()?;
            referenced.push(u32::try_from(row).ok()?);
            word &= word - 1;
        }
    }
    // Payloads at null positions can be out of bounds. The output validity masks their ranks.
    let remapped = Buffer::from_trusted_len_iter(
        indices.iter().map(|idx| ranks.get(idx.as_()).copied().unwrap_or(0)),
    );
    Some((referenced.freeze(), remapped))
}
