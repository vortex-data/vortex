// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused compare kernel for [`BitPackedArray`] against a constant.
//!
//! Where [`super::stream_predicate`] unpacks a full 1024-element FastLanes block into a scratch
//! buffer and *then* folds a predicate over it, this path hands the comparison down into the
//! FastLanes [`BitPackingCompare::unchecked_unpack_cmp`] kernel, which compares each value against
//! the constant *as it is unpacked*, accumulating the boolean results straight into a 1024-bit
//! mask (`[u64; 16]`) in transposed FastLanes lane order - one register-resident word per lane, no
//! `[bool; 1024]` or `[T; 1024]` scratch. A single SIMD [`untranspose_bits`] per block then rotates
//! that mask into logical row order, which is copied directly into the output bit buffer.
//!
//! Only the full-chunk fast path uses the fused kernel. Sliced arrays (non-zero block offset) fall
//! back to the scalar streaming predicate, and inline patches are spliced in afterwards by
//! overwriting the bits at the patched indices with `cmp(patch_value, rhs)`.

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use fastlanes::untranspose_bits;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::stream_predicate::stream_predicate;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedIter;

const CHUNK_SIZE: usize = 1024;
/// `u64` words spanning one FastLanes block (1024 bits / 64).
const WORDS_PER_CHUNK: usize = CHUNK_SIZE / u64::BITS as usize;

/// Compare the unpacked values of a [`BitPackedArray`] against `rhs` using the fused FastLanes
/// `unpack_cmp` kernel, producing a [`BoolArray`].
///
/// `cmp(value, rhs)` defines the predicate; it must be the total-order comparison matching the
/// requested operator (e.g. `|a, b| a.is_lt(b)`).
pub(super) fn stream_compare_fused<T, F>(
    array: ArrayView<'_, BitPacked>,
    rhs: T,
    nullability: Nullability,
    cmp: F,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + BitPackedIter + FastLanesComparable,
    <T as FastLanesComparable>::Bitpacked: BitPacking + NativePType + BitPackingCompare,
    F: Fn(T, T) -> bool + Copy,
{
    let len = array.len();
    let bit_width = BitPackedArrayExt::bit_width(&array) as usize;
    let offset = BitPackedArrayExt::offset(&array) as usize;

    // The fused kernel consumes whole 1024-element blocks at a fixed packed width. A non-zero
    // block offset (from slicing) or a degenerate width has no clean full-chunk form, so defer
    // to the scalar streaming predicate, which handles every layout.
    if offset != 0 || len == 0 || bit_width == 0 {
        return stream_predicate::<T, _>(array, nullability, move |v| cmp(v, rhs), ctx);
    }

    let packed = BitPackedArrayExt::packed_slice::<<T as FastLanesComparable>::Bitpacked>(&array);
    let elems_per_chunk = 128 * bit_width / size_of::<<T as FastLanesComparable>::Bitpacked>();
    let num_chunks = len.div_ceil(CHUNK_SIZE);

    let mut words: BufferMut<u64> = BufferMut::zeroed(len.div_ceil(u64::BITS as usize));
    {
        let words = words.as_mut_slice();
        // Per block: fuse compare into a transposed 1024-bit mask, then untranspose into logical
        // row order. The packed buffer is zero-padded out to a whole final block, so every chunk -
        // including the trailing partial one - has exactly `elems_per_chunk` packed values; we just
        // copy fewer than 16 words out of the last block's untransposed mask.
        let mut transposed = [0u64; WORDS_PER_CHUNK];
        let mut logical = [0u64; WORDS_PER_CHUNK];
        for chunk in 0..num_chunks {
            let packed_chunk = &packed[chunk * elems_per_chunk..][..elems_per_chunk];
            // SAFETY: `packed_chunk` is exactly `128 * bit_width / size_of::<U>()` elements and
            // `bit_width <= U::T`, satisfying `unchecked_unpack_cmp`'s contract.
            unsafe {
                <<T as FastLanesComparable>::Bitpacked as BitPackingCompare>::unchecked_unpack_cmp::<
                    T,
                    _,
                >(bit_width, packed_chunk, &mut transposed, cmp, rhs);
            }
            untranspose_bits::<<T as FastLanesComparable>::Bitpacked>(&transposed, &mut logical);

            let block_start = chunk * CHUNK_SIZE;
            let block_bits = (len - block_start).min(CHUNK_SIZE);
            let word_off = chunk * WORDS_PER_CHUNK;
            let n_words = block_bits.div_ceil(u64::BITS as usize);
            words[word_off..][..n_words].copy_from_slice(&logical[..n_words]);
        }

        // Patched indices hold placeholder packed values, so their fused result is meaningless;
        // overwrite each with the comparison against the real patch value.
        if let Some(p) = array.patches() {
            let p_idx = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
            let p_val = p.values().clone().execute::<PrimitiveArray>(ctx)?;
            let p_off = p.offset();
            match_each_unsigned_integer_ptype!(p_idx.ptype(), |I| {
                let indices = p_idx.as_slice::<I>();
                let values = p_val.as_slice::<T>();
                for (&global, &value) in indices.iter().zip(values) {
                    let global: usize = global.as_();
                    set_bit(words, global - p_off, cmp(value, rhs));
                }
            });
        }
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);
    let validity = array.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Branchlessly write a single bit in a packed `u64` word buffer: clear the bit, then OR in the
/// new value. Avoids a data-dependent branch per patch in the patch-fixup loop, and touches the
/// target word through a single bounds-checked `&mut`.
#[inline]
fn set_bit(words: &mut [u64], idx: usize, value: bool) {
    let shift = idx % u64::BITS as usize;
    let mask = 1u64 << shift;
    let word = &mut words[idx / u64::BITS as usize];
    *word = (*word & !mask) | (u64::from(value) << shift);
}
