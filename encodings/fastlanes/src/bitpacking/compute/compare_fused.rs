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
//! that mask into logical row order.
//!
//! The packed blocks are walked through the regular [`crate::unpack_iter::BitUnpackedChunks`]
//! iterator (via [`crate::unpack_iter::BitUnpackedChunks::for_each_packed_chunk`]) rather than a
//! bespoke chunk loop, so chunk sizing and bounds live in one place.
//!
//! Slicing is handled by working in *padded* coordinates: bit `offset + i` holds element `i`. The
//! output buffer is over-allocated to whole 1024-bit blocks, so every block - the sliced first
//! block, the body, and the trailing partial - untransposes straight into a 64-bit-word-aligned
//! slot with no per-block temporary and only one shared scratch `[u64; 16]`. The leading `offset`
//! garbage rows and the trailing padding are trimmed afterwards: leading whole `u64` words are
//! dropped with [`BufferMut::split_off`] (an O(1) re-slice) and the residual `offset % 64` bits plus
//! `len` form the final [`BitBufferMut`] view. The result is therefore byte-aligned (offset 0) when
//! `offset % 8 == 0`. Inline patches are spliced in afterwards by overwriting the bits at the
//! patched indices with `cmp(patch_value, rhs)`.

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
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::stream_predicate::stream_predicate;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedIter;

const CHUNK_SIZE: usize = 1024;
/// `u64` words spanning one FastLanes block (1024 bits / 64).
const WORDS_PER_CHUNK: usize = CHUNK_SIZE / u64::BITS as usize;

/// Unpack one packed FastLanes block, comparing each value against `rhs` *as it is unpacked*, and
/// write the resulting 1024-bit mask (logical row order, LSB-first) into `out`.
///
/// `transposed` is caller-owned scratch reused across every block, so the hot loop makes no
/// per-call stack allocation; its prior contents are irrelevant (the kernel overwrites it).
#[inline]
fn unpack_cmp_block<T, F>(
    transposed: &mut [u64; WORDS_PER_CHUNK],
    out: &mut [u64; WORDS_PER_CHUNK],
    bit_width: usize,
    packed_chunk: &[<T as PhysicalPType>::Physical],
    cmp: F,
    rhs: T,
) where
    T: NativePType
        + PhysicalPType
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPacking + BitPackingCompare,
    F: Fn(T, T) -> bool,
{
    transposed.fill(0);
    // SAFETY: `packed_chunk` holds exactly `128 * bit_width / size_of::<U>()` packed elements and
    // `bit_width <= U::T`, satisfying `unchecked_unpack_cmp`'s contract.
    unsafe {
        <<T as PhysicalPType>::Physical as BitPackingCompare>::unchecked_unpack_cmp::<T, _>(
            bit_width,
            packed_chunk,
            transposed,
            cmp,
            rhs,
        );
    }
    untranspose_bits::<<T as PhysicalPType>::Physical>(transposed, out);
}

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
    T: NativePType
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPacking + NativePType + BitPackingCompare,
    F: Fn(T, T) -> bool + Copy,
{
    let len = array.len();
    let bit_width = array.bit_width() as usize;
    let offset = array.offset() as usize;

    // A degenerate width has no packed payload for the fused kernel to consume; defer to the scalar
    // streaming predicate, which handles every layout (including the empty array).
    if len == 0 || bit_width == 0 {
        return stream_predicate::<T, _>(array, nullability, move |v| cmp(v, rhs), ctx);
    }

    // Over-allocate to whole 1024-bit blocks in padded coordinates so every block - including the
    // trailing partial - has room for a full untranspose at a 64-bit-word-aligned offset.
    let num_chunks = (offset + len).div_ceil(CHUNK_SIZE);
    let mut words: BufferMut<u64> = BufferMut::zeroed(num_chunks * WORDS_PER_CHUNK);

    let chunks = array.unpacked_chunks::<T>()?;
    {
        let words = words.as_mut_slice();
        let mut transposed = [0u64; WORDS_PER_CHUNK];
        chunks.for_each_packed_chunk(|packed_chunk, range| {
            // Block starts are always 1024-aligned (padded coords), so the slot is a full block.
            let out = words[range.start / u64::BITS as usize..]
                .first_chunk_mut::<WORDS_PER_CHUNK>()
                .vortex_expect("over-allocated buffer holds a full block per chunk");
            unpack_cmp_block::<T, _>(&mut transposed, out, bit_width, packed_chunk, cmp, rhs);
        });

        // Patched indices hold placeholder packed values, so their fused result is meaningless;
        // overwrite each with the comparison against the real patch value. Patch positions are
        // logical (`global - p_off`); shift by `offset` into padded coordinates.
        if let Some(p) = array.patches() {
            let p_idx = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
            let p_val = p.values().clone().execute::<PrimitiveArray>(ctx)?;
            let p_off = p.offset();
            match_each_unsigned_integer_ptype!(p_idx.ptype(), |I| {
                let indices = p_idx.as_slice::<I>();
                let values = p_val.as_slice::<T>();
                for (&global, &value) in indices.iter().zip(values) {
                    let global: usize = global.as_();
                    set_bit(words, offset + global - p_off, cmp(value, rhs));
                }
            });
        }
    }

    // Trim the leading garbage: drop whole `u64` words covered entirely by the skipped `offset`
    // region (an O(1) re-slice; `u64` granularity keeps the byte buffer 8-aligned). The residual
    // `offset % 64` bits become the view offset, so the result is byte-aligned when `offset % 8 == 0`
    // and offset 0 when `offset % 64 == 0`.
    let head_words = offset / u64::BITS as usize;
    let bit_offset = offset % u64::BITS as usize;
    let bytes = words.split_off(head_words).into_byte_buffer();
    let bits = BitBufferMut::from_buffer(bytes, bit_offset, len);
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
