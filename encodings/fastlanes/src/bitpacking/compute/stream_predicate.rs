// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming, cache-reusable predicate evaluation over a [`BitPackedArray`].
//!
//! Walks the encoded array one 1024-element FastLanes block at a time through a single
//! reusable scratch buffer, splices any [`crate::patches::Patches`] into the unpacked block
//! in place via a sorted-index cursor, then folds a `Fn(T) -> bool` predicate over the
//! block. The fold matches the canonical [`vortex_buffer::BitBuffer::collect_bool`] shape
//! (pack 64 bools into a `u64` in a tight auto-vectorisable inner loop) and writes the
//! resulting words straight into the output bit buffer, so the materialised primitive
//! never appears anywhere.

use std::ops::Range;

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
use vortex_buffer::collect_bool_word;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedIter;

/// Stream `predicate` over the unpacked values of a [`BitPackedArray`], one FastLanes
/// block at a time, producing a [`BoolArray`].
pub(super) fn stream_predicate<T, P>(
    array: ArrayView<'_, BitPacked>,
    nullability: Nullability,
    predicate: P,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: BitPackedIter + NativePType + Copy,
    P: Fn(T) -> bool,
{
    let len = array.len();
    let num_words = len.div_ceil(64);
    let mut words: BufferMut<u64> = BufferMut::zeroed(num_words);

    if len > 0 {
        let mut chunks = array.unpacked_chunks::<T>()?;
        let words = words.as_mut_slice();

        if let Some(p) = array.patches() {
            let p_idx_arr = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
            let p_val_arr = p.values().clone().execute::<PrimitiveArray>(ctx)?;
            let p_off = p.offset();
            match_each_unsigned_integer_ptype!(p_idx_arr.ptype(), |I| {
                let p_idx = p_idx_arr.as_slice::<I>();
                let p_val = p_val_arr.as_slice::<T>();
                let mut p_cur: usize = 0;
                chunks.for_each_unpacked_chunk(|block, range| {
                    splice_patches::<T, I>(block, range.start, &mut p_cur, p_idx, p_val, p_off);
                    pack_predicate_block(words, range, block, &predicate);
                });
            });
        } else {
            chunks.for_each_unpacked_chunk(|block, range| {
                pack_predicate_block(words, range, block, &predicate);
            });
        }
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);
    let validity = array.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Overwrite the unpacked block in place with any patches falling in
/// `[chunk_start, chunk_start + block.len())`, then advance `cursor` past them. Sorted
/// indices mean the cursor only moves forward across the whole walk.
#[inline]
fn splice_patches<T, I>(
    block: &mut [T],
    chunk_start: usize,
    cursor: &mut usize,
    indices: &[I],
    values: &[T],
    patch_offset: usize,
) where
    T: Copy,
    I: AsPrimitive<usize>,
{
    let end = chunk_start + block.len();
    while *cursor < indices.len() {
        let global: usize = indices[*cursor].as_();
        let local = global - patch_offset;
        if local >= end {
            break;
        }
        debug_assert!(local >= chunk_start);
        block[local - chunk_start] = values[*cursor];
        *cursor += 1;
    }
}

/// Fold `predicate` over `block`, packing 64 bools into a `u64` per inner-loop pass and
/// writing the words directly into `words` at `range.start`.
#[inline]
fn pack_predicate_block<T, P>(words: &mut [u64], range: Range<usize>, block: &[T], predicate: &P)
where
    T: Copy,
    P: Fn(T) -> bool,
{
    debug_assert_eq!(range.len(), block.len());
    let start_bit = range.start;
    let active_len = range.len();
    if active_len == 0 {
        return;
    }

    if start_bit.is_multiple_of(64) {
        let mut word_idx = start_bit / 64;
        let mut chunks = block.chunks_exact(64);

        for chunk in chunks.by_ref() {
            words[word_idx] = collect_bool_word(chunk.len(), |bit_idx| predicate(chunk[bit_idx]));
            word_idx += 1;
        }

        let tail = chunks.remainder();
        if !tail.is_empty() {
            words[word_idx] = collect_bool_word(tail.len(), |bit_idx| predicate(tail[bit_idx]));
        }
    } else {
        // Unaligned cursor — array sliced at a non-64-aligned offset. Per-bit OR.
        for (bit_offset, &value) in block.iter().enumerate() {
            if predicate(value) {
                let bit_pos = start_bit + bit_offset;
                words[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }
}
