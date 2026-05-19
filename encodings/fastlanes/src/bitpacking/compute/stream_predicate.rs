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

use lending_iterator::LendingIterator;
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

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedIter;
use crate::unpack_iter::BitUnpackedChunks;

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
    let mut cursor: usize = 0;

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
                cursor = walk_blocks(&mut chunks, len, cursor, |block, c| {
                    splice_patches::<T, I>(block, c, &mut p_cur, p_idx, p_val, p_off);
                    write_block(words, c, block, &predicate, len)
                });
            });
        } else {
            cursor = walk_blocks(&mut chunks, len, cursor, |block, c| {
                write_block(words, c, block, &predicate, len)
            });
        }
    }

    debug_assert_eq!(cursor, len);
    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);
    let validity = array.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Walk every unpacked block (initial / full / trailer) in order, invoking `f` once per
/// block. `f` receives the block and the current bit cursor and returns the new cursor.
/// The internal scratch buffer is reused between calls, so `f` must consume the block
/// before returning.
fn walk_blocks<T, F>(
    chunks: &mut BitUnpackedChunks<T>,
    len: usize,
    start_cursor: usize,
    mut f: F,
) -> usize
where
    T: BitPackedIter,
    F: FnMut(&mut [T], usize) -> usize,
{
    let mut cursor = start_cursor;
    if let Some(initial) = chunks.initial() {
        cursor = f(initial, cursor);
    }
    // When `num_chunks == 1` and not sliced at the tail, `initial` already consumed the
    // whole array and `full_chunks` would re-yield the same data. Guard with the cursor.
    if cursor < len {
        let mut iter = chunks.full_chunks();
        while let Some(chunk) = iter.next() {
            cursor = f(chunk, cursor);
            if cursor >= len {
                break;
            }
        }
    }
    if cursor < len
        && let Some(trailer) = chunks.trailer()
    {
        cursor = f(trailer, cursor);
    }
    cursor
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
/// writing the words directly into `words` at `start_bit`. Auto-vectorises into the same
/// `pcmpeq + psllq + por` shape that arrow-ord's `apply_op` lowers to.
#[inline]
fn write_block<T, P>(
    words: &mut [u64],
    start_bit: usize,
    block: &[T],
    predicate: &P,
    total_len: usize,
) -> usize
where
    T: Copy,
    P: Fn(T) -> bool,
{
    let end_bit = (start_bit + block.len()).min(total_len);
    let active_len = end_bit - start_bit;
    if active_len == 0 {
        return start_bit;
    }

    if start_bit.is_multiple_of(64) {
        let mut word_idx = start_bit / 64;
        let full_words = active_len / 64;
        for w in 0..full_words {
            let mut packed = 0u64;
            for b in 0..64 {
                // SAFETY: w * 64 + b < full_words * 64 <= active_len <= block.len().
                let v = unsafe { *block.get_unchecked(w * 64 + b) };
                packed |= (predicate(v) as u64) << b;
            }
            // SAFETY: word_idx < num_words = total_len.div_ceil(64) by construction.
            unsafe {
                *words.get_unchecked_mut(word_idx) = packed;
            }
            word_idx += 1;
        }
        let tail = active_len % 64;
        if tail > 0 {
            let base = full_words * 64;
            let mut packed = 0u64;
            for b in 0..tail {
                // SAFETY: base + b < active_len <= block.len().
                let v = unsafe { *block.get_unchecked(base + b) };
                packed |= (predicate(v) as u64) << b;
            }
            unsafe {
                *words.get_unchecked_mut(word_idx) = packed;
            }
        }
    } else {
        // Unaligned cursor — array sliced at a non-64-aligned offset. Per-bit OR.
        for b in 0..active_len {
            // SAFETY: b < active_len <= block.len().
            let v = unsafe { *block.get_unchecked(b) };
            if predicate(v) {
                let bit_pos = start_bit + b;
                unsafe {
                    *words.get_unchecked_mut(bit_pos / 64) |= 1u64 << (bit_pos % 64);
                }
            }
        }
    }

    end_bit
}
