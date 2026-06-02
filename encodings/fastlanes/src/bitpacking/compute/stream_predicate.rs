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

use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
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
use vortex_buffer::pack_bools_into_words;
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
    let mut words: BufferMut<u64> = BufferMut::zeroed(len.div_ceil(u64::BITS as usize));

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
                    p_cur = splice_patches::<T, I>(block, range.start, p_cur, p_idx, p_val, p_off);
                    pack_bools_into_words(words, range.start, block.len(), |i| predicate(block[i]));
                });
            });
        } else {
            chunks.for_each_unpacked_chunk(|block, range| {
                pack_bools_into_words(words, range.start, block.len(), |i| predicate(block[i]));
            });
        }
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);
    let validity = array.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Overwrite the unpacked block in place with any patches falling in
/// `[chunk_start, chunk_start + block.len())`, starting from `cursor` and returning the
/// advanced cursor. Sorted indices mean the cursor only moves forward across the walk.
#[inline]
fn splice_patches<T, I>(
    block: &mut [T],
    chunk_start: usize,
    mut cursor: usize,
    indices: &[I],
    values: &[T],
    patch_offset: usize,
) -> usize
where
    T: Copy,
    I: AsPrimitive<usize>,
{
    let end = chunk_start + block.len();
    while cursor < indices.len() {
        let global: usize = indices[cursor].as_();
        let local = global - patch_offset;
        if local >= end {
            break;
        }
        debug_assert!(local >= chunk_start);
        block[local - chunk_start] = values[cursor];
        cursor += 1;
    }
    cursor
}

/// Compare every element of a [`BitPackedArray`](crate::BitPackedArray) against the constant
/// `value` with `cmp`, producing a [`BoolArray`].
///
/// Unlike [`stream_predicate`], this uses the FastLanes fused unpack-and-compare kernel
/// ([`fastlanes::BitPackingCompare`]): each value is unpacked in-register and compared on the
/// spot, so neither the unpacked primitive nor a per-element scratch is materialised. Patches
/// are applied afterwards by overwriting the result bit at each patched index with
/// `cmp(patch_value, value)`.
pub(super) fn stream_compare<T, F>(
    array: ArrayView<'_, BitPacked>,
    value: T,
    cmp: F,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: BitPackedIter
        + NativePType
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPackingCompare,
    F: Fn(T, T) -> bool + Copy,
{
    let len = array.len();
    let mut words: BufferMut<u64> = BufferMut::zeroed(len.div_ceil(u64::BITS as usize));

    if len > 0 {
        let mut chunks = array.unpacked_chunks::<T>()?;
        let words = words.as_mut_slice();
        chunks.for_each_compared_chunk(cmp, value, |bools, range| {
            pack_bools_into_words(words, range.start, bools.len(), |i| bools[i]);
        });
    }

    let mut bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);

    if let Some(p) = array.patches() {
        let p_idx = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let p_val = p.values().clone().execute::<PrimitiveArray>(ctx)?;
        let p_off = p.offset();
        match_each_unsigned_integer_ptype!(p_idx.ptype(), |I| {
            apply_compare_patches::<T, I, F>(
                &mut bits,
                p_idx.as_slice::<I>(),
                p_val.as_slice::<T>(),
                p_off,
                cmp,
                value,
            );
        });
    }

    let validity = array.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Overwrite the result bit at each patched index with `cmp(patch_value, value)`. The fused
/// compare kernel reads the (truncated) packed value at patched positions, so those bits are
/// stale until corrected here.
fn apply_compare_patches<T, I, F>(
    bits: &mut BitBufferMut,
    indices: &[I],
    values: &[T],
    indices_offset: usize,
    cmp: F,
    value: T,
) where
    T: NativePType,
    I: AsPrimitive<usize>,
    F: Fn(T, T) -> bool,
{
    let len = bits.len();
    for (&raw_idx, &patch_value) in indices.iter().zip(values.iter()) {
        let i: usize = raw_idx.as_();
        if i < indices_offset {
            continue;
        }
        let pos = i - indices_offset;
        if pos >= len {
            break;
        }
        if cmp(patch_value, value) {
            bits.set(pos);
        } else {
            bits.unset(pos);
        }
    }
}
