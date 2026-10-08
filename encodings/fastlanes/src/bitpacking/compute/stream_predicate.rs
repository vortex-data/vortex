// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming, cache-reusable predicate evaluation over a [`BitPackedArray`].
//!
//! Walks the encoded array one 1024-element FastLanes block at a time through a single
//! reusable scratch buffer, splices any [`Patches`] into the unpacked block
//! in place via a sorted-index cursor, then folds a `Fn(T) -> bool` predicate over the
//! block. Full bitmap words use the lane predicate kernel, while partial words preserve
//! bits from adjacent blocks. The fold writes the
//! resulting words straight into the output bit buffer, so the materialised primitive
//! never appears anywhere.
//!
//! [`BitPackedArray`]: crate::BitPackedArray
//! [`Patches`]: vortex_array::patches::Patches

use std::mem::MaybeUninit;

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
use vortex_buffer::pack_bools_into_words;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::unpack_iter::BitPacked as BitPackedIter;

/// Stream `predicate` over the unpacked values of a [`BitPackedArray`](crate::BitPackedArray), one FastLanes
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
        let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
        let mut chunks = array.unpacked_chunks::<T>(&mut scratch)?;
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
                    pack_predicate(words, range.start, block, &predicate);
                });
            });
        } else {
            chunks.for_each_unpacked_chunk(|block, range| {
                pack_predicate(words, range.start, block, &predicate);
            });
        }
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);
    let validity = array.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

fn pack_predicate<T: Copy>(
    words: &mut [u64],
    start: usize,
    values: &[T],
    predicate: &impl Fn(T) -> bool,
) {
    let prefix = if start.is_multiple_of(64) {
        0
    } else {
        (64 - start % 64).min(values.len())
    };
    if prefix != 0 {
        pack_bools_into_words(words, start, prefix, |i| predicate(values[i]));
    }
    let values = &values[prefix..];
    let full_len = values.len() / 64 * 64;
    let word_start = (start + prefix) / 64;
    values[..full_len].map_bits_into(
        &mut words[word_start..word_start + full_len / 64],
        predicate,
    );
    let tail = &values[full_len..];
    pack_bools_into_words(words, start + prefix + full_len, tail.len(), |i| {
        predicate(tail[i])
    });
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
