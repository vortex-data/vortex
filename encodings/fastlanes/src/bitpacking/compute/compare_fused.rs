// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-streaming compare kernel for [`BitPackedArray`] against a constant.
//!
//! Walks the encoded array one 1024-element FastLanes block at a time through the regular
//! [`crate::unpack_iter::BitUnpackedChunks`] iterator (the same path used by decompress,
//! `is_constant`, and `stream_predicate`), splices any [`crate::patches::Patches`] into the
//! unpacked block in place, then folds `cmp(value, rhs)` over the block into a [`BitBuffer`].
//! The materialised primitive never appears: each block reuses a single scratch buffer and the
//! per-element bool is packed straight into the output words.

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
use vortex_error::VortexResult;

use super::stream_predicate::splice_patches;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedIter;

/// Compare the unpacked values of a [`BitPackedArray`] against `rhs`, one FastLanes block at a
/// time, producing a [`BoolArray`].
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
    T: NativePType + BitPackedIter + Copy,
    F: Fn(T, T) -> bool,
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
                    pack_bools_into_words(words, range.start, block.len(), |i| cmp(block[i], rhs));
                });
            });
        } else {
            chunks.for_each_unpacked_chunk(|block, range| {
                pack_bools_into_words(words, range.start, block.len(), |i| cmp(block[i], rhs));
            });
        }
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);
    let validity = array.validity()?.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}
