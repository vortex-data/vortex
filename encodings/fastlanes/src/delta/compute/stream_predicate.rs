// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming, cache-resident predicate evaluation over a [`DeltaArray`].
//!
//! A [`DeltaArray`] stores its values as FastLanes delta chunks of exactly 1024 elements. The
//! canonical "decompress, then compare" path materialises the full primitive buffer on the heap
//! (one `T` per element) and then walks it a second time to evaluate the predicate, streaming the
//! whole array through memory twice.
//!
//! This module instead decompresses one 1024-element chunk at a time into a pair of stack scratch
//! buffers (undelta then untranspose, exactly as [`super::super::array::delta_decompress`] does),
//! folds a `Fn(T) -> bool` predicate over the chunk while it is still hot in L1, and packs the
//! resulting bools straight into the output [`vortex_buffer::BitBuffer`] via
//! [`pack_bools_into_words`] (the same word-at-a-time shape as
//! [`vortex_buffer::BitBuffer::collect_bool`]). The materialised primitive never exists.

use fastlanes::Delta as FlDelta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::pack_bools_into_words;
use vortex_error::VortexResult;

use crate::Delta;
use crate::bit_transpose::untranspose_validity;
use crate::delta::array::DeltaArrayExt;

const CHUNK_SIZE: usize = 1024;

/// Stream `predicate` over the logical values of a [`DeltaArray`], one FastLanes chunk at a time,
/// producing a [`BoolArray`].
///
/// `predicate` operates in the array's *original* (possibly signed) domain. Decompression runs in
/// the unsigned domain — `wrapping_add` over the raw bytes inverts the `wrapping_sub` applied at
/// compress time — and each decompressed word is reinterpreted back to `T` before the predicate
/// sees it, mirroring the `reinterpret_cast` in the eager decompress path.
pub(super) fn stream_predicate<T, P>(
    array: ArrayView<'_, Delta>,
    nullability: Nullability,
    predicate: P,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + Copy,
    P: Fn(T) -> bool,
{
    let len = array.len();

    let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
    let deltas = array.deltas().clone().execute::<PrimitiveArray>(ctx)?;

    let start = array.offset();
    let end = start + len;

    // Validity is stored transposed alongside the deltas; recover and slice it to the logical
    // window, exactly as the eager decompress path does.
    let validity = untranspose_validity(&deltas.validity()?, ctx)?.slice(start..end)?;

    let original_ptype = deltas.ptype();
    let bases = bases.reinterpret_cast(original_ptype.to_unsigned());
    let deltas = deltas.reinterpret_cast(original_ptype.to_unsigned());

    let mut words: BufferMut<u64> = BufferMut::zeroed(len.div_ceil(u64::BITS as usize));

    if len > 0 {
        match_each_unsigned_integer_ptype!(deltas.ptype(), |U| {
            const LANES: usize = U::LANES;
            debug_assert_eq!(size_of::<U>(), size_of::<T>());
            stream_blocks::<U, T, LANES, _>(
                bases.as_slice::<U>(),
                deltas.as_slice::<U>(),
                start,
                end,
                words.as_mut_slice(),
                &predicate,
            );
        });
    }

    let bits = BitBufferMut::from_buffer(words.into_byte_buffer(), 0, len);
    let validity = validity.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Decompress each 1024-element chunk into stack scratch buffers, then fold `predicate` over the
/// logical window `[start, end)` of that chunk, packing bools into `words` at output position
/// `global_index - start`.
fn stream_blocks<U, T, const LANES: usize, P>(
    bases: &[U],
    deltas: &[U],
    start: usize,
    end: usize,
    words: &mut [u64],
    predicate: &P,
) where
    U: NativePType + FlDelta + Transpose + FastLanes,
    T: NativePType + Copy,
    P: Fn(T) -> bool,
{
    let (chunks, remainder) = deltas.as_chunks::<CHUNK_SIZE>();
    debug_assert!(
        remainder.is_empty(),
        "deltas must be padded to a multiple of {CHUNK_SIZE}"
    );

    let mut transposed = [U::default(); CHUNK_SIZE];
    let mut decoded = [U::default(); CHUNK_SIZE];

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_start = i * CHUNK_SIZE;
        let chunk_end = chunk_start + CHUNK_SIZE;
        // Skip chunks fully outside the logical window (slicing can clip both ends).
        if chunk_end <= start || chunk_start >= end {
            continue;
        }

        // SAFETY: `bases` holds `LANES` entries per chunk; `i` ranges over `chunks`.
        let base_ref = unsafe { &*(bases[i * LANES..(i + 1) * LANES].as_ptr().cast()) };
        FlDelta::undelta::<LANES>(chunk, base_ref, &mut transposed);
        Transpose::untranspose(&transposed, &mut decoded);

        // Reinterpret the unsigned-domain chunk back to the original `T`. `U` is `T`'s unsigned
        // counterpart so the layouts are identical (checked in the caller).
        // SAFETY: `size_of::<U>() == size_of::<T>()` and both are `Copy` plain-old-data.
        let values: &[T; CHUNK_SIZE] = unsafe { &*(decoded.as_ptr().cast()) };

        let lo = start.max(chunk_start);
        let hi = end.min(chunk_end);
        let local_lo = lo - chunk_start;
        let count = hi - lo;
        let out_offset = lo - start;
        pack_bools_into_words(words, out_offset, count, |k| {
            predicate(values[local_lo + k])
        });
    }
}
