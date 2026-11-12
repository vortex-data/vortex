// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use fastlanes::FastLanes;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::ChunkDecoder;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::chunks::DeltaChunks;

/// Above one index per this many rows, decoding the whole array and then taking is faster than
/// grouping the indices by chunk.
const FULL_ARRAY_DECODE_RATIO: usize = 8;

/// Take from a Delta array by decoding each chunk that holds a taken value once.
impl TakeExecute for Delta {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if indices.len() * FULL_ARRAY_DECODE_RATIO > array.len() {
            let decoded = array.array().clone().execute::<PrimitiveArray>(ctx)?;
            return decoded.into_array().take(indices.clone()).map(Some);
        }

        let validity = array.validity()?.take(indices)?;
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let index_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
        let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
        let ptype = bases.ptype();
        // Deltas are decoded as unsigned; `wrapping_add` is the same for either signedness.
        let bases = bases.reinterpret_cast(ptype.to_unsigned());
        let taken = match_each_unsigned_integer_ptype!(bases.ptype(), |T| {
            let mut deltas = None;
            let chunks = DeltaChunks::<T, { T::LANES }>::for_array(
                array,
                bases.as_slice(),
                &mut deltas,
                ctx,
            )?;
            match_each_integer_ptype!(indices.ptype(), |I| {
                take_chunked::<T, I>(
                    &chunks,
                    array.offset(),
                    array.len(),
                    indices.as_slice::<I>(),
                    &index_validity,
                )?
                .into_byte_buffer()
            })
        });
        Ok(Some(
            PrimitiveArray::from_byte_buffer(taken, ptype, validity).into_array(),
        ))
    }
}

/// The values at `indices` of the chunks of `chunks` from `offset` on, which hold `len` values.
///
/// Each taken chunk is decoded once. Null indices take an unspecified value.
fn take_chunked<T: NativePType, I: IntegerPType>(
    chunks: &impl ChunkDecoder<T>,
    offset: usize,
    len: usize,
    indices: &[I],
    index_validity: &Mask,
) -> VortexResult<Buffer<T>> {
    // Group the valid indices by chunk with a counting sort, keeping each one's output position.
    let num_chunks = (offset + len).div_ceil(FL_CHUNK_SIZE);
    let mut chunk_starts = vec![0usize; num_chunks + 1];
    let mut positions = Vec::with_capacity(indices.len());
    for (out, index) in indices.iter().enumerate() {
        if !index_validity.value(out) {
            continue;
        }
        let Some(index) = index.to_usize().filter(|&index| index < len) else {
            vortex_bail!("Take index {index} is out of bounds for length {len}");
        };
        let position = offset + index;
        chunk_starts[position / FL_CHUNK_SIZE + 1] += 1;
        positions.push((out, position));
    }
    for chunk in 0..num_chunks {
        chunk_starts[chunk + 1] += chunk_starts[chunk];
    }
    let mut next = chunk_starts.clone();
    let mut grouped = vec![(0usize, 0usize); positions.len()];
    for (out, position) in positions {
        let chunk = position / FL_CHUNK_SIZE;
        grouped[next[chunk]] = (out, position % FL_CHUNK_SIZE);
        next[chunk] += 1;
    }

    let mut output = BufferMut::<T>::zeroed(indices.len());
    let mut values = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
    for chunk in 0..num_chunks {
        let taken = &grouped[chunk_starts[chunk]..chunk_starts[chunk + 1]];
        if taken.is_empty() {
            continue;
        }
        chunks.decode_chunk(chunk, &mut values);
        for &(out, within) in taken {
            // SAFETY: `decode_chunk` initialized every value of the chunk.
            output[out] = unsafe { values[within].assume_init() };
        }
    }
    Ok(output.freeze())
}
