// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::chunks::DeltaChunks;
use crate::take_chunks::check_indices;
use crate::take_chunks::nullable_positions;
use crate::take_chunks::take_grouped;
use crate::take_chunks::valid_positions;

/// Above one index per this many rows, decoding the whole array and then taking is faster than
/// grouping the indices by chunk.
const FULL_ARRAY_DECODE_RATIO: usize = 8;

/// Take from a Delta array by decoding each chunk that holds a taken value once.
///
/// The indices are grouped by chunk first, since a Delta chunk has to be decoded whole to read any
/// value of it.
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
                take_from_chunks(&chunks, array, indices.as_slice::<I>(), &index_validity)?
                    .into_byte_buffer()
            })
        });
        Ok(Some(
            PrimitiveArray::from_byte_buffer(taken, ptype, validity).into_array(),
        ))
    }
}

/// The values of `array` at `indices`, read from its decoded `chunks`.
fn take_from_chunks<T: Copy, I: IntegerPType>(
    chunks: &impl ChunkDecoder<T>,
    array: ArrayView<'_, Delta>,
    indices: &[I],
    index_validity: &Mask,
) -> VortexResult<Buffer<T>> {
    check_indices(indices, index_validity, array.len())?;
    let offset = array.offset();
    let num_chunks = (offset + array.len()).div_ceil(FL_CHUNK_SIZE);
    // Reading any value of a chunk decodes all of it, so never decode selectively.
    Ok(match index_validity {
        Mask::AllTrue(_) => take_grouped(chunks, valid_positions(indices, offset), num_chunks, 0),
        _ => take_grouped(
            chunks,
            nullable_positions(indices, index_validity, offset),
            num_chunks,
            0,
        ),
    })
}
