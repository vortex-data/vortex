// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrayref::array_mut_ref;
use arrayref::array_ref;
use fastlanes::RLE;
use num_traits::AsPrimitive;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::FL_CHUNK_SIZE;
use crate::RLEArray;

/// Decompresses an RLE array back into a primitive array.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is from nested match_each_* macros"
)]
pub fn rle_decompress(array: &RLEArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    match_each_native_ptype!(array.values().dtype().as_ptype(), |V| {
        match_each_unsigned_integer_ptype!(array.values_idx_offsets().dtype().as_ptype(), |O| {
            // RLE indices are always u16 (or u8 if downcasted).
            match array.indices().dtype().as_ptype() {
                PType::U8 => rle_decode_typed::<V, u8, O>(array, ctx),
                PType::U16 => rle_decode_typed::<V, u16, O>(array, ctx),
                _ => vortex_panic!(
                    "Unsupported index type for RLE decoding: {}",
                    array.indices().dtype().as_ptype()
                ),
            }
        })
    })
}

/// Decompresses an `RLEArray` into to a primitive array of unsigned integers.
fn rle_decode_typed<V, I, O>(
    array: &RLEArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray>
where
    V: NativePType + RLE + Clone + Copy,
    I: NativePType + Into<usize>,
    O: NativePType + AsPrimitive<u64>,
{
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    let values = values.as_slice::<V>();

    let indices = array.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let indices = indices.as_slice::<I>();
    assert!(indices.len().is_multiple_of(FL_CHUNK_SIZE));

    let chunk_start_idx = array.offset() / FL_CHUNK_SIZE;
    let chunk_end_idx = (array.offset() + array.len()).div_ceil(FL_CHUNK_SIZE);
    let num_chunks = chunk_end_idx - chunk_start_idx;

    let mut buffer = BufferMut::<V>::with_capacity(num_chunks * FL_CHUNK_SIZE);
    let buffer_uninit = buffer.spare_capacity_mut();

    let values_idx_offsets = array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let values_idx_offsets = values_idx_offsets.as_slice::<O>();

    for chunk_idx in 0..num_chunks {
        // Offsets in `values_idx_offsets` are absolute and need to be shifted
        // by the offset of the first chunk, respective the current slice, in
        // order to make them relative.
        let value_idx_offset =
            (values_idx_offsets[chunk_idx].as_() - values_idx_offsets[0].as_()) as usize;

        let chunk_values = &values[value_idx_offset..];
        let chunk_indices = &indices[chunk_idx * FL_CHUNK_SIZE..];

        // SAFETY: `MaybeUninit<T>` and `T` have the same layout.
        let buffer_values: &mut [V] = unsafe {
            std::mem::transmute(&mut buffer_uninit[chunk_idx * FL_CHUNK_SIZE..][..FL_CHUNK_SIZE])
        };

        V::decode(
            chunk_values,
            array_ref![chunk_indices, 0, FL_CHUNK_SIZE],
            array_mut_ref![buffer_values, 0, FL_CHUNK_SIZE],
        );
    }

    unsafe {
        buffer.set_len(num_chunks * FL_CHUNK_SIZE);
    }

    let offset_within_chunk = array.offset();

    Ok(PrimitiveArray::new(
        buffer
            .freeze()
            .slice(offset_within_chunk..(offset_within_chunk + array.len())),
        Validity::copy_from_array(&array.clone().into_array())?,
    ))
}
