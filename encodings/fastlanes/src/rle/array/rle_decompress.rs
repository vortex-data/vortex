// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::RLE;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::FL_CHUNK_SIZE;
use crate::RLEArray;
use crate::rle::RLEArrayExt;

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
    assert!(indices.len().is_multiple_of(FL_CHUNK_SIZE));
    let has_invalid = !indices.all_valid(ctx)?;
    let (indices_sl, _) = indices.as_slice::<I>().as_chunks::<FL_CHUNK_SIZE>();

    let chunk_start_idx = array.offset() / FL_CHUNK_SIZE;
    let chunk_end_idx = (array.offset() + array.len()).div_ceil(FL_CHUNK_SIZE);
    let num_chunks = chunk_end_idx - chunk_start_idx;

    let mut buffer = BufferMut::<V>::with_capacity(num_chunks * FL_CHUNK_SIZE);
    let (out_buf, _) = buffer.spare_capacity_mut().as_chunks_mut::<FL_CHUNK_SIZE>();

    let values_idx_offsets = array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let values_idx_offsets = values_idx_offsets.as_slice::<O>();

    for (chunk_idx, (chunk_indices, chunk_out)) in
        indices_sl.iter().zip(out_buf.iter_mut()).enumerate()
    {
        // Offsets in `values_idx_offsets` are absolute and need to be shifted
        // by the offset of the first chunk, respective of the current slice,
        // to make them relative.
        let value_idx_offset =
            (values_idx_offsets[chunk_idx].as_() - values_idx_offsets[0].as_()) as usize;

        let next_value_idx_offset = if chunk_idx + 1 < num_chunks {
            (values_idx_offsets[chunk_idx + 1].as_() - values_idx_offsets[0].as_()) as usize
        } else {
            values.len()
        };
        let num_chunk_values = u16::try_from(next_value_idx_offset - value_idx_offset)
            .vortex_expect("There can be at most 1024 values in RLE chunk");

        // SAFETY: `MaybeUninit<T>` and `T` have the same layout.
        let buffer_values: &mut [V; FL_CHUNK_SIZE] = unsafe { std::mem::transmute(chunk_out) };
        let chunk_values = &values[value_idx_offset..];
        if num_chunk_values == 1 {
            // Single-value chunk: fill directly to avoid out-of-bounds index
            // access. The indices may contain values other than 0 when they
            // have been further compressed (e.g., as a masked constant).
            buffer_values.fill(chunk_values[0]);
        } else if has_invalid {
            // When the indices array has invalid (null) positions, those
            // positions may contain arbitrary garbage values after further
            // compression. Clamp all indices into [0, num_chunk_values) to
            // prevent out-of-bounds access in the fastlanes decoder.
            let mut sanitized: [u16; FL_CHUNK_SIZE] = [0; FL_CHUNK_SIZE];
            for (idx_out, idx) in sanitized.iter_mut().zip(chunk_indices) {
                let idx: u16 =
                    NumCast::from(*idx).vortex_expect("RLE indices are always less than u16");
                *idx_out = idx.min(num_chunk_values - 1);
            }
            V::decode(chunk_values, &sanitized, buffer_values);
        } else {
            V::decode(chunk_values, chunk_indices, buffer_values);
        }
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
