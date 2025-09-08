// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use arrayref::{array_mut_ref, array_ref};
use fastlanes::RLE;
use vortex_array::arrays::{PrimitiveArray, PrimitiveVTable};
use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
use vortex_array::vtable::ValidityHelper;
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;

use crate::{FL_CHUNK_SIZE, RLEArray};

impl RLEArray {
    /// Encodes a primitive array of unsigned integers using FastLanes RLE.
    pub fn encode(array: &PrimitiveArray) -> VortexResult<Self> {
        vortex_dtype::match_each_unsigned_integer_ptype!(array.ptype(), |T| {
            rle_encode_typed::<T>(array)
        })
    }
}

/// Decompresses an RLE array back into a primitive array.
pub fn rle_decompress(array: &RLEArray) -> PrimitiveArray {
    match_each_unsigned_integer_ptype!(array.ptype(), |T| { rle_decode_typed::<T>(array) })
}

/// Encodes a primitive array of unsigned integers using FastLanes RLE.
///
/// In case the input array length is % 1024 != 0, the last chunk will be padded.
/// This allows for decompressing RLE arrays without branching on the chunk size.
fn rle_encode_typed<T>(array: &PrimitiveArray) -> VortexResult<RLEArray>
where
    T: NativePType + RLE + Clone + Hash + Eq,
{
    let values = array.as_slice::<T>();
    let len = values.len();

    // Allocate capacity up to the next multiple of chunk size.
    let mut values_buf = BufferMut::<T>::with_capacity(len.next_multiple_of(FL_CHUNK_SIZE));
    let mut indices_buf = BufferMut::<u16>::with_capacity(len.next_multiple_of(FL_CHUNK_SIZE));
    let mut value_chunk_offsets = BufferMut::<u64>::with_capacity(len);

    let values_uninit = values_buf.spare_capacity_mut();
    let indices_uninit = indices_buf.spare_capacity_mut();
    let mut value_count_acc = 0; // Value count accumulator.

    // Note that the loop iterates the last chunk even if partial.
    for chunk_start_idx in (0..len).step_by(FL_CHUNK_SIZE) {
        let chunk_end = std::cmp::min(chunk_start_idx + FL_CHUNK_SIZE, len);
        let chunk_len = chunk_end - chunk_start_idx;
        let chunk_slice = &values[chunk_start_idx..chunk_end];

        // SAFETY:
        // `MaybeUninit<T>` and `T` have the same layout.
        // `MaybeUninit<u16>` and `u16` have the same layout.
        let rle_vals: &mut [T] = unsafe {
            std::mem::transmute(
                &mut values_uninit[value_count_acc..value_count_acc + FL_CHUNK_SIZE],
            )
        };
        let rle_idxs: &mut [u16] = unsafe {
            std::mem::transmute(
                &mut indices_uninit[chunk_start_idx..chunk_start_idx + FL_CHUNK_SIZE],
            )
        };

        // Capture chunk start indices. This is necessary as indices
        // returned from `T::encode` are relative to the chunk.
        //
        // The first chunk offset is always 0.
        value_chunk_offsets.push(value_count_acc as u64);

        let value_count = if chunk_len == FL_CHUNK_SIZE {
            T::encode(
                array_ref![chunk_slice, 0, FL_CHUNK_SIZE],
                array_mut_ref![rle_vals, 0, FL_CHUNK_SIZE],
                array_mut_ref![rle_idxs, 0, FL_CHUNK_SIZE],
            )
        } else {
            // Repeat the last value for padding to prevent
            // accounting for an additional value change.
            let mut padded_chunk = [values[chunk_end - 1]; FL_CHUNK_SIZE];
            padded_chunk[..chunk_len].copy_from_slice(chunk_slice);

            T::encode(
                &padded_chunk,
                array_mut_ref![rle_vals, 0, FL_CHUNK_SIZE],
                array_mut_ref![rle_idxs, 0, FL_CHUNK_SIZE],
            )
        };

        value_count_acc += value_count;
    }

    unsafe {
        values_buf.set_len(value_count_acc);
        indices_buf.set_len(array.len().next_multiple_of(FL_CHUNK_SIZE));
    }

    RLEArray::try_new(
        values_buf.into_array(),
        indices_buf.into_array(),
        value_chunk_offsets.into_array(),
        array.validity().clone(),
        array.len(),
    )
}

/// Decompresses an `RLEArray` into to a primitive array of unsigned integers.
fn rle_decode_typed<T>(array: &RLEArray) -> PrimitiveArray
where
    T: NativePType + RLE + Clone + Copy,
{
    let values = array.values().as_::<PrimitiveVTable>().as_slice::<T>();
    let indices = array.indices().as_::<PrimitiveVTable>().as_slice::<u16>();
    let validity_mask = array.validity_mask();

    let chunk_start_idx = array.offset / FL_CHUNK_SIZE;
    let chunk_end_idx = (array.offset() + array.len()).div_ceil(FL_CHUNK_SIZE);
    let num_chunks = chunk_end_idx - chunk_start_idx;

    let mut builder = PrimitiveBuilder::<T>::with_capacity(
        array.validity().nullability(),
        num_chunks * FL_CHUNK_SIZE,
    );

    let mut range = builder.uninit_range(num_chunks * FL_CHUNK_SIZE);

    for (iter_idx, chunk_idx) in (chunk_start_idx..chunk_end_idx).enumerate() {
        let chunk_values = &values[array.value_chunk_offset(chunk_idx)..];
        let chunk_indices = &indices[chunk_idx * FL_CHUNK_SIZE..];

        // SAFETY: `MaybeUninit<T>` and `T` have the same layout.
        let builder_values: &mut [T] = unsafe {
            std::mem::transmute(range.slice_uninit_mut(iter_idx * FL_CHUNK_SIZE, FL_CHUNK_SIZE))
        };

        T::decode(
            chunk_values,
            array_ref![chunk_indices, 0, FL_CHUNK_SIZE],
            array_mut_ref![builder_values, 0, FL_CHUNK_SIZE],
        );
    }

    unsafe {
        range.finish();
    }

    let offset_within_chunk = array.offset_in_chunk(array.offset);

    builder.set_validity(validity_mask);
    builder
        .finish_into_primitive()
        .slice(offset_within_chunk..(offset_within_chunk + array.len()))
        .to_primitive()
}

#[cfg(test)]
mod test {
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::Buffer;

    use super::*;

    #[test]
    fn test_different_types() {
        // u8
        let values_u8: Buffer<u8> = [1, 1, 2, 2, 3, 3].iter().copied().collect();
        let array_u8 = values_u8.into_array();
        let encoded_u8 = RLEArray::encode(&array_u8.to_primitive()).unwrap();
        let decoded_u8 = encoded_u8.to_primitive();
        assert_eq!(decoded_u8.as_slice::<u8>(), &[1, 1, 2, 2, 3, 3]);

        // u16
        let values_u16: Buffer<u16> = [100, 100, 200, 200].iter().copied().collect();
        let array_u16 = values_u16.into_array();
        let encoded_u16 = RLEArray::encode(&array_u16.to_primitive()).unwrap();
        let decoded_u16 = encoded_u16.to_primitive();
        assert_eq!(decoded_u16.as_slice::<u16>(), &[100, 100, 200, 200]);

        // u64
        let values_u64: Buffer<u64> = [1000, 1000, 2000].iter().copied().collect();
        let array_u64 = values_u64.into_array();
        let encoded_u64 = RLEArray::encode(&array_u64.to_primitive()).unwrap();
        let decoded_u64 = encoded_u64.to_primitive();
        assert_eq!(decoded_u64.as_slice::<u64>(), &[1000, 1000, 2000]);
    }

    #[test]
    fn test_length() {
        let values: Buffer<u32> = [1, 1, 2, 2, 2, 3].iter().copied().collect();
        let array = values.into_array();
        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();
        assert_eq!(encoded.len(), 6);
    }

    #[test]
    fn test_empty_length() {
        let values: Buffer<u32> = Buffer::empty();
        let array = values.into_array();
        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();

        assert_eq!(encoded.len(), 0);
        assert_eq!(encoded.values.len(), 0);
    }

    #[test]
    fn test_encode_decode() {
        let values: Buffer<u32> = [1, 1, 1, 2, 2, 3, 3, 3, 3].iter().copied().collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();
        assert_eq!(encoded.values.len(), 3);

        let decoded = encoded.to_primitive(); // Verify round-trip
        assert_eq!(decoded.as_slice::<u32>(), &[1, 1, 1, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_single_value() {
        let values: Buffer<u16> = vec![42; 2000].into_iter().collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();
        assert_eq!(encoded.values.len(), 2); // 2 chunks, each storing value 42

        let decoded = encoded.to_primitive(); // Verify round-trip
        assert_eq!(decoded.as_slice::<u16>(), &vec![42; 2000]);
    }

    #[test]
    fn test_all_different() {
        let values: Buffer<u8> = (0u8..=255).collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();
        assert_eq!(encoded.values.len(), 256);

        let decoded = encoded.to_primitive(); // Verify round-trip
        assert_eq!(decoded.as_slice::<u8>(), &(0u8..=255).collect::<Vec<_>>());
    }

    #[test]
    fn test_partial_last_chunk() {
        // Test array with partial last chunk (not divisible by 1024)
        let values: Buffer<u32> = (0..1500).map(|i| (i / 100) as u32).collect();
        let expected: Vec<u32> = (0..1500).map(|i| (i / 100) as u32).collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();
        let decoded = encoded.to_primitive();

        assert_eq!(encoded.len(), 1500);
        assert_eq!(decoded.as_slice::<u32>(), expected.as_slice());
        // Should have 2 chunks: 1024 + 476 elements
        assert_eq!(encoded.value_chunk_offsets().len(), 2);
    }

    #[test]
    fn test_multi_chunk() {
        // Test array that spans exactly 2 chunks (2048 elements)
        let values: Buffer<u32> = (0..2048).map(|i| (i / 100) as u32).collect();
        let expected: Vec<u32> = (0..2048).map(|i| (i / 100) as u32).collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();
        let decoded = encoded.to_primitive();

        assert_eq!(encoded.len(), 2048);
        assert_eq!(decoded.as_slice::<u32>(), expected.as_slice());
        assert_eq!(encoded.value_chunk_offsets().len(), 2);
    }

    #[test]
    fn test_chunk_boundary_access() {
        // Test accessing elements around chunk boundaries
        let values: Buffer<u16> = (0..3000).map(|i| (i / 50) as u16).collect();
        let expected: Vec<u16> = (0..3000).map(|i| (i / 50) as u16).collect();
        let array = values.into_array();

        let encoded = RLEArray::encode(&array.to_primitive()).unwrap();

        // Test access at chunk boundaries
        for &idx in &[1023, 1024, 1025, 2047, 2048, 2049] {
            if idx < encoded.len() {
                let original_value = expected[idx];
                let encoded_value = encoded.scalar_at(idx).as_primitive().as_::<u16>().unwrap();
                assert_eq!(original_value, encoded_value, "Mismatch at index {}", idx);
            }
        }
    }
}
