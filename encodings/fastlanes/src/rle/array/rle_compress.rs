// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrayref::array_mut_ref;
use fastlanes::RLE;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::RLEArray;
use crate::RLEData;
use crate::fill_forward_nulls;

impl RLEData {
    /// Encodes a primitive array of unsigned integers using FastLanes RLE.
    pub fn encode(array: &PrimitiveArray) -> VortexResult<RLEArray> {
        match_each_native_ptype!(array.ptype(), |T| { rle_encode_typed::<T>(array) })
    }
}

/// Encodes a primitive array of unsigned integers using FastLanes RLE.
///
/// In case the input array length is % 1024 != 0, the last chunk is padded.
fn rle_encode_typed<T>(array: &PrimitiveArray) -> VortexResult<RLEArray>
where
    T: NativePType + RLE,
    NativeValue<T>: RLE,
{
    // Fill-forward null values so the RLE encoder doesn't see garbage at null positions,
    // which would create spurious run boundaries and inflate the dictionary.
    let values = fill_forward_nulls(array.to_buffer::<T>(), &array.validity());
    let len = values.len();
    let padded_len = len.next_multiple_of(FL_CHUNK_SIZE);

    // Allocate capacity up to the next multiple of chunk size.
    let mut values_buf = BufferMut::<NativeValue<T>>::with_capacity(padded_len);
    let mut indices_buf = BufferMut::<u16>::with_capacity(padded_len);

    // Pre-allocate for one offset per chunk.
    let mut values_idx_offsets = BufferMut::<u64>::with_capacity(len.div_ceil(FL_CHUNK_SIZE));

    let values_uninit = values_buf.spare_capacity_mut();
    let indices_uninit = indices_buf.spare_capacity_mut();
    let mut value_count_acc = 0; // Chunk value count prefix sum.

    let (chunks, remainder) = values.as_chunks::<FL_CHUNK_SIZE>();

    let mut process_chunk = |chunk_start_idx: usize, input: &[T; FL_CHUNK_SIZE]| {
        // SAFETY: NativeValue is repr(transparent)
        let input: &[NativeValue<T>; FL_CHUNK_SIZE] = unsafe { std::mem::transmute(input) };

        // SAFETY: `MaybeUninit<NativeValue<T>>` and `NativeValue<T>` have the same layout.
        let rle_vals: &mut [NativeValue<T>] =
            unsafe { std::mem::transmute(&mut values_uninit[value_count_acc..][..FL_CHUNK_SIZE]) };

        // SAFETY: `MaybeUninit<u16>` and `u16` have the same layout.
        let rle_idxs: &mut [u16] =
            unsafe { std::mem::transmute(&mut indices_uninit[chunk_start_idx..][..FL_CHUNK_SIZE]) };

        // Capture chunk start indices. This is necessary as indices
        // returned from `T::encode` are relative to the chunk.
        values_idx_offsets.push(value_count_acc as u64);

        let value_count = NativeValue::<T>::encode(
            input,
            array_mut_ref![rle_vals, 0, FL_CHUNK_SIZE],
            array_mut_ref![rle_idxs, 0, FL_CHUNK_SIZE],
        );

        value_count_acc += value_count;
    };

    for (chunk_idx, chunk_slice) in chunks.iter().enumerate() {
        process_chunk(chunk_idx * FL_CHUNK_SIZE, chunk_slice);
    }

    if !remainder.is_empty() {
        // Repeat the last value for padding to prevent
        // accounting for an additional value change.
        let mut padded_chunk = [values[len - 1]; FL_CHUNK_SIZE];
        padded_chunk[..remainder.len()].copy_from_slice(remainder);
        process_chunk((len / FL_CHUNK_SIZE) * FL_CHUNK_SIZE, &padded_chunk);
    }

    unsafe {
        values_buf.set_len(value_count_acc);
        indices_buf.set_len(padded_len);
    }

    // SAFETY: NativeValue<T> is repr(transparent) to T.
    let values_buf = unsafe { values_buf.transmute::<T>().freeze() };

    RLEArray::try_from_data(RLEData::try_new(
        values_buf.into_array(),
        PrimitiveArray::new(indices_buf.freeze(), padded_validity(array)).into_array(),
        values_idx_offsets.into_array(),
        0,
        array.len(),
    )?)
}

/// Returns validity padded to the next 1024 chunk for a given array.
fn padded_validity(array: &PrimitiveArray) -> Validity {
    match array.validity() {
        Validity::NonNullable => Validity::NonNullable,
        Validity::AllValid => Validity::AllValid,
        Validity::AllInvalid => Validity::AllInvalid,
        Validity::Array(validity_array) => {
            let len = array.len();
            let padded_len = len.next_multiple_of(FL_CHUNK_SIZE);

            if len == padded_len {
                return Validity::Array(validity_array);
            }

            let mut builder = BitBufferMut::with_capacity(padded_len);

            let bool_array = validity_array.to_bool();
            builder.append_buffer(&bool_array.to_bit_buffer());
            builder.append_n(false, padded_len - len);

            Validity::from(builder.freeze())
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::half::f16;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn test_encode_decode() {
        // u8
        let array_u8: Buffer<u8> = buffer![1, 1, 2, 2, 3, 3];
        let encoded_u8 =
            RLEData::encode(&PrimitiveArray::new(array_u8, Validity::NonNullable)).unwrap();
        let decoded_u8 = encoded_u8.as_array().to_primitive();
        let expected_u8 = PrimitiveArray::from_iter(vec![1u8, 1, 2, 2, 3, 3]);
        assert_arrays_eq!(decoded_u8, expected_u8);

        // u16
        let array_u16: Buffer<u16> = buffer![100, 100, 200, 200];
        let encoded_u16 =
            RLEData::encode(&PrimitiveArray::new(array_u16, Validity::NonNullable)).unwrap();
        let decoded_u16 = encoded_u16.as_array().to_primitive();
        let expected_u16 = PrimitiveArray::from_iter(vec![100u16, 100, 200, 200]);
        assert_arrays_eq!(decoded_u16, expected_u16);

        // u64
        let array_u64: Buffer<u64> = buffer![1000, 1000, 2000];
        let encoded_u64 =
            RLEData::encode(&PrimitiveArray::new(array_u64, Validity::NonNullable)).unwrap();
        let decoded_u64 = encoded_u64.as_array().to_primitive();
        let expected_u64 = PrimitiveArray::from_iter(vec![1000u64, 1000, 2000]);
        assert_arrays_eq!(decoded_u64, expected_u64);
    }

    #[test]
    fn test_length() {
        let values: Buffer<u32> = buffer![1, 1, 2, 2, 2, 3];
        let encoded = RLEData::encode(&PrimitiveArray::new(values, Validity::NonNullable)).unwrap();
        assert_eq!(encoded.len(), 6);
    }

    #[test]
    fn test_empty_length() {
        let values: Buffer<u32> = Buffer::empty();
        let encoded = RLEData::encode(&PrimitiveArray::new(values, Validity::NonNullable)).unwrap();

        assert_eq!(encoded.len(), 0);
        assert_eq!(encoded.values().len(), 0);
    }

    #[test]
    fn test_single_value() {
        let values: Buffer<u16> = vec![42; 2000].into_iter().collect();

        let encoded = RLEData::encode(&PrimitiveArray::new(values, Validity::NonNullable)).unwrap();
        assert_eq!(encoded.values().len(), 2); // 2 chunks, each storing value 42

        let decoded = encoded.as_array().to_primitive(); // Verify round-trip
        let expected = PrimitiveArray::from_iter(vec![42u16; 2000]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn test_all_different() {
        let values: Buffer<u8> = (0u8..=255).collect();

        let encoded = RLEData::encode(&PrimitiveArray::new(values, Validity::NonNullable)).unwrap();
        assert_eq!(encoded.values().len(), 256);

        let decoded = encoded.as_array().to_primitive(); // Verify round-trip
        let expected = PrimitiveArray::from_iter((0u8..=255).collect::<Vec<_>>());
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn test_partial_last_chunk() {
        // Test array with partial last chunk (not divisible by 1024)
        let values: Buffer<u32> = (0..1500).map(|i| (i / 100) as u32).collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);

        let encoded = RLEData::encode(&array).unwrap();

        assert_eq!(encoded.len(), 1500);
        assert_arrays_eq!(encoded, array);
        // 2 chunks: 1024 + 476 elements
        assert_eq!(encoded.values_idx_offsets().len(), 2);
    }

    #[test]
    fn test_two_full_chunks() {
        // Array that spans exactly 2 chunks (2048 elements)
        let values: Buffer<u32> = (0..2048).map(|i| (i / 100) as u32).collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);

        let encoded = RLEData::encode(&array).unwrap();

        assert_eq!(encoded.len(), 2048);
        assert_arrays_eq!(encoded, array);
        assert_eq!(encoded.values_idx_offsets().len(), 2);
    }

    #[rstest]
    #[case::u8((0u8..100).collect::<Buffer<u8>>())]
    #[case::u16((0u16..2000).collect::<Buffer<u16>>())]
    #[case::u32((0u32..2000).collect::<Buffer<u32>>())]
    #[case::u64((0u64..2000).collect::<Buffer<u64>>())]
    #[case::i8((-100i8..100).collect::<Buffer<i8>>())]
    #[case::i16((-2000i16..2000).collect::<Buffer<i16>>())]
    #[case::i32((-2000i32..2000).collect::<Buffer<i32>>())]
    #[case::i64((-2000i64..2000).collect::<Buffer<i64>>())]
    #[case::f16((-2000..2000).map(|i| f16::from_f32(i as f32)).collect::<Buffer<f16>>())]
    #[case::f32((-2000..2000).map(|i| i as f32).collect::<Buffer<f32>>())]
    #[case::f64((-2000..2000).map(|i| i as f64).collect::<Buffer<f64>>())]
    fn test_roundtrip_primitive_types<T: NativePType>(#[case] values: Buffer<T>) {
        let primitive = values.clone().into_array().to_primitive();
        let result = RLEData::encode(&primitive).unwrap();
        let decoded = result.as_array().to_primitive();
        let expected = PrimitiveArray::new(values, primitive.validity());
        assert_arrays_eq!(decoded, expected);
    }

    // Regression test: RLE compression properly supports decoding pos/neg zeros
    // See <https://github.com/vortex-data/vortex/issues/6491>
    #[rstest]
    #[case(vec![f16::ZERO, f16::NEG_ZERO])]
    #[case(vec![0f32, -0f32])]
    #[case(vec![0f64, -0f64])]
    fn test_float_zeros<T: NativePType + RLE>(#[case] values: Vec<T>) {
        let primitive = PrimitiveArray::from_iter(values);
        let rle = RLEData::encode(&primitive).unwrap();
        let decoded = rle.as_array().to_primitive();
        assert_arrays_eq!(primitive, decoded);
    }
}
