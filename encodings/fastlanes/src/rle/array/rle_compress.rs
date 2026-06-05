// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;

use fastlanes::RLE as FastLanesRLE;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::RLE;
use crate::RLEArray;
use crate::RLEData;
use crate::fill_forward_nulls;

impl RLEData {
    /// Encodes a primitive array of unsigned integers using FastLanes RLE.
    pub fn encode(
        array: ArrayView<'_, Primitive>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RLEArray> {
        let array = array.into_owned();
        match_each_native_ptype!(array.ptype(), |T| { rle_encode_typed::<T>(&array, ctx) })
    }
}

/// Encodes a primitive array of unsigned integers using FastLanes RLE.
///
/// In case the input array length is % 1024 != 0, the last chunk is padded.
fn rle_encode_typed<T>(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<RLEArray>
where
    T: NativePType + FastLanesRLE,
    NativeValue<T>: FastLanesRLE,
{
    // Fill-forward null values so the RLE encoder doesn't see garbage at null positions,
    // which would create spurious run boundaries and inflate the dictionary.
    let values = fill_forward_nulls(array.to_buffer::<T>(), &array.validity()?, ctx)?;
    let len = values.len();
    let padded_len = len.next_multiple_of(FL_CHUNK_SIZE);

    // Allocate capacity up to the next multiple of chunk size.
    let mut values_buf = BufferMut::<NativeValue<T>>::with_capacity(padded_len);
    let mut indices_buf = BufferMut::<u16>::with_capacity(padded_len);

    // Pre-allocate for one offset per chunk.
    let mut values_idx_offsets = BufferMut::<u64>::with_capacity(len.div_ceil(FL_CHUNK_SIZE));

    let values_uninit = values_buf.spare_capacity_mut();
    // We don't care about the trailing chunk that exists due to overallocation by the underlying allocator.
    let (indices_uninit, _) = indices_buf
        .spare_capacity_mut()
        .as_chunks_mut::<FL_CHUNK_SIZE>();
    let mut value_count_acc = 0; // Chunk value count prefix sum.

    let (chunks, remainder) = values.as_chunks::<FL_CHUNK_SIZE>();

    let mut process_chunk =
        |input: &[T; FL_CHUNK_SIZE], rle_idxs: &mut [mem::MaybeUninit<u16>; FL_CHUNK_SIZE]| {
            // SAFETY: NativeValue is repr(transparent)
            let input: &[NativeValue<T>; FL_CHUNK_SIZE] = unsafe { mem::transmute(input) };
            let rle_idxs: &mut [u16; FL_CHUNK_SIZE] = unsafe { mem::transmute(rle_idxs) };

            // SAFETY: `MaybeUninit<NativeValue<T>>` and `NativeValue<T>` have the same layout.
            let rle_vals: &mut [NativeValue<T>] =
                unsafe { mem::transmute(&mut values_uninit[value_count_acc..][..FL_CHUNK_SIZE]) };

            // Capture chunk start indices. This is necessary as indices
            // returned from `T::encode` are relative to the chunk.
            values_idx_offsets.push(value_count_acc as u64);

            let value_count = NativeValue::<T>::encode(
                input,
                unsafe { &mut *(rle_vals.as_mut_ptr() as *mut [_; FL_CHUNK_SIZE]) },
                rle_idxs,
            );

            value_count_acc += value_count;
        };

    for (chunk_slice, rle_idxs) in chunks.iter().zip(indices_uninit.iter_mut()) {
        // SAFETY: `MaybeUninit<u16>` and `u16` have the same layout.
        process_chunk(chunk_slice, rle_idxs);
    }

    if !remainder.is_empty() {
        // Repeat the last value for padding to prevent
        // accounting for an additional value change.
        let mut padded_chunk = [values[len - 1]; FL_CHUNK_SIZE];
        padded_chunk[..remainder.len()].copy_from_slice(remainder);
        // There might be more entries in indices_uninit than necessary if the allocator gave us extra memory.
        // Remainder has to go to the last chunk after full chunks have been processed.
        let last_idx_chunk = &mut indices_uninit[chunks.len()];
        process_chunk(&padded_chunk, last_idx_chunk);
    }

    unsafe {
        values_buf.set_len(value_count_acc);
        indices_buf.set_len(padded_len);
    }

    // SAFETY: NativeValue<T> is repr(transparent) to T.
    let values_buf = unsafe { values_buf.transmute::<T>().freeze() };

    RLE::try_new(
        values_buf.into_array(),
        PrimitiveArray::new(indices_buf.freeze(), padded_validity(array, ctx)?).into_array(),
        values_idx_offsets.into_array(),
        0,
        array.len(),
    )
}

/// Returns validity padded to the next 1024 chunk for a given array.
fn padded_validity(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<Validity> {
    match array
        .validity()
        .vortex_expect("RLE validity should be derivable")
    {
        Validity::NonNullable => Ok(Validity::NonNullable),
        Validity::AllValid => Ok(Validity::AllValid),
        Validity::AllInvalid => Ok(Validity::AllInvalid),
        Validity::Array(validity_array) => {
            let len = array.len();
            let padded_len = len.next_multiple_of(FL_CHUNK_SIZE);

            if len == padded_len {
                return Ok(Validity::Array(validity_array));
            }

            let mut builder = BitBufferMut::with_capacity(padded_len);

            let bool_array = validity_array.execute::<BoolArray>(ctx)?;
            builder.append_buffer(&bool_array.to_bit_buffer());
            builder.append_n(false, padded_len - len);

            Ok(Validity::from(builder.freeze()))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::half::f16;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::rle::array::RLEArrayExt;

    #[test]
    fn test_encode_decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // u8
        let array_u8: Buffer<u8> = buffer![1, 1, 2, 2, 3, 3];
        let encoded_u8 = RLEData::encode(
            PrimitiveArray::new(array_u8, Validity::NonNullable).as_view(),
            &mut ctx,
        )?;
        let decoded_u8 = encoded_u8
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected_u8 = PrimitiveArray::from_iter(vec![1u8, 1, 2, 2, 3, 3]);
        assert_arrays_eq!(decoded_u8, expected_u8);

        // u16
        let array_u16: Buffer<u16> = buffer![100, 100, 200, 200];
        let encoded_u16 = RLEData::encode(
            PrimitiveArray::new(array_u16, Validity::NonNullable).as_view(),
            &mut ctx,
        )?;
        let decoded_u16 = encoded_u16
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected_u16 = PrimitiveArray::from_iter(vec![100u16, 100, 200, 200]);
        assert_arrays_eq!(decoded_u16, expected_u16);

        // u64
        let array_u64: Buffer<u64> = buffer![1000, 1000, 2000];
        let encoded_u64 = RLEData::encode(
            PrimitiveArray::new(array_u64, Validity::NonNullable).as_view(),
            &mut ctx,
        )?;
        let decoded_u64 = encoded_u64
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected_u64 = PrimitiveArray::from_iter(vec![1000u64, 1000, 2000]);
        assert_arrays_eq!(decoded_u64, expected_u64);
        Ok(())
    }

    #[test]
    fn test_length() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Buffer<u32> = buffer![1, 1, 2, 2, 2, 3];
        let encoded = RLEData::encode(
            PrimitiveArray::new(values, Validity::NonNullable).as_view(),
            &mut ctx,
        )
        .unwrap();
        assert_eq!(encoded.len(), 6);
    }

    #[test]
    fn test_empty_length() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Buffer<u32> = Buffer::empty();
        let encoded = RLEData::encode(
            PrimitiveArray::new(values, Validity::NonNullable).as_view(),
            &mut ctx,
        )
        .unwrap();

        assert_eq!(encoded.len(), 0);
        assert_eq!(encoded.values().len(), 0);
    }

    #[test]
    fn test_single_value() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Buffer<u16> = vec![42; 2000].into_iter().collect();

        let encoded = RLEData::encode(
            PrimitiveArray::new(values, Validity::NonNullable).as_view(),
            &mut ctx,
        )?;
        assert_eq!(encoded.values().len(), 2); // 2 chunks, each storing value 42

        let decoded = encoded
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?; // Verify round-trip
        let expected = PrimitiveArray::from_iter(vec![42u16; 2000]);
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn test_all_different() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Buffer<u8> = (0u8..=255).collect();

        let encoded = RLEData::encode(
            PrimitiveArray::new(values, Validity::NonNullable).as_view(),
            &mut ctx,
        )?;
        assert_eq!(encoded.values().len(), 256);

        let decoded = encoded
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?; // Verify round-trip
        let expected = PrimitiveArray::from_iter((0u8..=255).collect::<Vec<_>>());
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn test_partial_last_chunk() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Test array with partial last chunk (not divisible by 1024)
        let values: Buffer<u32> = (0..1500).map(|i| (i / 100) as u32).collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);

        let encoded = RLEData::encode(array.as_view(), &mut ctx).unwrap();

        assert_eq!(encoded.len(), 1500);
        assert_arrays_eq!(encoded, array);
        // 2 chunks: 1024 + 476 elements
        assert_eq!(encoded.values_idx_offsets().len(), 2);
    }

    #[test]
    fn test_two_full_chunks() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Array that spans exactly 2 chunks (2048 elements)
        let values: Buffer<u32> = (0..2048).map(|i| (i / 100) as u32).collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);

        let encoded = RLEData::encode(array.as_view(), &mut ctx).unwrap();

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
    fn test_roundtrip_primitive_types<T: NativePType>(
        #[case] values: Buffer<T>,
    ) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let primitive = values
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let result = RLEData::encode(primitive.as_view(), &mut ctx)?;
        let decoded = result
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected = PrimitiveArray::new(values, primitive.validity()?);
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    /// Replaces the indices of an RLE array with MaskedArray(ConstantArray(1u16), validity).
    ///
    /// Simulates a compressor that represents indices as a masked constant.
    /// Valid when every chunk has at least two RLE dictionary entries (the
    /// fill-forward default at index 0 and the actual value at index 1), which
    /// holds whenever the first position of each chunk is null.
    fn with_masked_constant_indices(
        rle: &RLEArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RLEArray> {
        let indices_prim = rle.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let masked_indices = MaskedArray::try_new(
            ConstantArray::new(1u16, indices_prim.len()).into_array(),
            indices_prim.validity()?,
        )?
        .into_array();
        RLE::try_new(
            rle.values().clone(),
            masked_indices,
            rle.values_idx_offsets().clone(),
            rle.offset(),
            rle.len(),
        )
    }

    #[test]
    fn test_encode_all_null_chunk() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<Option<u32>> = vec![None; FL_CHUNK_SIZE];
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let decoded = with_masked_constant_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(decoded, original);
        Ok(())
    }

    #[test]
    fn test_encode_all_null_chunk_then_value_chunk() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // First chunk is entirely null, second chunk has a value preceded by nulls.
        let mut values: Vec<Option<u32>> = vec![None; 2 * FL_CHUNK_SIZE];
        values[FL_CHUNK_SIZE + 100] = Some(42);
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let decoded = with_masked_constant_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(decoded, original);
        Ok(())
    }

    #[test]
    fn test_encode_one_value_near_end() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Single distinct value near the end of the chunk.
        let mut values: Vec<Option<u32>> = vec![None; FL_CHUNK_SIZE];
        values[1000] = Some(42);
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let decoded = with_masked_constant_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(decoded, original);
        Ok(())
    }

    #[test]
    fn test_encode_value_chunk_then_all_null_remainder() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // 1085 elements (2 chunks: 1024 + 61 padded to 1024).
        // Chunk 0 has -1i16 at scattered positions (273..=366), rest null.
        // Chunk 1 (the remainder) is entirely null.
        const NEG1_POSITIONS: &[usize] = &[
            273, 276, 277, 278, 279, 281, 282, 284, 285, 286, 287, 288, 289, 291, 292, 293, 296,
            298, 299, 302, 304, 308, 310, 311, 313, 314, 315, 317, 318, 322, 324, 325, 334, 335,
            336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 350, 352, 353, 355, 358,
            359, 362, 363, 364, 366,
        ];
        let mut values: Vec<Option<i16>> = vec![None; 1085];
        for &pos in NEG1_POSITIONS {
            values[pos] = Some(-1);
        }
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let decoded = with_masked_constant_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(decoded, original);
        Ok(())
    }

    /// Replaces indices at invalid (null) positions with random garbage values.
    ///
    /// This simulates a compressor that doesn't preserve index values at null
    /// positions, which can happen when indices are further compressed and the
    /// compressor clobbers invalid entries with arbitrary data.
    fn with_random_invalid_indices(
        rle: &RLEArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RLEArray> {
        let indices_prim = rle.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let mut indices_data: Vec<u16> = indices_prim.as_slice::<u16>().to_vec();

        // Use a simple deterministic "random" sequence.
        let mut rng_state: u32 = 0xDEAD_BEEF;
        let validity = indices_prim.validity()?;
        for (i, idx) in indices_data.iter_mut().enumerate() {
            if !validity.execute_is_valid(i, ctx).unwrap_or(true) {
                // xorshift32
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 17;
                rng_state ^= rng_state << 5;
                *idx = rng_state as u16;
            }
        }

        let clobbered_indices =
            PrimitiveArray::new(Buffer::from(indices_data), indices_prim.validity()?).into_array();

        RLE::try_new(
            rle.values().clone(),
            clobbered_indices,
            rle.values_idx_offsets().clone(),
            rle.offset(),
            rle.len(),
        )
    }

    #[test]
    fn test_random_invalid_indices_all_null_chunk() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<Option<u32>> = vec![None; FL_CHUNK_SIZE];
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let clobbered = with_random_invalid_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(clobbered, original);
        Ok(())
    }

    #[test]
    fn test_random_invalid_indices_sparse_values() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut values: Vec<Option<u32>> = vec![None; FL_CHUNK_SIZE];
        values[0] = Some(10);
        values[500] = Some(20);
        values[1000] = Some(30);
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let clobbered = with_random_invalid_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(clobbered, original);
        Ok(())
    }

    #[test]
    fn test_random_invalid_indices_multi_chunk() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Two chunks: first has scattered values, second is all null.
        let mut values: Vec<Option<i16>> = vec![None; 2 * FL_CHUNK_SIZE];
        values[0] = Some(10);
        values[500] = Some(20);
        values[FL_CHUNK_SIZE + 100] = Some(42);
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let clobbered = with_random_invalid_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(clobbered, original);
        Ok(())
    }

    #[test]
    fn test_random_invalid_indices_partial_last_chunk() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // 1085 elements: chunk 0 has values at scattered positions, chunk 1 is
        // a partial (61 elements padded to 1024) that is entirely null.
        let mut values: Vec<Option<u32>> = vec![None; 1085];
        for i in (100..200).step_by(7) {
            values[i] = Some(i as u32);
        }
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let clobbered = with_random_invalid_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(clobbered, original);
        Ok(())
    }

    #[test]
    fn test_random_invalid_indices_mostly_valid() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Most positions are valid, only a few are null with garbage indices.
        let mut values: Vec<Option<u64>> =
            (0..FL_CHUNK_SIZE).map(|i| Some((i / 100) as u64)).collect();
        // Sprinkle in some nulls.
        for i in (0..FL_CHUNK_SIZE).step_by(37) {
            values[i] = None;
        }
        let original = PrimitiveArray::from_option_iter(values);
        let rle = RLEData::encode(original.as_view(), &mut ctx)?;
        let clobbered = with_random_invalid_indices(&rle, &mut ctx)?;
        assert_arrays_eq!(clobbered, original);
        Ok(())
    }

    // Regression test: RLE compression properly supports decoding pos/neg zeros
    // See <https://github.com/vortex-data/vortex/issues/6491>
    #[rstest]
    #[case(vec![f16::ZERO, f16::NEG_ZERO])]
    #[case(vec![0f32, -0f32])]
    #[case(vec![0f64, -0f64])]
    fn test_float_zeros<T: NativePType + fastlanes::RLE>(
        #[case] values: Vec<T>,
    ) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter(values);
        let rle = RLEData::encode(primitive.as_view(), &mut ctx)?;
        let decoded = rle.as_array().clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(primitive, decoded);
        Ok(())
    }
}
