// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ALP;
use crate::Exponents;
use crate::alp::ALPArray;
use crate::alp::ALPFloat;

#[macro_export]
macro_rules! match_each_alp_float_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use vortex_array::dtype::PType;
        use vortex_error::vortex_panic;
        let ptype = $self;
        match ptype {
            PType::F32 => {
                type $enc = f32;
                $body
            }
            PType::F64 => {
                type $enc = f64;
                $body
            }
            _ => vortex_panic!("ALP can only encode f32 and f64, got {}", ptype),
        }
    }};
}

pub fn alp_encode(
    parray: ArrayView<'_, Primitive>,
    exponents: Option<Exponents>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ALPArray> {
    let (exponents, encoded, patches) = match parray.ptype() {
        PType::F32 => alp_encode_components_typed::<f32>(parray, exponents, ctx)?,
        PType::F64 => alp_encode_components_typed::<f64>(parray, exponents, ctx)?,
        _ => vortex_bail!("ALP can only encode f32 and f64"),
    };

    // SAFETY: alp_encode_components_typed must return well-formed components
    unsafe { Ok(ALP::new_unchecked(encoded, exponents, patches)) }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "u64 index cast to usize is safe for reasonable array sizes"
)]
fn alp_encode_components_typed<T>(
    values: ArrayView<'_, Primitive>,
    exponents: Option<Exponents>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Exponents, ArrayRef, Option<Patches>)>
where
    T: ALPFloat,
{
    let values_slice = values.as_slice::<T>();

    let (exponents, encoded, exceptional_positions, exceptional_values, mut chunk_offsets) =
        T::encode(values_slice, exponents);

    let encoded_array = PrimitiveArray::new(encoded, values.validity()?).into_array();

    let validity = values
        .array()
        .validity()?
        .to_mask(values.array().len(), ctx)?;
    // exceptional_positions may contain exceptions at invalid positions (which contain garbage
    // data). We remove null exceptions in order to keep the Patches small.
    let (valid_exceptional_positions, valid_exceptional_values): (Buffer<u64>, Buffer<T>) =
        match validity {
            Mask::AllTrue(_) => (exceptional_positions, exceptional_values),
            Mask::AllFalse(_) => {
                // no valid positions, ergo nothing worth patching
                (Buffer::empty(), Buffer::empty())
            }
            Mask::Values(is_valid) => {
                let (pos, vals): (BufferMut<u64>, BufferMut<T>) = exceptional_positions
                    .into_iter()
                    .zip_eq(exceptional_values)
                    .filter(|(index, _)| {
                        let is_valid = is_valid.value(*index as usize);
                        if !is_valid {
                            let patch_chunk = *index as usize / 1024;
                            for chunk_idx in (patch_chunk + 1)..chunk_offsets.len() {
                                chunk_offsets[chunk_idx] -= 1;
                            }
                        }
                        is_valid
                    })
                    .unzip();
                (pos.freeze(), vals.freeze())
            }
        };
    let patches = if valid_exceptional_positions.is_empty() {
        None
    } else {
        let patches_validity = if values.dtype().is_nullable() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };
        let valid_exceptional_values =
            PrimitiveArray::new(valid_exceptional_values, patches_validity).into_array();

        Some(Patches::new(
            values_slice.len(),
            0,
            valid_exceptional_positions.into_array(),
            valid_exceptional_values,
            Some(chunk_offsets.into_array()),
        )?)
    };
    Ok((exponents, encoded_array, patches))
}

#[cfg(test)]
mod tests {
    use core::f64;

    use f64::consts::E;
    use f64::consts::PI;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::alp::array::ALPArrayExt;
    use crate::alp::array::ALPArraySlotsExt;
    use crate::decompress_into_array;

    #[test]
    fn test_compress() {
        let array = PrimitiveArray::new(buffer![1.234f32; 1025], Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded = PrimitiveArray::from_iter(vec![1234i32; 1025]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
        assert_arrays_eq!(decoded, array);
    }

    #[test]
    fn test_nullable_compress() {
        let array = PrimitiveArray::from_option_iter([None, Some(1.234f32), None]);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded = PrimitiveArray::from_option_iter([None, Some(1234i32), None]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
        let expected = PrimitiveArray::from_option_iter(vec![None, Some(1.234f32), None]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    #[expect(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_patched_compress() {
        let values = buffer![1.234f64, 2.718, PI, 4.0];
        let array = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_some());
        let expected_encoded = PrimitiveArray::from_iter(vec![1234i64, 2718, 1234, 4000]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
        let expected_decoded = PrimitiveArray::new(values, Validity::NonNullable);
        assert_arrays_eq!(decoded, expected_decoded);
    }

    #[test]
    #[expect(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_compress_ignores_invalid_exceptional_values() {
        let values = buffer![1.234f64, 2.718, PI, 4.0];
        let array = PrimitiveArray::new(values, Validity::from_iter([true, true, false, true]));
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded =
            PrimitiveArray::from_option_iter(buffer![Some(1234i64), Some(2718), None, Some(4000)]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
        assert_arrays_eq!(decoded, array);
    }

    #[test]
    #[expect(clippy::approx_constant)] // ALP doesn't like E
    fn test_nullable_patched_scalar_at() {
        let array = PrimitiveArray::from_option_iter([
            Some(1.234f64),
            Some(2.718),
            Some(PI),
            Some(4.0),
            None,
        ]);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_some());

        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        assert_arrays_eq!(encoded, array);

        let _decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
    }

    #[test]
    fn roundtrips_close_fractional() {
        let original = PrimitiveArray::from_iter([195.26274f32, 195.27837, -48.815685]);
        let alp_arr = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert_arrays_eq!(alp_arr, original);
    }

    #[test]
    fn roundtrips_all_null() {
        let original =
            PrimitiveArray::new(buffer![195.26274f64, PI, -48.815685], Validity::AllInvalid);
        let alp_arr = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let decompressed = alp_arr.into_array().to_primitive();

        assert_eq!(
            // The second and third values become exceptions and are replaced
            [195.26274, 195.26274, 195.26274],
            decompressed.as_slice::<f64>()
        );

        assert_arrays_eq!(decompressed, original);
    }

    #[test]
    fn non_finite_numbers() {
        let original = PrimitiveArray::new(
            buffer![0.0f32, -0.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
            Validity::NonNullable,
        );
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let decoded = encoded.as_array().to_primitive();
        for idx in 0..original.len() {
            let decoded_val = decoded.as_slice::<f32>()[idx];
            let original_val = original.as_slice::<f32>()[idx];
            assert!(
                decoded_val.is_eq(original_val),
                "Expected {original_val} but got {decoded_val}"
            );
        }
    }

    #[test]
    fn test_chunk_offsets() {
        let mut values = vec![1.0f64; 3072];

        values[1023] = PI;
        values[1024] = E;
        values[1025] = PI;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap().to_primitive();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64, 1, 3]);
        assert_arrays_eq!(chunk_offsets, expected_offsets);

        let patch_indices = patches.indices().to_primitive();
        let expected_indices = PrimitiveArray::from_iter(vec![1023u64, 1024, 1025]);
        assert_arrays_eq!(patch_indices, expected_indices);

        let patch_values = patches.values().to_primitive();
        let expected_values = PrimitiveArray::from_iter(vec![PI, E, PI]);
        assert_arrays_eq!(patch_values, expected_values);
    }

    #[test]
    fn test_chunk_offsets_no_patches_in_middle() {
        let mut values = vec![1.0f64; 3072];
        values[0] = PI;
        values[2048] = E;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap().to_primitive();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64, 1, 1]);
        assert_arrays_eq!(chunk_offsets, expected_offsets);

        let patch_indices = patches.indices().to_primitive();
        let expected_indices = PrimitiveArray::from_iter(vec![0u64, 2048]);
        assert_arrays_eq!(patch_indices, expected_indices);

        let patch_values = patches.values().to_primitive();
        let expected_values = PrimitiveArray::from_iter(vec![PI, E]);
        assert_arrays_eq!(patch_values, expected_values);
    }

    #[test]
    fn test_chunk_offsets_trailing_empty_chunks() {
        let mut values = vec![1.0f64; 3072];
        values[0] = PI;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap().to_primitive();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64, 1, 1]);
        assert_arrays_eq!(chunk_offsets, expected_offsets);

        let patch_indices = patches.indices().to_primitive();
        let expected_indices = PrimitiveArray::from_iter(vec![0u64]);
        assert_arrays_eq!(patch_indices, expected_indices);

        let patch_values = patches.values().to_primitive();
        let expected_values = PrimitiveArray::from_iter(vec![PI]);
        assert_arrays_eq!(patch_values, expected_values);
    }

    #[test]
    fn test_chunk_offsets_single_chunk() {
        let mut values = vec![1.0f64; 512];
        values[0] = PI;
        values[100] = E;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap().to_primitive();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64]);
        assert_arrays_eq!(chunk_offsets, expected_offsets);

        let patch_indices = patches.indices().to_primitive();
        let expected_indices = PrimitiveArray::from_iter(vec![0u64, 100]);
        assert_arrays_eq!(patch_indices, expected_indices);

        let patch_values = patches.values().to_primitive();
        let expected_values = PrimitiveArray::from_iter(vec![PI, E]);
        assert_arrays_eq!(patch_values, expected_values);
    }

    #[test]
    fn test_slice_half_chunk_f32_roundtrip() {
        // Create 1024 elements, encode, slice to first 512, then decode
        let values = vec![1.234f32; 1024];
        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(sliced_alp, expected_slice);
    }

    #[test]
    fn test_slice_half_chunk_f64_roundtrip() {
        let values = vec![5.678f64; 1024];
        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(sliced_alp, expected_slice);
    }

    #[test]
    fn test_slice_half_chunk_with_patches_roundtrip() {
        let mut values = vec![1.0f64; 1024];
        values[100] = PI;
        values[200] = E;
        values[600] = 42.42;

        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(sliced_alp, expected_slice);
        assert!(encoded.patches().is_some());
    }

    #[test]
    fn test_slice_across_chunks_with_patches_roundtrip() {
        let mut values = vec![1.0f64; 2048];
        values[100] = PI;
        values[200] = E;
        values[600] = 42.42;
        values[800] = 42.42;
        values[1000] = 42.42;
        values[1023] = 42.42;

        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(1023..1025).unwrap();

        let expected_slice = original.slice(1023..1025).unwrap();
        assert_arrays_eq!(sliced_alp, expected_slice);
        assert!(encoded.patches().is_some());
    }

    #[test]
    fn test_slice_half_chunk_nullable_roundtrip() {
        let values = (0..1024)
            .map(|i| if i % 3 == 0 { None } else { Some(2.5f32) })
            .collect::<Vec<_>>();

        let original = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();
        let decoded = sliced_alp.to_primitive();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(decoded, expected_slice);
    }

    #[test]
    fn test_large_f32_array_uniform_values() {
        let size = 10_000;
        let array = PrimitiveArray::new(buffer![42.125f32; size], Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert!(encoded.patches().is_none());
        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
        assert_arrays_eq!(decoded, array);
    }

    #[test]
    fn test_large_f64_array_uniform_values() {
        let size = 50_000;
        let array = PrimitiveArray::new(buffer![123.456789f64; size], Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert!(encoded.patches().is_none());
        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
        assert_arrays_eq!(decoded, array);
    }

    #[test]
    fn test_large_f32_array_with_patches() {
        let size = 5_000;
        let mut values = vec![1.5f32; size];
        values[100] = std::f32::consts::PI;
        values[1500] = std::f32::consts::E;
        values[3000] = f32::NEG_INFINITY;
        values[4500] = f32::INFINITY;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert!(encoded.patches().is_some());
        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();
        assert_arrays_eq!(decoded, array);
    }

    #[test]
    fn test_large_f64_array_with_patches() {
        let size = 8_000;
        let mut values = vec![2.2184f64; size];
        values[0] = PI;
        values[1000] = E;
        values[2000] = f64::NAN;
        values[3000] = f64::INFINITY;
        values[4000] = f64::NEG_INFINITY;
        values[5000] = 0.0;
        values[6000] = -0.0;
        values[7000] = 999.999999999;

        let array = PrimitiveArray::new(Buffer::from(values.clone()), Validity::NonNullable);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        assert!(encoded.patches().is_some());
        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        for idx in 0..size {
            let decoded_val = decoded.as_slice::<f64>()[idx];
            let original_val = values[idx];
            assert!(
                decoded_val.is_eq(original_val),
                "At index {idx}: Expected {original_val} but got {decoded_val}"
            );
        }
    }

    #[test]
    fn test_large_nullable_array() {
        let size = 12_000;
        let values: Vec<Option<f32>> = (0..size)
            .map(|i| {
                if i % 7 == 0 {
                    None
                } else {
                    Some((i as f32) * 0.1)
                }
            })
            .collect();

        let array = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        assert_arrays_eq!(decoded, array);
    }

    #[test]
    fn test_large_mixed_validity_with_patches() {
        let size = 6_000;
        let mut values = vec![10.125f64; size];

        values[500] = PI;
        values[1500] = E;
        values[2500] = f64::INFINITY;
        values[3500] = f64::NEG_INFINITY;
        values[4500] = f64::NAN;

        let validity = Validity::from_iter((0..size).map(|i| !matches!(i, 500 | 2500)));

        let array = PrimitiveArray::new(Buffer::from(values), validity);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let decoded =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        assert_arrays_eq!(decoded, array);
    }

    /// Regression test for patch_chunk index-out-of-bounds when slicing a multi-chunk
    /// ALP array mid-chunk with patches in the trailing chunk.
    ///
    /// The bug: chunk_offsets are sliced at chunk granularity (1024-row boundaries)
    /// but patches indices/values are sliced at element granularity. When a slice ends
    /// mid-chunk, patches_end_idx could exceed patches_indices.len(), causing OOB panic
    /// during decompression.
    #[test]
    fn test_slice_mid_chunk_with_patches_in_trailing_chunk() {
        // 3 chunks (3072 elements), patches scattered across all chunks.
        let mut values = vec![1.0f64; 3072];
        // Chunk 0 patches (indices 0..1024)
        values[100] = PI;
        values[500] = E;
        // Chunk 1 patches (indices 1024..2048)
        values[1100] = PI;
        values[1500] = E;
        values[1900] = PI;
        // Chunk 2 patches (indices 2048..3072)
        values[2100] = PI;
        values[2500] = E;
        values[2900] = PI;

        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_some());

        // Slice ending mid-chunk-2 (element 2500 is inside chunk 2 = 2048..3072).
        // This creates a mismatch: chunk_offsets includes the full chunk 2 offset,
        // but patches_indices only includes patches up to element 2500.
        let sliced_alp = encoded.slice(0..2500).unwrap();
        let expected = original.slice(0..2500).unwrap();
        assert_arrays_eq!(sliced_alp, expected);

        // Also test slicing that starts mid-chunk (both start and end mid-chunk).
        let sliced_alp = encoded.slice(500..2500).unwrap();
        let expected = original.slice(500..2500).unwrap();
        assert_arrays_eq!(sliced_alp, expected);
    }
}
