// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use ::alp::ENCODE_CHUNK_SIZE;
use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::dispatch;
use fearless_simd_macros::simd;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
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
use crate::sample_valid_values;
use crate::sanitize_nulls;

/// Matches the `alp` crate's exponent search sample budget and run length.
const ALP_SAMPLE_SIZE: usize = 64;
const ALP_SAMPLE_BLOCK: usize = 8;

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

/// Runs the `alp` encoder compiled for the best SIMD level the CPU supports.
fn encode_into<T: ALPFloat>(
    values: &[T],
    exponents: Option<Exponents>,
    encoded: &mut [MaybeUninit<T::ALPInt>],
    patch_indices: &mut Vec<u64>,
    patch_values: &mut Vec<T>,
    chunk_offsets: &mut Vec<u64>,
) -> Exponents {
    dispatch!(Level::new(), simd => encode_into_simd(
        simd,
        values,
        exponents,
        encoded,
        patch_indices,
        patch_values,
        chunk_offsets,
    ))
}

/// The `alp` encoder is generic, so it is instantiated in this crate; whatever of it LLVM inlines
/// here is compiled with the target features of `S`.
#[simd]
fn encode_into_simd<S: Simd, T: ALPFloat>(
    _: S,
    values: &[T],
    exponents: Option<Exponents>,
    encoded: &mut [MaybeUninit<T::ALPInt>],
    patch_indices: &mut Vec<u64>,
    patch_values: &mut Vec<T>,
    chunk_offsets: &mut Vec<u64>,
) -> Exponents {
    ::alp::encode_into(
        values,
        exponents,
        encoded,
        patch_indices,
        patch_values,
        chunk_offsets,
    )
}

/// Finds the best ALP exponents for a sample, compiled for the best SIMD level the CPU supports.
fn find_best_exponents<T: ALPFloat>(values: &[T]) -> Exponents {
    dispatch!(Level::new(), simd => find_best_exponents_simd(simd, values))
}

#[simd]
fn find_best_exponents_simd<S: Simd, T: ALPFloat>(_: S, values: &[T]) -> Exponents {
    ::alp::find_best_exponents(values)
}

fn alp_encode_components_typed<T>(
    values: ArrayView<'_, Primitive>,
    exponents: Option<Exponents>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Exponents, ArrayRef, Option<Patches>)>
where
    T: ALPFloat + NativePType,
    T::ALPInt: NativePType,
{
    let values_slice = values.as_slice::<T>();
    let len = values_slice.len();
    let validity = values.validity()?;
    let validity_mask = validity.execute_mask(len, ctx)?;

    // Encode straight into a Vortex buffer, so that the encoded array owns its values rather than
    // adopting a `Vec` allocated by the encoder.
    let mut encoded = BufferMut::<T::ALPInt>::with_capacity(len);

    // Estimate capacity to be one patch per 32 values.
    let mut exceptional_positions = Vec::with_capacity(len / 32);
    let mut exceptional_values = Vec::with_capacity(len / 32);

    // There's exactly one offset per chunk.
    let mut chunk_offsets = Vec::with_capacity(len.div_ceil(ENCODE_CHUNK_SIZE));

    let exponents = match validity_mask {
        Mask::AllFalse(_) => {
            let exponents = exponents.unwrap_or(Exponents { e: 0, f: 0 });
            let encoded_array =
                PrimitiveArray::new(Buffer::<T::ALPInt>::zeroed(len), validity).into_array();
            return Ok((exponents, encoded_array, None));
        }
        Mask::AllTrue(_) => encode_into::<T>(
            values_slice,
            exponents,
            &mut encoded.spare_capacity_mut()[..len],
            &mut exceptional_positions,
            &mut exceptional_values,
            &mut chunk_offsets,
        ),
        Mask::Values(mask_values) => {
            // Sample only valid values so dirty garbage bits in null slots cannot corrupt exponent
            // selection, then replace null slots with a value guaranteed to round-trip cleanly so
            // null indices never emit patches or widen the encoded integer range.
            let sample = sample_valid_values(
                values_slice,
                &mask_values,
                ALP_SAMPLE_SIZE,
                ALP_SAMPLE_BLOCK,
            );
            let exp = exponents.unwrap_or_else(|| find_best_exponents::<T>(&sample));
            let fill = sample
                .iter()
                .copied()
                .find(|&value| T::encode_single(value, exp).is_some())
                .unwrap_or_else(T::zero);
            let sanitized = sanitize_nulls(values_slice, &mask_values, fill);

            encode_into::<T>(
                &sanitized,
                Some(exp),
                &mut encoded.spare_capacity_mut()[..len],
                &mut exceptional_positions,
                &mut exceptional_values,
                &mut chunk_offsets,
            )
        }
    };
    // SAFETY: `encode_into` initializes exactly one element per value.
    unsafe { encoded.set_len(len) };

    let encoded_array = PrimitiveArray::new(encoded.freeze(), validity).into_array();

    let patches = if exceptional_positions.is_empty() {
        None
    } else {
        let patches_validity = if values.dtype().is_nullable() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };
        let patch_values =
            PrimitiveArray::new(Buffer::from(exceptional_values), patches_validity).into_array();

        Some(Patches::new(
            len,
            0,
            Buffer::from(exceptional_positions).into_array(),
            patch_values,
            Some(Buffer::from(chunk_offsets).into_array()),
            ctx,
        )?)
    };
    Ok((exponents, encoded_array, patches))
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E as E_F32;
    use std::f32::consts::PI as PI_F32;
    use std::f64::consts::E;
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::*;
    use crate::alp::array::ALPArrayExt;
    use crate::alp::array::ALPArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_compress() {
        let array = PrimitiveArray::new(buffer![1.234f32; 1025], Validity::NonNullable);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded = PrimitiveArray::from_iter(vec![1234i32; 1025]);
        assert_arrays_eq!(
            encoded.encoded(),
            expected_encoded,
            &mut SESSION.create_execution_ctx()
        );
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_nullable_compress() {
        let array = PrimitiveArray::from_option_iter([None, Some(1.234f32), None]);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded = PrimitiveArray::from_option_iter([None, Some(1234i32), None]);
        assert_arrays_eq!(
            encoded.encoded(),
            expected_encoded,
            &mut SESSION.create_execution_ctx()
        );
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        let expected = PrimitiveArray::from_option_iter(vec![None, Some(1.234f32), None]);
        assert_arrays_eq!(decoded, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    #[expect(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_patched_compress() {
        let values = buffer![1.234f64, 2.718, PI, 4.0];
        let array = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        assert!(encoded.patches().is_some());
        let expected_encoded = PrimitiveArray::from_iter(vec![1234i64, 2718, 1234, 4000]);
        assert_arrays_eq!(
            encoded.encoded(),
            expected_encoded,
            &mut SESSION.create_execution_ctx()
        );
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        let expected_decoded = PrimitiveArray::new(values, Validity::NonNullable);
        assert_arrays_eq!(
            decoded,
            expected_decoded,
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    #[expect(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_compress_ignores_invalid_exceptional_values() {
        let values = buffer![1.234f64, 2.718, PI, 4.0];
        let array = PrimitiveArray::new(values, Validity::from_iter([true, true, false, true]));
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded =
            PrimitiveArray::from_option_iter(buffer![Some(1234i64), Some(2718), None, Some(4000)]);
        assert_arrays_eq!(
            encoded.encoded(),
            expected_encoded,
            &mut SESSION.create_execution_ctx()
        );
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
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
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        assert!(encoded.patches().is_some());

        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        assert_arrays_eq!(encoded, array, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn roundtrips_close_fractional() {
        let original = PrimitiveArray::from_iter([195.26274f32, 195.27837, -48.815685]);
        let alp_arr = alp_encode(
            original.as_view(),
            None,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert_arrays_eq!(alp_arr, original, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn roundtrips_all_null() {
        let mut ctx = SESSION.create_execution_ctx();
        let original =
            PrimitiveArray::new(buffer![195.26274f64, PI, -48.815685], Validity::AllInvalid);
        let alp_arr = alp_encode(original.as_view(), None, &mut ctx).unwrap();

        assert_arrays_eq!(alp_arr, original, &mut ctx);
    }

    #[test]
    fn non_finite_numbers() {
        let mut ctx = SESSION.create_execution_ctx();
        let original = PrimitiveArray::new(
            buffer![0.0f32, -0.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
            Validity::NonNullable,
        );
        let encoded = alp_encode(original.as_view(), None, &mut ctx).unwrap();
        let decoded = encoded
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        for idx in 0..original.len() {
            let decoded_val = decoded.as_slice::<f32>()[idx];
            let original_val = original.as_slice::<f32>()[idx];
            assert!(
                NativePType::is_eq(decoded_val, original_val),
                "Expected {original_val} but got {decoded_val}"
            );
        }
    }

    #[test]
    fn test_chunk_offsets() {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = vec![1.0f64; 3072];

        values[1023] = PI;
        values[1024] = E;
        values[1025] = PI;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64, 1, 3]);
        assert_arrays_eq!(chunk_offsets, expected_offsets, &mut ctx);

        let patch_indices = patches.indices().clone();
        let expected_indices = PrimitiveArray::from_iter(vec![1023u64, 1024, 1025]);
        assert_arrays_eq!(patch_indices, expected_indices, &mut ctx);

        let patch_values = patches.values().clone();
        let expected_values = PrimitiveArray::from_iter(vec![PI, E, PI]);
        assert_arrays_eq!(patch_values, expected_values, &mut ctx);
    }

    #[test]
    fn test_chunk_offsets_no_patches_in_middle() {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = vec![1.0f64; 3072];
        values[0] = PI;
        values[2048] = E;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64, 1, 1]);
        assert_arrays_eq!(chunk_offsets, expected_offsets, &mut ctx);

        let patch_indices = patches.indices().clone();
        let expected_indices = PrimitiveArray::from_iter(vec![0u64, 2048]);
        assert_arrays_eq!(patch_indices, expected_indices, &mut ctx);

        let patch_values = patches.values().clone();
        let expected_values = PrimitiveArray::from_iter(vec![PI, E]);
        assert_arrays_eq!(patch_values, expected_values, &mut ctx);
    }

    #[test]
    fn test_chunk_offsets_trailing_empty_chunks() {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = vec![1.0f64; 3072];
        values[0] = PI;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64, 1, 1]);
        assert_arrays_eq!(chunk_offsets, expected_offsets, &mut ctx);

        let patch_indices = patches.indices().clone();
        let expected_indices = PrimitiveArray::from_iter(vec![0u64]);
        assert_arrays_eq!(patch_indices, expected_indices, &mut ctx);

        let patch_values = patches.values().clone();
        let expected_values = PrimitiveArray::from_iter(vec![PI]);
        assert_arrays_eq!(patch_values, expected_values, &mut ctx);
    }

    #[test]
    fn test_chunk_offsets_single_chunk() {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = vec![1.0f64; 512];
        values[0] = PI;
        values[100] = E;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        let patches = encoded.patches().unwrap();

        let chunk_offsets = patches.chunk_offsets().clone().unwrap();
        let expected_offsets = PrimitiveArray::from_iter(vec![0u64]);
        assert_arrays_eq!(chunk_offsets, expected_offsets, &mut ctx);

        let patch_indices = patches.indices().clone();
        let expected_indices = PrimitiveArray::from_iter(vec![0u64, 100]);
        assert_arrays_eq!(patch_indices, expected_indices, &mut ctx);

        let patch_values = patches.values().clone();
        let expected_values = PrimitiveArray::from_iter(vec![PI, E]);
        assert_arrays_eq!(patch_values, expected_values, &mut ctx);
    }

    #[test]
    fn test_slice_half_chunk_f32_roundtrip() {
        // Create 1024 elements, encode, slice to first 512, then decode
        let values = vec![1.234f32; 1024];
        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(
            sliced_alp,
            expected_slice,
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_slice_half_chunk_f64_roundtrip() {
        let values = vec![5.678f64; 1024];
        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded = alp_encode(
            original.as_view(),
            None,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(
            sliced_alp,
            expected_slice,
            &mut SESSION.create_execution_ctx()
        );
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
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(
            sliced_alp,
            expected_slice,
            &mut SESSION.create_execution_ctx()
        );
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
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();

        let sliced_alp = encoded.slice(1023..1025).unwrap();

        let expected_slice = original.slice(1023..1025).unwrap();
        assert_arrays_eq!(
            sliced_alp,
            expected_slice,
            &mut SESSION.create_execution_ctx()
        );
        assert!(encoded.patches().is_some());
    }

    #[test]
    fn test_slice_half_chunk_nullable_roundtrip() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..1024)
            .map(|i| if i % 3 == 0 { None } else { Some(2.5f32) })
            .collect::<Vec<_>>();

        let original = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(original.as_view(), None, &mut ctx).unwrap();

        let sliced_alp = encoded.slice(512..1024).unwrap();

        let expected_slice = original.slice(512..1024).unwrap();
        assert_arrays_eq!(sliced_alp, expected_slice, &mut ctx);
    }

    #[test]
    fn test_large_f32_array_uniform_values() {
        let size = 10_000;
        let array = PrimitiveArray::new(buffer![42.125f32; size], Validity::NonNullable);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();

        assert!(encoded.patches().is_none());
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_large_f64_array_uniform_values() {
        let size = 50_000;
        let array = PrimitiveArray::new(buffer![123.456789f64; size], Validity::NonNullable);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();

        assert!(encoded.patches().is_none());
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_large_f32_array_with_patches() {
        let size = 5_000;
        let mut values = vec![1.5f32; size];
        values[100] = PI_F32;
        values[1500] = E_F32;
        values[3000] = f32::NEG_INFINITY;
        values[4500] = f32::INFINITY;

        let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();

        assert!(encoded.patches().is_some());
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
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
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();

        assert!(encoded.patches().is_some());
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();

        for idx in 0..size {
            let decoded_val = decoded.as_slice::<f64>()[idx];
            let original_val = values[idx];
            assert!(
                NativePType::is_eq(decoded_val, original_val),
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
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();

        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
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
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();

        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
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
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().is_some());

        // Slice ending mid-chunk-2 (element 2500 is inside chunk 2 = 2048..3072).
        // This creates a mismatch: chunk_offsets includes the full chunk 2 offset,
        // but patches_indices only includes patches up to element 2500.
        let sliced_alp = encoded.slice(0..2500).unwrap();
        let expected = original.slice(0..2500).unwrap();
        assert_arrays_eq!(sliced_alp, expected, &mut SESSION.create_execution_ctx());

        // Also test slicing that starts mid-chunk (both start and end mid-chunk).
        let sliced_alp = encoded.slice(500..2500).unwrap();
        let expected = original.slice(500..2500).unwrap();
        assert_arrays_eq!(sliced_alp, expected, &mut SESSION.create_execution_ctx());
    }

    fn dirty_null_values_f32() -> [f32; 8] {
        [
            f32::NAN,
            f32::from_bits(0xdead_beef),
            1e30_f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
            -1e30_f32,
            1.234_567_f32,
        ]
    }

    fn dirty_null_values_f64() -> [f64; 8] {
        [
            f64::NAN,
            f64::from_bits(0xdead_beef_cafe_babe),
            1e300_f64,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(1),
            -1e300_f64,
            1.234_567_89_f64,
        ]
    }

    fn check_dirty_nulls_alp<T>(
        clean_values: &[T],
        dirty_values: &[T],
        valid_exception: T,
    ) -> VortexResult<()>
    where
        T: ALPFloat + NativePType,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 10_000;
        let mut raw = Vec::with_capacity(len);
        let mut valid_only = Vec::with_capacity(len);
        let mut is_valid = Vec::with_capacity(len);

        for row in 0..len {
            // Make leading rows and 10% of all rows dirty nulls so they overlap the exponent sample.
            if row < 16 || row % 10 == 0 {
                raw.push(dirty_values[row % dirty_values.len()]);
                is_valid.push(false);
            } else {
                let value = clean_values[row % clean_values.len()];
                raw.push(value);
                valid_only.push(value);
                is_valid.push(true);
            }
        }

        let array = PrimitiveArray::new(Buffer::from(raw), Validity::from_iter(is_valid));
        let clean_array = PrimitiveArray::new(Buffer::from(valid_only), Validity::NonNullable);

        let encoded = alp_encode(array.as_view(), None, &mut ctx)?;
        let clean_encoded = alp_encode(clean_array.as_view(), None, &mut ctx)?;

        assert_eq!(encoded.exponents(), clean_encoded.exponents());
        assert!(encoded.patches().is_none());
        assert_arrays_eq!(encoded.clone(), array, &mut ctx);

        // Verify take, filter, slice, and scalar_at remain sound on dummy null slots.
        let take_indices = PrimitiveArray::from_iter([0u32, 16, 20, 17, 9990, 9999]).into_array();
        assert_arrays_eq!(
            encoded.take(take_indices.clone())?,
            array.take(take_indices)?,
            &mut ctx
        );
        let filter_mask = Mask::from_iter((0..len).map(|row| row % 7 == 0));
        assert_arrays_eq!(
            encoded.filter(filter_mask.clone())?,
            array.filter(filter_mask)?,
            &mut ctx
        );
        assert_arrays_eq!(encoded.slice(10..50)?, array.slice(10..50)?, &mut ctx);
        assert!(encoded.execute_scalar(0, &mut ctx)?.is_null());
        assert_eq!(
            encoded.execute_scalar(16, &mut ctx)?,
            array.execute_scalar(16, &mut ctx)?
        );

        // Mix valid exceptions with dirty nulls: only valid exceptions should produce patches.
        let mut mixed_raw = Vec::with_capacity(2048);
        let mut mixed_valid = Vec::with_capacity(2048);
        let mut expected_patches = 0usize;
        for row in 0..2048 {
            if row % 5 == 0 {
                mixed_raw.push(dirty_values[row % dirty_values.len()]);
                mixed_valid.push(false);
            } else if row % 17 == 0 {
                mixed_raw.push(valid_exception);
                mixed_valid.push(true);
                expected_patches += 1;
            } else {
                mixed_raw.push(clean_values[row % clean_values.len()]);
                mixed_valid.push(true);
            }
        }
        let mixed_array =
            PrimitiveArray::new(Buffer::from(mixed_raw), Validity::from_iter(mixed_valid));
        let mixed_encoded = alp_encode(mixed_array.as_view(), None, &mut ctx)?;
        let patches = mixed_encoded
            .patches()
            .expect("valid exceptions should still produce patches");
        assert_eq!(patches.num_patches(), expected_patches);
        assert_arrays_eq!(mixed_encoded, mixed_array, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_dirty_nulls_alp_f32() -> VortexResult<()> {
        check_dirty_nulls_alp(
            &[1.25_f32, 2.50, 3.75, 4.00],
            &dirty_null_values_f32(),
            PI_F32,
        )
    }

    #[test]
    fn test_dirty_nulls_alp_f64() -> VortexResult<()> {
        check_dirty_nulls_alp(&[1.25_f64, 2.50, 3.75, 4.00], &dirty_null_values_f64(), PI)
    }

    #[rstest]
    #[case::f32(1.25_f32, f32::INFINITY, dirty_null_values_f32().to_vec())]
    #[case::f64(1.25_f64, f64::INFINITY, dirty_null_values_f64().to_vec())]
    fn test_dirty_nulls_alp_edge_cases<T>(
        #[case] valid_value: T,
        #[case] valid_exception: T,
        #[case] dirty_values: Vec<T>,
    ) -> VortexResult<()>
    where
        T: ALPFloat + NativePType,
    {
        let mut ctx = SESSION.create_execution_ctx();

        // All-null array filled with dirty garbage bits.
        let all_null_raw: Vec<T> = (0..1024)
            .map(|row| dirty_values[row % dirty_values.len()])
            .collect();
        let all_null = PrimitiveArray::new(Buffer::from(all_null_raw), Validity::AllInvalid);
        let all_null_encoded = alp_encode(all_null.as_view(), None, &mut ctx)?;
        assert!(all_null_encoded.patches().is_none());
        assert_arrays_eq!(all_null_encoded.clone(), all_null, &mut ctx);
        assert!(all_null_encoded.execute_scalar(0, &mut ctx)?.is_null());

        // Single valid value surrounded by dirty nulls.
        let mut one_valid_raw: Vec<T> = (0..1024)
            .map(|row| dirty_values[row % dirty_values.len()])
            .collect();
        let mut one_valid_mask = vec![false; 1024];
        one_valid_raw[777] = valid_value;
        one_valid_mask[777] = true;
        let one_valid = PrimitiveArray::new(
            Buffer::from(one_valid_raw),
            Validity::from_iter(one_valid_mask),
        );
        let one_valid_encoded = alp_encode(one_valid.as_view(), None, &mut ctx)?;
        let single_clean =
            PrimitiveArray::new(Buffer::from(vec![valid_value]), Validity::NonNullable);
        let single_clean_encoded = alp_encode(single_clean.as_view(), None, &mut ctx)?;
        assert_eq!(
            one_valid_encoded.exponents(),
            single_clean_encoded.exponents()
        );
        assert!(one_valid_encoded.patches().is_none());
        assert_arrays_eq!(one_valid_encoded, one_valid, &mut ctx);

        // Every valid value is an exception, exercising the zero-fill fallback for dirty nulls.
        let mut all_exc_raw = Vec::with_capacity(256);
        let mut all_exc_valid = Vec::with_capacity(256);
        let mut valid_count = 0usize;
        for row in 0..256 {
            if row % 4 == 0 {
                all_exc_raw.push(valid_exception);
                all_exc_valid.push(true);
                valid_count += 1;
            } else {
                all_exc_raw.push(dirty_values[row % dirty_values.len()]);
                all_exc_valid.push(false);
            }
        }
        let all_exc = PrimitiveArray::new(
            Buffer::from(all_exc_raw),
            Validity::from_iter(all_exc_valid),
        );
        let all_exc_encoded = alp_encode(all_exc.as_view(), None, &mut ctx)?;
        assert_eq!(
            all_exc_encoded
                .patches()
                .expect("valid exceptions must still be patched")
                .num_patches(),
            valid_count
        );
        assert_arrays_eq!(all_exc_encoded, all_exc, &mut ctx);

        Ok(())
    }
}
