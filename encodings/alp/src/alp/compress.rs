// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::PType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::Exponents;
use crate::alp::{ALPArray, ALPFloat};

#[macro_export]
macro_rules! match_each_alp_float_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use vortex_dtype::PType;
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

pub fn alp_encode(parray: &PrimitiveArray, exponents: Option<Exponents>) -> VortexResult<ALPArray> {
    let (exponents, encoded, patches) = match parray.ptype() {
        PType::F32 => alp_encode_components_typed::<f32>(parray, exponents)?,
        PType::F64 => alp_encode_components_typed::<f64>(parray, exponents)?,
        _ => vortex_bail!("ALP can only encode f32 and f64"),
    };

    // SAFETY: alp_encode_components_typed must return well-formed components
    unsafe {
        Ok(ALPArray::new_unchecked(
            encoded,
            exponents,
            patches,
            parray.dtype().clone(),
        ))
    }
}

#[allow(clippy::cast_possible_truncation)]
fn alp_encode_components_typed<T>(
    values: &PrimitiveArray,
    exponents: Option<Exponents>,
) -> VortexResult<(Exponents, ArrayRef, Option<Patches>)>
where
    T: ALPFloat,
{
    let values_slice = values.as_slice::<T>();

    let (exponents, encoded, exceptional_positions, exceptional_values, mut chunk_offsets) =
        T::encode(values_slice, exponents);

    let encoded_array = PrimitiveArray::new(encoded, values.validity().clone()).into_array();

    let validity = values.validity_mask();
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
        ))
    };
    Ok((exponents, encoded_array, patches))
}

pub fn decompress(array: &ALPArray) -> PrimitiveArray {
    let encoded = array.encoded().to_primitive();
    let validity = encoded.validity().clone();
    let ptype = array.dtype().as_ptype();

    let decoded = match_each_alp_float_ptype!(ptype, |T| {
        PrimitiveArray::new::<T>(
            <T>::decode_buffer(encoded.into_buffer_mut(), array.exponents()),
            validity,
        )
    });

    if let Some(patches) = array.patches() {
        decoded.patch(patches)
    } else {
        decoded
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use f64::consts::{E, PI};
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::NativePType;

    use super::*;

    #[test]
    fn test_compress() {
        let array = PrimitiveArray::new(buffer![1.234f32; 1025], Validity::NonNullable);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded = PrimitiveArray::from_iter(vec![1234i32; 1025]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = decompress(&encoded);
        assert_arrays_eq!(decoded, array);
    }

    #[test]
    fn test_nullable_compress() {
        let array = PrimitiveArray::from_option_iter([None, Some(1.234f32), None]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded = PrimitiveArray::from_option_iter([None, Some(1234i32), None]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = decompress(&encoded);
        let expected = PrimitiveArray::from_option_iter(vec![None, Some(1.234f32), None]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    #[allow(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_patched_compress() {
        let values = buffer![1.234f64, 2.718, PI, 4.0];
        let array = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_some());
        let expected_encoded = PrimitiveArray::from_iter(vec![1234i64, 2718, 1234, 4000]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded = decompress(&encoded);
        let expected_decoded = PrimitiveArray::new(values, Validity::NonNullable);
        assert_arrays_eq!(decoded, expected_decoded);
    }

    #[test]
    #[allow(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_compress_ignores_invalid_exceptional_values() {
        let values = buffer![1.234f64, 2.718, PI, 4.0];
        let array = PrimitiveArray::new(values, Validity::from_iter([true, true, false, true]));
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        let expected_encoded =
            PrimitiveArray::from_option_iter(buffer![Some(1234i64), Some(2718), None, Some(4000)]);
        assert_arrays_eq!(encoded.encoded(), expected_encoded);
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded = decompress(&encoded);
        assert_arrays_eq!(decoded, array);
    }

    #[test]
    #[allow(clippy::approx_constant)] // ALP doesn't like E
    fn test_nullable_patched_scalar_at() {
        let array = PrimitiveArray::from_option_iter([
            Some(1.234f64),
            Some(2.718),
            Some(PI),
            Some(4.0),
            None,
        ]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_some());

        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        assert_arrays_eq!(&encoded, array);

        let _decoded = decompress(&encoded);
    }

    #[test]
    fn roundtrips_close_fractional() {
        let original = PrimitiveArray::from_iter([195.26274f32, 195.27837, -48.815685]);
        let alp_arr = alp_encode(&original, None).unwrap();
        let decompressed = alp_arr.to_primitive();
        assert_arrays_eq!(decompressed, original);
    }

    #[test]
    fn roundtrips_all_null() {
        let original =
            PrimitiveArray::new(buffer![195.26274f64, PI, -48.815685], Validity::AllInvalid);
        let alp_arr = alp_encode(&original, None).unwrap();
        let decompressed = alp_arr.to_primitive();

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
        let encoded = alp_encode(&original, None).unwrap();
        let decoded = encoded.to_primitive();
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
        let encoded = alp_encode(&array, None).unwrap();
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
        let encoded = alp_encode(&array, None).unwrap();
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
        let encoded = alp_encode(&array, None).unwrap();
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
        let encoded = alp_encode(&array, None).unwrap();
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
}
