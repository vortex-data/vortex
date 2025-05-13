use itertools::Itertools as _;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{NativePType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::ScalarType;

use crate::Exponents;
use crate::alp::{ALPArray, ALPFloat};

#[macro_export]
macro_rules! match_each_alp_float_ptype {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use vortex_dtype::PType;
        use vortex_error::vortex_panic;
        let ptype = $self;
        match ptype {
            PType::F32 => __with__! { f32 },
            PType::F64 => __with__! { f64 },
            _ => vortex_panic!("ALP can only encode f32 and f64, got {}", ptype),
        }
    })
}

pub fn alp_encode(parray: &PrimitiveArray, exponents: Option<Exponents>) -> VortexResult<ALPArray> {
    let (exponents, encoded, patches) = match parray.ptype() {
        PType::F32 => alp_encode_components_typed::<f32>(parray, exponents)?,
        PType::F64 => alp_encode_components_typed::<f64>(parray, exponents)?,
        _ => vortex_bail!("ALP can only encode f32 and f64"),
    };

    ALPArray::try_new(encoded, exponents, patches)
}

#[allow(clippy::cast_possible_truncation)]
fn alp_encode_components_typed<T>(
    values: &PrimitiveArray,
    exponents: Option<Exponents>,
) -> VortexResult<(Exponents, ArrayRef, Option<Patches>)>
where
    T: ALPFloat + NativePType,
    T::ALPInt: NativePType,
    T: ScalarType,
{
    let values_slice = values.as_slice::<T>();

    let (exponents, encoded, exceptional_positions, exceptional_values) =
        T::encode(values_slice, exponents);

    let encoded_array = PrimitiveArray::new(encoded, values.validity().clone()).into_array();

    let validity = values.validity_mask()?;
    // exceptional_positions may contain exceptions at invalid positions (which contain garbage
    // data). We remove invalid exceptional positions in order to keep the Patches small.
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
                    .filter(|(index, _)| is_valid.value(*index as usize))
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
        ))
    };
    Ok((exponents, encoded_array, patches))
}

pub fn decompress(array: &ALPArray) -> VortexResult<PrimitiveArray> {
    let encoded = array.encoded().to_primitive()?;
    let validity = encoded.validity().clone();
    let ptype = array.dtype().try_into()?;

    let decoded = match_each_alp_float_ptype!(ptype, |$T| {
        PrimitiveArray::new::<$T>(
            <$T>::decode_buffer(encoded.into_buffer_mut(), array.exponents()),
            validity,
        )
    });

    if let Some(patches) = array.patches() {
        decoded.patch(patches)
    } else {
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use vortex_array::validity::Validity;
    use vortex_buffer::{Buffer, buffer};
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn test_compress() {
        let array = PrimitiveArray::new(buffer![1.234f32; 1025], Validity::NonNullable);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().unwrap().as_slice::<i32>(),
            vec![1234; 1025]
        );
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = decompress(&encoded).unwrap();
        assert_eq!(array.as_slice::<f32>(), decoded.as_slice::<f32>());
    }

    #[test]
    fn test_nullable_compress() {
        let array = PrimitiveArray::from_option_iter([None, Some(1.234f32), None]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().unwrap().as_slice::<i32>(),
            vec![0, 1234, 0]
        );
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = decompress(&encoded).unwrap();
        let expected = vec![0f32, 1.234f32, 0f32];
        assert_eq!(decoded.as_slice::<f32>(), expected.as_slice());
    }

    #[test]
    #[allow(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_patched_compress() {
        let values = buffer![1.234f64, 2.718, f64::consts::PI, 4.0];
        let array = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_some());
        assert_eq!(
            encoded.encoded().to_primitive().unwrap().as_slice::<i64>(),
            vec![1234i64, 2718, 1234, 4000]
        );
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded = decompress(&encoded).unwrap();
        assert_eq!(values.as_slice(), decoded.as_slice::<f64>());
    }

    #[test]
    #[allow(clippy::approx_constant)] // Clippy objects to 2.718, an approximation of e, the base of the natural logarithm.
    fn test_compress_ignores_invalid_exceptional_values() {
        let values = buffer![1.234f64, 2.718, f64::consts::PI, 4.0];
        let array = PrimitiveArray::new(values, Validity::from_iter([true, true, false, true]));
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().unwrap().as_slice::<i64>(),
            vec![1234i64, 2718, 1234, 4000]
        );
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded = decompress(&encoded).unwrap();
        assert_eq!(decoded.scalar_at(0).unwrap(), array.scalar_at(0).unwrap());
        assert_eq!(decoded.scalar_at(1).unwrap(), array.scalar_at(1).unwrap());
        assert!(!decoded.is_valid(2).unwrap());
        assert_eq!(decoded.scalar_at(3).unwrap(), array.scalar_at(3).unwrap());
    }

    #[test]
    #[allow(clippy::approx_constant)] // ALP doesn't like E
    fn test_nullable_patched_scalar_at() {
        let array = PrimitiveArray::from_option_iter([
            Some(1.234f64),
            Some(2.718),
            Some(f64::consts::PI),
            Some(4.0),
            None,
        ]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_some());

        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        for idx in 0..3 {
            let s = encoded.scalar_at(idx).unwrap();
            assert!(s.is_valid());
        }

        assert!(!encoded.is_valid(4).unwrap());
        let s = encoded.scalar_at(4).unwrap();
        assert!(s.is_null());

        let _decoded = decompress(&encoded).unwrap();
    }

    #[test]
    fn roundtrips_close_fractional() {
        let original = PrimitiveArray::from_iter([195.26274f32, 195.27837, -48.815685]);
        let alp_arr = alp_encode(&original, None).unwrap();
        let decompressed = alp_arr.to_primitive().unwrap();
        assert_eq!(original.as_slice::<f32>(), decompressed.as_slice::<f32>());
    }

    #[test]
    fn roundtrips_all_null() {
        let original = PrimitiveArray::new(
            Buffer::from_iter([195.26274f64, f64::consts::PI, -48.815685]),
            Validity::AllInvalid,
        );
        let alp_arr = alp_encode(&original, None).unwrap();
        let decompressed = alp_arr.to_primitive().unwrap();
        assert_eq!(
            // The second and third values become exceptions and are replaced
            [195.26274, 195.26274, 195.26274],
            decompressed.as_slice::<f64>()
        );
        assert_eq!(original.validity(), decompressed.validity());
        assert_eq!(original.scalar_at(0).unwrap(), Scalar::null_typed::<f64>());
        assert_eq!(original.scalar_at(1).unwrap(), Scalar::null_typed::<f64>());
        assert_eq!(original.scalar_at(2).unwrap(), Scalar::null_typed::<f64>());
    }

    #[test]
    fn non_finite_numbers() {
        let original = PrimitiveArray::new(
            buffer![0.0f32, -0.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
            Validity::NonNullable,
        );
        let encoded = alp_encode(&original, None).unwrap();
        let decoded = encoded.to_primitive().unwrap();
        for idx in 0..original.len() {
            let decoded_val = decoded.as_slice::<f32>()[idx];
            let original_val = original.as_slice::<f32>()[idx];
            assert!(
                decoded_val.is_eq(original_val),
                "Expected {original_val} but got {decoded_val}"
            );
        }
    }
}
