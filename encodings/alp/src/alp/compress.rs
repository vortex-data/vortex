use vortex_array::array::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{NativePType, PType};
use vortex_error::{vortex_bail, VortexResult, VortexUnwrap};
use vortex_scalar::ScalarType;

use crate::alp::{ALPArray, ALPFloat};
use crate::Exponents;

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

pub fn alp_encode_components<T>(
    values: &PrimitiveArray,
    exponents: Option<Exponents>,
) -> (Exponents, ArrayData, Option<Patches>)
where
    T: ALPFloat + NativePType,
    T::ALPInt: NativePType,
    T: ScalarType,
{
    let (exponents, encoded, exc_pos, exc) = T::encode(values.as_slice::<T>(), exponents);
    let len = encoded.len();
    (
        exponents,
        PrimitiveArray::new(encoded, values.validity()).into_array(),
        (!exc.is_empty()).then(|| {
            let position_arr = exc_pos.into_array();
            let patch_validity = values.validity().take(&position_arr).vortex_unwrap();
            Patches::new(
                len,
                position_arr,
                PrimitiveArray::new(exc, patch_validity).into_array(),
            )
        }),
    )
}

pub fn alp_encode(parray: &PrimitiveArray) -> VortexResult<ALPArray> {
    let (exponents, encoded, patches) = match parray.ptype() {
        PType::F32 => alp_encode_components::<f32>(parray, None),
        PType::F64 => alp_encode_components::<f64>(parray, None),
        _ => vortex_bail!("ALP can only encode f32 and f64"),
    };
    ALPArray::try_new(encoded, exponents, patches)
}

pub fn decompress(array: ALPArray) -> VortexResult<PrimitiveArray> {
    let encoded = array.encoded().into_primitive()?;
    let validity = encoded.validity();
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

    use vortex_array::compute::scalar_at;
    use vortex_array::validity::Validity;
    use vortex_buffer::{buffer, Buffer};

    use super::*;

    #[test]
    fn test_compress() {
        let array = PrimitiveArray::new(buffer![1.234f32; 1025], Validity::NonNullable);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded
                .encoded()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            vec![1234; 1025]
        );
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = decompress(encoded).unwrap();
        assert_eq!(array.as_slice::<f32>(), decoded.as_slice::<f32>());
    }

    #[test]
    fn test_nullable_compress() {
        let array = PrimitiveArray::from_option_iter([None, Some(1.234f32), None]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded
                .encoded()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            vec![0, 1234, 0]
        );
        assert_eq!(encoded.exponents(), Exponents { e: 9, f: 6 });

        let decoded = decompress(encoded).unwrap();
        let expected = vec![0f32, 1.234f32, 0f32];
        assert_eq!(decoded.as_slice::<f32>(), expected.as_slice());
    }

    #[test]
    #[allow(clippy::approx_constant)] // ALP doesn't like E
    fn test_patched_compress() {
        let values = buffer![1.234f64, 2.718, f64::consts::PI, 4.0];
        let array = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_some());
        assert_eq!(
            encoded
                .encoded()
                .into_primitive()
                .unwrap()
                .as_slice::<i64>(),
            vec![1234i64, 2718, 1234, 4000] // fill forward
        );
        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        let decoded = decompress(encoded).unwrap();
        assert_eq!(values.as_slice(), decoded.as_slice::<f64>());
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
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_some());

        assert_eq!(encoded.exponents(), Exponents { e: 16, f: 13 });

        for idx in 0..3 {
            let s = scalar_at(encoded.as_ref(), idx).unwrap();
            assert!(s.is_valid());
        }

        let s = scalar_at(encoded.as_ref(), 4).unwrap();
        assert!(s.is_null());

        let _decoded = decompress(encoded).unwrap();
    }

    #[test]
    fn roundtrips_close_fractional() {
        let original = PrimitiveArray::from_iter([195.26274f32, 195.27837, -48.815685]);
        let alp_arr = alp_encode(&original).unwrap();
        let decompressed = alp_arr.into_primitive().unwrap();
        assert_eq!(original.as_slice::<f32>(), decompressed.as_slice::<f32>());
    }

    #[test]
    fn roundtrips_all_null() {
        let original = PrimitiveArray::new(
            Buffer::from_iter([195.26274f64, f64::consts::PI, -48.815685]),
            Validity::AllInvalid,
        );
        let alp_arr = alp_encode(&original).unwrap();
        let decompressed = alp_arr.into_primitive().unwrap();
        assert_eq!(original.as_slice::<f64>(), decompressed.as_slice::<f64>());
        assert_eq!(original.validity(), decompressed.validity());
    }
}
