use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::{ArrayData, ArrayLen, IntoArrayData};
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_scalar::{PrimitiveScalar, Scalar};

use crate::{match_each_alp_float_ptype, ALPArray, ALPEncoding, ALPFloat};

impl CompareFn<ALPArray> for ALPEncoding {
    fn compare(
        &self,
        lhs: &ALPArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        if let Some(const_scalar) = rhs.as_constant() {
            let pscalar = PrimitiveScalar::try_from(&const_scalar)?;

            match_each_alp_float_ptype!(pscalar.ptype(), |$T| {
                match pscalar.typed_value::<$T>() {
                    Some(value) => return alp_scalar_compare(lhs, value, operator).map(Some),
                    None => return Ok(Some(ConstantArray::new(
                        Scalar::bool(false, Nullability::Nullable), lhs.len()
                    ).into_array())),
                }
            });
        }

        Ok(None)
    }
}

fn alp_scalar_compare<F: ALPFloat + Into<Scalar>>(
    alp: &ALPArray,
    value: F,
    operator: Operator,
) -> VortexResult<ArrayData>
where
    F::ALPInt: Into<Scalar>,
{
    let encoded = F::encode_single(value, alp.exponents());
    match encoded {
        Ok(encoded) => {
            let s = ConstantArray::new(encoded, alp.len());
            compare(alp.encoded(), s.as_ref(), operator)
        }
        Err(exception) => {
            if let Some(patches) = alp.patches().as_ref() {
                let s = ConstantArray::new(exception, alp.len());
                compare(patches, s.as_ref(), operator)
            } else {
                Ok(
                    ConstantArray::new(Scalar::bool(false, Nullability::Nullable), alp.len())
                        .into_array(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::IntoArrayVariant;
    use vortex_dtype::{DType, Nullability, PType};

    use super::*;
    use crate::alp_encode;

    #[test]
    fn basic_comparison_test() {
        let array = PrimitiveArray::from(vec![1.234f32; 1025]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded
                .encoded()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            vec![1234; 1025]
        );

        let r = alp_scalar_compare(&encoded, 1.3_f32, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(!v);
        }

        let r = alp_scalar_compare(&encoded, 1.234f32, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(v);
        }
    }

    #[test]
    fn compare_with_patches() {
        let array =
            PrimitiveArray::from(vec![1.234f32, 1.5, 19.0, std::f32::consts::E, 1_000_000.9]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_some());

        let r = alp_scalar_compare(&encoded, 1_000_000.9_f32, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        let buffer = r.boolean_buffer();
        assert!(buffer.value(buffer.len() - 1));
    }

    #[test]
    fn compare_to_null() {
        let array = PrimitiveArray::from(vec![1.234f32; 1025]);
        let encoded = alp_encode(&array).unwrap();

        let other = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
            array.len(),
        );

        let r = compare(encoded, other.as_ref(), Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(!v);
        }
    }
}
