use std::fmt::Debug;

use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData};
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::{PrimitiveScalar, Scalar};

use crate::{match_each_alp_float_ptype, ALPArray, ALPEncoding, ALPFloat};

// TODO(joe): add fuzzing.

impl CompareFn<ALPArray> for ALPEncoding {
    fn compare(
        &self,
        lhs: &ALPArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        if lhs.patches().is_some() {
            // TODO(joe): support patches
            return Ok(None);
        }
        if lhs.dtype().is_nullable() || rhs.dtype().is_nullable() {
            // TODO(joe): support nullability
            return Ok(None);
        }

        if let Some(const_scalar) = rhs.as_constant() {
            let pscalar = PrimitiveScalar::try_from(&const_scalar)?;

            match_each_alp_float_ptype!(pscalar.ptype(), |$T| {
                match pscalar.typed_value::<$T>() {
                    Some(value) => return alp_scalar_compare(lhs, value, operator),
                    None => vortex_bail!("Failed to convert scalar {:?} to ALP type {:?}", pscalar, pscalar.ptype()),
                }
            });
        }

        Ok(None)
    }
}

// We can compare a scalar to an ALPArray by encoding the scalar into the ALP domain and comparing
// the encoded value to the encoded values in the ALPArray. There are fixups when the value doesn't
// encode into the ALP domain.
fn alp_scalar_compare<F: ALPFloat + Into<Scalar>>(
    alp: &ALPArray,
    value: F,
    operator: Operator,
) -> VortexResult<Option<ArrayData>>
where
    F::ALPInt: Into<Scalar>,
    <F as ALPFloat>::ALPInt: Debug,
{
    // TODO(joe): support patches, this is checked above.
    if alp.patches().is_some() {
        return Ok(None);
    }

    let exponents = alp.exponents();
    // If the scalar doesn't fit into the ALP domain,
    // it cannot be equal to any values in the encoded array.
    let encoded = F::encode_single(value, alp.exponents());
    match encoded {
        Some(encoded) => {
            let s = ConstantArray::new(encoded, alp.len());
            Ok(Some(compare(alp.encoded(), s.as_ref(), operator)?))
        }
        None => match operator {
            // Since this value is not encodable it cannot be equal to any value in the encoded
            // array.
            Operator::Eq => Ok(Some(ConstantArray::new(false, alp.len()).into_array())),
            // Since this value is not encodable it cannot be equal to any value in the encoded
            // array, hence != to all values in the encoded array.
            Operator::NotEq => Ok(Some(ConstantArray::new(true, alp.len()).into_array())),
            Operator::Gt | Operator::Gte => Ok(Some(compare(
                alp.encoded(),
                ConstantArray::new(F::encode_above(value, exponents), alp.len()),
                // Since the encoded value is unencodable gte is equivalent to gt.
                // Consider a value v, between two encodable values v_l (just less) and
                // v_a (just above), then for all encodable values (u), v > u <=> v_g >= u
                Operator::Gte,
            )?)),
            Operator::Lt | Operator::Lte => Ok(Some(compare(
                alp.encoded(),
                ConstantArray::new(F::encode_below(value, exponents), alp.len()),
                // Since the encoded values unencodable lt is equivalent to lte.
                // See Gt | Gte for further explanation.
                Operator::Lte,
            )?)),
        },
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::{compare, Operator};
    use vortex_array::{ArrayLen, IntoArrayVariant};
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use super::*;
    use crate::alp_encode;

    fn test_alp_compare<F: ALPFloat + Into<Scalar>>(
        alp: &ALPArray,
        value: F,
        operator: Operator,
    ) -> Option<Vec<bool>>
    where
        F::ALPInt: Into<Scalar>,
        <F as ALPFloat>::ALPInt: Debug,
    {
        alp_scalar_compare(alp, value, operator)
            .unwrap()
            .map(|a| a.into_bool().unwrap().boolean_buffer().iter().collect())
    }

    #[test]
    fn basic_comparison_test() {
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
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

        let r = alp_scalar_compare(&encoded, 1.3_f32, Operator::Eq)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(!v);
        }

        let r = alp_scalar_compare(&encoded, 1.234f32, Operator::Eq)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(v);
        }
    }

    #[test]
    fn comparison_with_unencodable_value() {
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
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

        #[allow(clippy::excessive_precision)]
        let r_eq = alp_scalar_compare(&encoded, 1.234444_f32, Operator::Eq)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        assert!(r_eq.boolean_buffer().iter().all(|v| !v));

        #[allow(clippy::excessive_precision)]
        let r_neq = alp_scalar_compare(&encoded, 1.234444f32, Operator::NotEq)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        assert!(r_neq.boolean_buffer().iter().all(|v| v));
    }

    #[test]
    fn comparison_range() {
        let array = PrimitiveArray::from_iter([0.0605_f32; 10]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded
                .encoded()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            vec![605; 10]
        );

        let r_gte = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Gte)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        // !(0.0605_f32 >= 0.06051_f32);
        assert!(r_gte.boolean_buffer().iter().all(|v| !v));

        let r_gt = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Gt)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        // (0.0605_f32 > 0.06051_f32);
        assert!(r_gt.boolean_buffer().iter().all(|v| !v));

        let r_lte = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Lte)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        // 0.0605_f32 <= 0.06051_f32;
        assert!(r_lte.boolean_buffer().iter().all(|v| v));

        let r_lt = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Lt)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        //0.0605_f32 < 0.06051_f32;
        assert!(r_lt.boolean_buffer().iter().all(|v| v));
    }

    #[test]
    fn comparison_zeroes() {
        let array = PrimitiveArray::from_iter([0.0_f32; 10]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded
                .encoded()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            vec![0; 10]
        );

        let r_gte = test_alp_compare(&encoded, -0.00000001_f32, Operator::Gte).unwrap();
        assert_eq!(r_gte, vec![true; 10]);

        let r_gte = test_alp_compare(&encoded, -0.0_f32, Operator::Gte).unwrap();
        assert_eq!(r_gte, vec![true; 10]);

        let r_gt = test_alp_compare(&encoded, -0.0000000001f32, Operator::Gt).unwrap();
        assert_eq!(r_gt, vec![true; 10]);

        let r_gte = test_alp_compare(&encoded, -0.0_f32, Operator::Gt).unwrap();
        assert_eq!(r_gte, vec![false; 10]);

        let r_lte = test_alp_compare(&encoded, 0.06051_f32, Operator::Lte).unwrap();
        assert_eq!(r_lte, vec![true; 10]);

        let r_lt = test_alp_compare(&encoded, 0.06051_f32, Operator::Lt).unwrap();
        assert_eq!(r_lt, vec![true; 10]);

        let r_lt = test_alp_compare(&encoded, -0.00001_f32, Operator::Lt).unwrap();
        assert_eq!(r_lt, vec![false; 10]);
    }

    #[test]
    fn compare_with_patches() {
        let array =
            PrimitiveArray::from_iter([1.234f32, 1.5, 19.0, std::f32::consts::E, 1_000_000.9]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_some());

        // Not supported!
        assert!(alp_scalar_compare(&encoded, 1_000_000.9_f32, Operator::Eq)
            .unwrap()
            .is_none())
    }

    #[test]
    fn compare_to_null() {
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
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
