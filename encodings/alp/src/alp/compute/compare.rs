// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator, compare};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{PrimitiveScalar, Scalar};

use super::compare_common::{EncodedComparison, encode_for_comparison};
use crate::{ALPArray, ALPFloat, ALPVTable, match_each_alp_float_ptype};

// TODO(joe): add fuzzing.

impl CompareKernel for ALPVTable {
    fn compare(
        &self,
        lhs: &ALPArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
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

            match_each_alp_float_ptype!(pscalar.ptype(), |T| {
                match pscalar.typed_value::<T>() {
                    Some(value) => return alp_scalar_compare(lhs, value, operator),
                    None => vortex_bail!(
                        "Failed to convert scalar {:?} to ALP type {:?}",
                        pscalar,
                        pscalar.ptype()
                    ),
                }
            });
        }

        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(ALPVTable).lift());

/// We can compare a scalar to an ALPArray by encoding the scalar into the ALP domain and comparing
/// the encoded value to the encoded values in the ALPArray. There are fixups when the value doesn't
/// encode into the ALP domain.
///
/// This uses the common `encode_for_comparison` logic shared with the expression pushdown optimization.
fn alp_scalar_compare<F: ALPFloat + Into<Scalar>>(
    alp: &ALPArray,
    value: F,
    operator: Operator,
) -> VortexResult<Option<ArrayRef>>
where
    F::ALPInt: Into<Scalar>,
    <F as ALPFloat>::ALPInt: Debug,
{
    // TODO(joe): support patches, this is checked above.
    if alp.patches().is_some() {
        return Ok(None);
    }

    let exponents = alp.exponents();

    // Use the common comparison logic from compare_common.rs
    match encode_for_comparison(value, exponents, operator) {
        EncodedComparison::Encoded { value, operator } => {
            // Compare the encoded array with the encoded scalar value
            let s = ConstantArray::new(value, alp.len());
            Ok(Some(compare(alp.encoded(), s.as_ref(), operator)?))
        }
        EncodedComparison::Constant(result) => {
            // Return a constant result for all elements
            Ok(Some(ConstantArray::new(result, alp.len()).into_array()))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::{Operator, compare};
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
            .map(|a| a.to_bool().bit_buffer().iter().collect())
    }

    #[test]
    fn basic_comparison_test() {
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![1234; 1025]
        );

        let r = alp_scalar_compare(&encoded, 1.3_f32, Operator::Eq)
            .unwrap()
            .unwrap()
            .to_bool();

        for v in r.bit_buffer().iter() {
            assert!(!v);
        }

        let r = alp_scalar_compare(&encoded, 1.234f32, Operator::Eq)
            .unwrap()
            .unwrap()
            .to_bool();

        for v in r.bit_buffer().iter() {
            assert!(v);
        }
    }

    #[test]
    fn comparison_with_unencodable_value() {
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![1234; 1025]
        );

        #[allow(clippy::excessive_precision)]
        let r_eq = alp_scalar_compare(&encoded, 1.234444_f32, Operator::Eq)
            .unwrap()
            .unwrap()
            .to_bool();

        assert!(r_eq.bit_buffer().iter().all(|v| !v));

        #[allow(clippy::excessive_precision)]
        let r_neq = alp_scalar_compare(&encoded, 1.234444f32, Operator::NotEq)
            .unwrap()
            .unwrap()
            .to_bool();

        assert!(r_neq.bit_buffer().iter().all(|v| v));
    }

    #[test]
    fn comparison_range() {
        let array = PrimitiveArray::from_iter([0.0605_f32; 10]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![605; 10]
        );

        let r_gte = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Gte)
            .unwrap()
            .unwrap()
            .to_bool();

        // !(0.0605_f32 >= 0.06051_f32);
        assert!(r_gte.bit_buffer().iter().all(|v| !v));

        let r_gt = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Gt)
            .unwrap()
            .unwrap()
            .to_bool();

        // (0.0605_f32 > 0.06051_f32);
        assert!(r_gt.bit_buffer().iter().all(|v| !v));

        let r_lte = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Lte)
            .unwrap()
            .unwrap()
            .to_bool();

        // 0.0605_f32 <= 0.06051_f32;
        assert!(r_lte.bit_buffer().iter().all(|v| v));

        let r_lt = alp_scalar_compare(&encoded, 0.06051_f32, Operator::Lt)
            .unwrap()
            .unwrap()
            .to_bool();

        //0.0605_f32 < 0.06051_f32;
        assert!(r_lt.bit_buffer().iter().all(|v| v));
    }

    #[test]
    fn comparison_zeroes() {
        let array = PrimitiveArray::from_iter([0.0_f32; 10]);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![0; 10]
        );

        let r_gte = test_alp_compare(&encoded, -0.00000001_f32, Operator::Gte).unwrap();
        assert_eq!(r_gte, vec![true; 10]);

        let r_gte = test_alp_compare(&encoded, -0.0_f32, Operator::Gte).unwrap();
        assert_eq!(r_gte, vec![true; 10]);

        let r_gt = test_alp_compare(&encoded, -0.0000000001f32, Operator::Gt).unwrap();
        assert_eq!(r_gt, vec![true; 10]);

        let r_gte = test_alp_compare(&encoded, -0.0_f32, Operator::Gt).unwrap();
        assert_eq!(r_gte, vec![true; 10]);

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
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().is_some());

        // Not supported!
        assert!(
            alp_scalar_compare(&encoded, 1_000_000.9_f32, Operator::Eq)
                .unwrap()
                .is_none()
        )
    }

    #[test]
    fn compare_to_null() {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded = alp_encode(&array, None).unwrap();

        let other = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
            array.len(),
        );

        let r = compare(encoded.as_ref(), other.as_ref(), Operator::Eq)
            .unwrap()
            .to_bool();

        for v in r.bit_buffer().iter() {
            assert!(!v);
        }
    }

    #[rstest]
    #[case(f32::NAN, false)]
    #[case(-1.0f32 / 0.0f32, true)]
    #[case(f32::INFINITY, false)]
    #[case(f32::NEG_INFINITY, true)]
    fn compare_to_non_finite_gt(#[case] value: f32, #[case] result: bool) {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded = alp_encode(&array, None).unwrap();

        let gte = test_alp_compare(&encoded, value, Operator::Gt).unwrap();

        assert_eq!(gte, [result; 10]);
    }

    #[rstest]
    #[case(f32::NAN, true)]
    #[case(-1.0f32 / 0.0f32, false)]
    #[case(f32::INFINITY, true)]
    #[case(f32::NEG_INFINITY, false)]
    fn compare_to_non_finite_lt(#[case] value: f32, #[case] result: bool) {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded = alp_encode(&array, None).unwrap();

        let lt = test_alp_compare(&encoded, value, Operator::Lt).unwrap();

        assert_eq!(lt, [result; 10]);
    }
}
