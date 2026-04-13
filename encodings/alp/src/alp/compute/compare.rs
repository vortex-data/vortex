// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;
use crate::ALPFloat;
use crate::match_each_alp_float_ptype;

// TODO(joe): add fuzzing.

impl CompareKernel for ALP {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
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
            let pscalar = const_scalar.as_primitive_opt().ok_or_else(|| {
                vortex_err!(
                    "ALP Compare RHS had the wrong type {}, expected {}",
                    const_scalar,
                    const_scalar.dtype()
                )
            })?;

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

/// We can compare a scalar to an ALPArray by encoding the scalar into the ALP domain and comparing
/// the encoded value to the encoded values in the ALPArray. There are fixups when the value doesn't
/// encode into the ALP domain.
fn alp_scalar_compare<F: ALPFloat + Into<Scalar>>(
    alp: ArrayView<ALP>,
    value: F,
    operator: CompareOperator,
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
    // If the scalar doesn't fit into the ALP domain,
    // it cannot be equal to any values in the encoded array.
    let encoded = F::encode_single(value, alp.exponents());
    match encoded {
        Some(encoded) => {
            let s = ConstantArray::new(encoded, alp.len());
            Ok(Some(
                alp.encoded()
                    .binary(s.into_array(), Operator::from(operator))?,
            ))
        }
        None => match operator {
            // Since this value is not encodable it cannot be equal to any value in the encoded
            // array.
            CompareOperator::Eq => Ok(Some(ConstantArray::new(false, alp.len()).into_array())),
            // Since this value is not encodable it cannot be equal to any value in the encoded
            // array, hence != to all values in the encoded array.
            CompareOperator::NotEq => Ok(Some(ConstantArray::new(true, alp.len()).into_array())),
            CompareOperator::Gt | CompareOperator::Gte => {
                // Per IEEE 754 totalOrder semantics the ordering is -Nan < -Inf < Inf < Nan.
                // All values in the encoded array are definitely finite
                let is_not_finite = NativePType::is_infinite(value) || NativePType::is_nan(value);
                if is_not_finite {
                    Ok(Some(
                        ConstantArray::new(value.is_sign_negative(), alp.len()).into_array(),
                    ))
                } else {
                    Ok(Some(
                        alp.encoded().binary(
                            ConstantArray::new(F::encode_above(value, exponents), alp.len())
                                .into_array(),
                            // Since the encoded value is unencodable gte is equivalent to gt.
                            // Consider a value v, between two encodable values v_l (just less) and
                            // v_a (just above), then for all encodable values (u), v > u <=> v_g >= u
                            Operator::Gte,
                        )?,
                    ))
                }
            }
            CompareOperator::Lt | CompareOperator::Lte => {
                // Per IEEE 754 totalOrder semantics the ordering is -Nan < -Inf < Inf < Nan.
                // All values in the encoded array are definitely finite
                let is_not_finite = NativePType::is_infinite(value) || NativePType::is_nan(value);
                if is_not_finite {
                    Ok(Some(
                        ConstantArray::new(value.is_sign_positive(), alp.len()).into_array(),
                    ))
                } else {
                    Ok(Some(
                        alp.encoded().binary(
                            ConstantArray::new(F::encode_below(value, exponents), alp.len())
                                .into_array(),
                            // Since the encoded values unencodable lt is equivalent to lte.
                            // See Gt | Gte for further explanation.
                            Operator::Lte,
                        )?,
                    ))
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;

    use super::*;
    use crate::alp_encode;

    fn test_alp_compare<F: ALPFloat + Into<Scalar>>(
        alp: ArrayView<ALP>,
        value: F,
        operator: CompareOperator,
    ) -> Option<ArrayRef>
    where
        F::ALPInt: Into<Scalar>,
        <F as ALPFloat>::ALPInt: Debug,
    {
        alp_scalar_compare(alp, value, operator).unwrap()
    }

    #[test]
    fn basic_comparison_test() {
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
        let encoded = alp_encode(array.as_view(), None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![1234; 1025]
        );

        let r = alp_scalar_compare(encoded.as_view(), 1.3_f32, CompareOperator::Eq)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([false; 1025]);
        assert_arrays_eq!(r, expected);

        let r = alp_scalar_compare(encoded.as_view(), 1.234f32, CompareOperator::Eq)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([true; 1025]);
        assert_arrays_eq!(r, expected);
    }

    #[test]
    fn comparison_with_unencodable_value() {
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
        let encoded = alp_encode(array.as_view(), None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![1234; 1025]
        );

        let r_eq = alp_scalar_compare(encoded.as_view(), 1.234444_f32, CompareOperator::Eq)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([false; 1025]);
        assert_arrays_eq!(r_eq, expected);

        let r_neq = alp_scalar_compare(encoded.as_view(), 1.234444f32, CompareOperator::NotEq)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([true; 1025]);
        assert_arrays_eq!(r_neq, expected);
    }

    #[test]
    fn comparison_range() {
        let array = PrimitiveArray::from_iter([0.0605_f32; 10]);
        let encoded = alp_encode(array.as_view(), None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![605; 10]
        );

        // !(0.0605_f32 >= 0.06051_f32);
        let r_gte = alp_scalar_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Gte)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([false; 10]);
        assert_arrays_eq!(r_gte, expected);

        // (0.0605_f32 > 0.06051_f32);
        let r_gt = alp_scalar_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Gt)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([false; 10]);
        assert_arrays_eq!(r_gt, expected);

        // 0.0605_f32 <= 0.06051_f32;
        let r_lte = alp_scalar_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lte)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lte, expected);

        //0.0605_f32 < 0.06051_f32;
        let r_lt = alp_scalar_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lt)
            .unwrap()
            .unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lt, expected);
    }

    #[test]
    fn comparison_zeroes() {
        let array = PrimitiveArray::from_iter([0.0_f32; 10]);
        let encoded = alp_encode(array.as_view(), None).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().as_slice::<i32>(),
            vec![0; 10]
        );

        let r_gte =
            test_alp_compare(encoded.as_view(), -0.00000001_f32, CompareOperator::Gte).unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gte, expected);

        let r_gte = test_alp_compare(encoded.as_view(), -0.0_f32, CompareOperator::Gte).unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gte, expected);

        let r_gt =
            test_alp_compare(encoded.as_view(), -0.0000000001f32, CompareOperator::Gt).unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gt, expected);

        let r_gte = test_alp_compare(encoded.as_view(), -0.0_f32, CompareOperator::Gt).unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gte, expected);

        let r_lte = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lte).unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lte, expected);

        let r_lt = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lt).unwrap();
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lt, expected);

        let r_lt = test_alp_compare(encoded.as_view(), -0.00001_f32, CompareOperator::Lt).unwrap();
        let expected = BoolArray::from_iter([false; 10]);
        assert_arrays_eq!(r_lt, expected);
    }

    #[test]
    fn compare_with_patches() {
        let array =
            PrimitiveArray::from_iter([1.234f32, 1.5, 19.0, std::f32::consts::E, 1_000_000.9]);
        let encoded = alp_encode(array.as_view(), None).unwrap();
        assert!(encoded.patches().is_some());

        // Not supported!
        assert!(
            alp_scalar_compare(encoded.as_view(), 1_000_000.9_f32, CompareOperator::Eq)
                .unwrap()
                .is_none()
        )
    }

    #[test]
    fn compare_to_null() {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded = alp_encode(array.as_view(), None).unwrap();

        let other = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
            array.len(),
        );

        let r = encoded
            .into_array()
            .binary(other.into_array(), Operator::Eq)
            .unwrap();
        // Comparing to null yields null results
        let expected = BoolArray::from_iter([None::<bool>; 10]);
        assert_arrays_eq!(r, expected);
    }

    #[rstest]
    #[case(f32::NAN, false)]
    #[case(-1.0f32 / 0.0f32, true)]
    #[case(f32::INFINITY, false)]
    #[case(f32::NEG_INFINITY, true)]
    fn compare_to_non_finite_gt(#[case] value: f32, #[case] result: bool) {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded = alp_encode(array.as_view(), None).unwrap();

        let r = test_alp_compare(encoded.as_view(), value, CompareOperator::Gt).unwrap();
        let expected = BoolArray::from_iter([result; 10]);
        assert_arrays_eq!(r, expected);
    }

    #[rstest]
    #[case(f32::NAN, true)]
    #[case(-1.0f32 / 0.0f32, false)]
    #[case(f32::INFINITY, true)]
    #[case(f32::NEG_INFINITY, false)]
    fn compare_to_non_finite_lt(#[case] value: f32, #[case] result: bool) {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded = alp_encode(array.as_view(), None).unwrap();

        let r = test_alp_compare(encoded.as_view(), value, CompareOperator::Lt).unwrap();
        let expected = BoolArray::from_iter([result; 10]);
        assert_arrays_eq!(r, expected);
    }
}
