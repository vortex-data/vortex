// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;
use crate::ALPFloat;
use crate::alp::compute::predicate::apply_patch_predicate;
use crate::alp::compute::predicate::constant_predicate;
use crate::match_each_alp_float_ptype;

impl CompareKernel for ALP {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(const_scalar) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(pscalar) = const_scalar.as_primitive_opt() else {
            return Ok(None);
        };
        if pscalar.ptype() != lhs.dtype().as_ptype() {
            return Ok(None);
        }

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        match_each_alp_float_ptype!(pscalar.ptype(), |T| {
            let value = pscalar
                .typed_value::<T>()
                .vortex_expect("compare adaptor strips null constants");
            alp_scalar_compare(lhs, value, operator, nullability, ctx).map(Some)
        })
    }
}

/// Compares an ALP array to a constant without decoding it.
///
/// A constant that round-trips through the exponents compares directly against the encoded
/// integers. Otherwise it falls strictly between two encodable values, so an ordering compares
/// against the nearest encodable neighbour and an equality has a constant answer. Patched rows
/// hold floats that ALP could not encode and are re-evaluated exactly.
fn alp_scalar_compare<F>(
    alp: ArrayView<'_, ALP>,
    value: F,
    operator: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    F: ALPFloat + NativePType,
    F::ALPInt: NativePType + Into<PValue>,
{
    let exponents = alp.exponents();
    let is_finite = !NativePType::is_infinite(value) && !NativePType::is_nan(value);

    let encoded = match F::encode_single(value, exponents) {
        Some(encoded) => compare_encoded(alp, encoded, Operator::from(operator), nullability)?,
        None => match operator {
            CompareOperator::Eq => constant_predicate(alp, false, nullability)?,
            CompareOperator::NotEq => constant_predicate(alp, true, nullability)?,
            CompareOperator::Gt | CompareOperator::Gte => {
                if is_finite {
                    compare_encoded(
                        alp,
                        F::encode_above(value, exponents),
                        Operator::Gte,
                        nullability,
                    )?
                } else {
                    constant_predicate(alp, value.is_sign_negative(), nullability)?
                }
            }
            CompareOperator::Lt | CompareOperator::Lte => {
                if is_finite {
                    compare_encoded(
                        alp,
                        F::encode_below(value, exponents),
                        Operator::Lte,
                        nullability,
                    )?
                } else {
                    constant_predicate(alp, value.is_sign_positive(), nullability)?
                }
            }
        },
    };

    let Some(patches) = alp.patches() else {
        return Ok(encoded);
    };
    let predicate = compare_predicate::<F>(operator);
    apply_patch_predicate(encoded, &patches, |patch| predicate(patch, value), ctx)
}

fn compare_encoded<I: NativePType + Into<PValue>>(
    alp: ArrayView<'_, ALP>,
    encoded: I,
    operator: Operator,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    alp.encoded().binary(
        ConstantArray::new(Scalar::primitive(encoded, nullability), alp.len()).into_array(),
        operator,
    )
}

/// The total-order comparison the primitive kernel applies to decoded floats.
fn compare_predicate<F: NativePType>(operator: CompareOperator) -> fn(F, F) -> bool {
    match operator {
        CompareOperator::Eq => F::is_eq,
        CompareOperator::NotEq => |lhs, rhs| !lhs.is_eq(rhs),
        CompareOperator::Lt => F::is_lt,
        CompareOperator::Lte => F::is_le,
        CompareOperator::Gt => F::is_gt,
        CompareOperator::Gte => F::is_ge,
    }
}

#[cfg(test)]
mod tests {
    use std::f32;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ALPArray;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn test_alp_compare(alp: ArrayView<ALP>, value: f32, operator: CompareOperator) -> ArrayRef {
        alp_scalar_compare(
            alp,
            value,
            operator,
            Nullability::NonNullable,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
    }

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// Asserts the kernel engages for `encoded` and matches the comparison over `values`.
    fn assert_matches_primitive(
        encoded: &ALPArray,
        values: &PrimitiveArray,
        constant: Scalar,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let rhs = ConstantArray::new(constant, values.len()).into_array();
        for operator in ALL_OPERATORS {
            let actual =
                <ALP as CompareKernel>::compare(encoded.as_view(), &rhs, operator, &mut ctx)?
                    .unwrap_or_else(|| panic!("ALP compare kernel must engage for {operator:?}"));
            let expected = values
                .clone()
                .into_array()
                .binary(rhs.clone(), Operator::from(operator))?;
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn basic_comparison_test() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        assert!(encoded.patches().is_none());
        let encoded_prim = encoded
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_eq!(encoded_prim.as_slice::<i32>(), vec![1234; 1025]);

        let r = test_alp_compare(encoded.as_view(), 1.3_f32, CompareOperator::Eq);
        let expected = BoolArray::from_iter([false; 1025]);
        assert_arrays_eq!(r, expected, &mut ctx);

        let r = test_alp_compare(encoded.as_view(), 1.234f32, CompareOperator::Eq);
        let expected = BoolArray::from_iter([true; 1025]);
        assert_arrays_eq!(r, expected, &mut ctx);
    }

    #[test]
    fn comparison_with_unencodable_value() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter([1.234f32; 1025]);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        assert!(encoded.patches().is_none());
        let encoded_prim = encoded
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_eq!(encoded_prim.as_slice::<i32>(), vec![1234; 1025]);

        let r_eq = test_alp_compare(encoded.as_view(), 1.234444_f32, CompareOperator::Eq);
        let expected = BoolArray::from_iter([false; 1025]);
        assert_arrays_eq!(r_eq, expected, &mut ctx);

        let r_neq = test_alp_compare(encoded.as_view(), 1.234444f32, CompareOperator::NotEq);
        let expected = BoolArray::from_iter([true; 1025]);
        assert_arrays_eq!(r_neq, expected, &mut ctx);
    }

    #[test]
    fn comparison_range() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter([0.0605_f32; 10]);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        assert!(encoded.patches().is_none());
        let encoded_prim = encoded
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_eq!(encoded_prim.as_slice::<i32>(), vec![605; 10]);

        // !(0.0605_f32 >= 0.06051_f32);
        let r_gte = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Gte);
        let expected = BoolArray::from_iter([false; 10]);
        assert_arrays_eq!(r_gte, expected, &mut ctx);

        // (0.0605_f32 > 0.06051_f32);
        let r_gt = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Gt);
        let expected = BoolArray::from_iter([false; 10]);
        assert_arrays_eq!(r_gt, expected, &mut ctx);

        // 0.0605_f32 <= 0.06051_f32;
        let r_lte = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lte);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lte, expected, &mut ctx);

        // 0.0605_f32 < 0.06051_f32;
        let r_lt = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lt);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lt, expected, &mut ctx);
    }

    #[test]
    fn comparison_zeroes() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter([0.0_f32; 10]);
        let encoded = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        assert!(encoded.patches().is_none());
        let encoded_prim = encoded
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_eq!(encoded_prim.as_slice::<i32>(), vec![0; 10]);

        let r_gte = test_alp_compare(encoded.as_view(), -0.00000001_f32, CompareOperator::Gte);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gte, expected, &mut ctx);

        let r_gte = test_alp_compare(encoded.as_view(), -0.0_f32, CompareOperator::Gte);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gte, expected, &mut ctx);

        let r_gt = test_alp_compare(encoded.as_view(), -0.0000000001f32, CompareOperator::Gt);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gt, expected, &mut ctx);

        let r_gte = test_alp_compare(encoded.as_view(), -0.0_f32, CompareOperator::Gt);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_gte, expected, &mut ctx);

        let r_lte = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lte);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lte, expected, &mut ctx);

        let r_lt = test_alp_compare(encoded.as_view(), 0.06051_f32, CompareOperator::Lt);
        let expected = BoolArray::from_iter([true; 10]);
        assert_arrays_eq!(r_lt, expected, &mut ctx);

        let r_lt = test_alp_compare(encoded.as_view(), -0.00001_f32, CompareOperator::Lt);
        let expected = BoolArray::from_iter([false; 10]);
        assert_arrays_eq!(r_lt, expected, &mut ctx);
    }

    #[rstest]
    #[case::encodable(1.5f32)]
    #[case::patched_value(1_000_000.9f32)]
    #[case::unencodable(1.234444f32)]
    #[case::nan(f32::NAN)]
    #[case::infinity(f32::INFINITY)]
    fn compare_with_patches(#[case] constant: f32) -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([
            1.234f32,
            1.5,
            19.0,
            f32::consts::E,
            1_000_000.9,
            f32::NAN,
            f32::NEG_INFINITY,
        ]);
        let encoded = alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx())?;
        assert!(encoded.patches().is_some());

        assert_matches_primitive(&encoded, &array, constant.into())
    }

    #[test]
    fn compare_sliced_with_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<f32> = (0..3000)
            .map(|i| {
                if i % 97 == 0 {
                    f32::consts::PI * i as f32
                } else {
                    i as f32 / 100.0
                }
            })
            .collect();
        let array = PrimitiveArray::from_iter(values);
        let encoded = alp_encode(array.as_view(), None, &mut ctx)?;
        assert!(encoded.patches().is_some());

        let sliced = encoded.into_array().slice(1_000..2_500)?;
        let expected_values = array.into_array().slice(1_000..2_500)?;
        let rhs = ConstantArray::new(f32::consts::PI * 1_164.0, sliced.len()).into_array();
        for operator in ALL_OPERATORS {
            let actual = sliced
                .clone()
                .binary(rhs.clone(), Operator::from(operator))?;
            let expected = expected_values
                .clone()
                .binary(rhs.clone(), Operator::from(operator))?;
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[rstest]
    #[case::encodable(1.5f32)]
    #[case::patched_value(1_000_000.9f32)]
    #[case::unencodable(1.234444f32)]
    #[case::nan(f32::NAN)]
    #[case::neg_infinity(f32::NEG_INFINITY)]
    fn compare_nullable(#[case] constant: f32) -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([
            Some(1.234f32),
            None,
            Some(1.5),
            Some(1_000_000.9),
            None,
            Some(f32::NAN),
        ]);
        let encoded = alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx())?;

        assert_matches_primitive(&encoded, &array, constant.into())?;
        assert_matches_primitive(
            &encoded,
            &array,
            Scalar::primitive(constant, Nullability::Nullable),
        )
    }

    #[test]
    fn compare_nullable_constant_result_keeps_validity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1.234f32), None, Some(1.234)]);
        let encoded = alp_encode(array.as_view(), None, &mut ctx)?;
        assert!(encoded.patches().is_none());

        let actual = alp_scalar_compare(
            encoded.as_view(),
            1.234444f32,
            CompareOperator::Eq,
            Nullability::Nullable,
            &mut ctx,
        )?;
        let expected = BoolArray::from_iter([Some(false), None, Some(false)]);
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn compare_to_null() {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();

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
        assert_arrays_eq!(r, expected, &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case(f32::NAN, false)]
    #[case(-1.0f32 / 0.0f32, true)]
    #[case(f32::INFINITY, false)]
    #[case(f32::NEG_INFINITY, true)]
    fn compare_to_non_finite_gt(#[case] value: f32, #[case] result: bool) {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();

        let r = test_alp_compare(encoded.as_view(), value, CompareOperator::Gt);
        let expected = BoolArray::from_iter([result; 10]);
        assert_arrays_eq!(r, expected, &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case(f32::NAN, true)]
    #[case(-1.0f32 / 0.0f32, false)]
    #[case(f32::INFINITY, true)]
    #[case(f32::NEG_INFINITY, false)]
    fn compare_to_non_finite_lt(#[case] value: f32, #[case] result: bool) {
        let array = PrimitiveArray::from_iter([1.234f32; 10]);
        let encoded =
            alp_encode(array.as_view(), None, &mut SESSION.create_execution_ctx()).unwrap();

        let r = test_alp_compare(encoded.as_view(), value, CompareOperator::Lt);
        let expected = BoolArray::from_iter([result; 10]);
        assert_arrays_eq!(r, expected, &mut SESSION.create_execution_ctx());
    }
}
