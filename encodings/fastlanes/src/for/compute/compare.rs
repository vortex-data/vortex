// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::PrimInt;
use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;

use crate::FoR;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;

impl CompareKernel for FoR {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(constant) = rhs.as_constant()
            && let Some(constant) = constant.as_primitive_opt()
            && constant.ptype() == lhs.dtype().as_ptype()
        {
            match_each_integer_ptype!(constant.ptype(), |T| {
                return compare_constant(
                    lhs,
                    constant
                        .typed_value::<T>()
                        .vortex_expect("null scalar handled in adaptor"),
                    rhs.dtype().nullability(),
                    operator,
                    ctx,
                );
            })
        }

        Ok(None)
    }
}

/// Compares the array to a constant through its encoded values.
///
/// Equality survives the wrapping subtraction, so it always compares against the wrapped
/// constant. Ordering needs `encoded = value - reference` to be exact. Encoding subtracts the
/// minimum, so the encoded values are non-negative distances, which an unsigned type holds
/// exactly. A signed type also needs the distances to stay non-negative in its own reading,
/// which the minimum of the encoded values proves; otherwise the comparison falls back to
/// decoding.
fn compare_constant<T>(
    lhs: ArrayView<'_, FoR>,
    rhs: T,
    nullability: Nullability,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + PrimInt + WrappingSub,
    T: TryFrom<PValue, Error = VortexError> + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    PValue: From<T>,
{
    // TODO(mk): support many references.
    let Some(reference) = lhs.constant_reference() else {
        return Ok(None);
    };
    let reference = reference
        .as_primitive()
        .typed_value::<T>()
        .vortex_expect("FoR reference is non-null and matches the array ptype");

    match operator {
        CompareOperator::Eq | CompareOperator::NotEq => {
            compare_encoded(lhs, rhs.wrapping_sub(&reference), nullability, operator).map(Some)
        }
        CompareOperator::Lt | CompareOperator::Lte | CompareOperator::Gt | CompareOperator::Gte => {
            if T::PTYPE.is_signed_int() && !encoded_is_non_negative::<T>(lhs, ctx)? {
                return Ok(None);
            }

            match rhs.checked_sub(&reference) {
                Some(delta) => compare_encoded(lhs, delta, nullability, operator).map(Some),
                None => {
                    // The distance does not fit the type: either the constant lies below the
                    // reference, and so below every value, or it lies more than `T::MAX` above
                    // the reference, and so above every value.
                    let every_value_greater = rhs < reference;
                    let answer = match operator {
                        CompareOperator::Lt | CompareOperator::Lte => !every_value_greater,
                        _ => every_value_greater,
                    };
                    constant_answer(lhs, answer, nullability).map(Some)
                }
            }
        }
    }
}

fn compare_encoded<T: NativePType + Into<PValue>>(
    lhs: ArrayView<'_, FoR>,
    encoded: T,
    nullability: Nullability,
    operator: CompareOperator,
) -> VortexResult<ArrayRef> {
    lhs.encoded().binary(
        ConstantArray::new(Scalar::primitive(encoded, nullability), lhs.len()).into_array(),
        Operator::from(operator),
    )
}

/// Whether every valid encoded value is non-negative in the signed reading of `T`.
fn encoded_is_non_negative<T>(lhs: ArrayView<'_, FoR>, ctx: &mut ExecutionCtx) -> VortexResult<bool>
where
    T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
{
    Ok(lhs
        .encoded()
        .statistics()
        .compute_min::<T>(ctx)
        .is_none_or(|min| min.is_ge(T::zero())))
}

/// A constant answer for every row that still carries the array's validity.
fn constant_answer(
    lhs: ArrayView<'_, FoR>,
    value: bool,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    Ok(match lhs.validity()?.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(Scalar::bool(value, nullability), len).into_array()
        }
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array()
        }
        Validity::Array(validity) => ConstantArray::new(Scalar::bool(value, nullability), len)
            .into_array()
            .mask(validity)?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedData;
    use crate::FoR;
    use crate::FoRArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn for_arr(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[test]
    fn test_compare_constant() {
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::AllValid).into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            30i32,
            Nullability::NonNullable,
            CompareOperator::Eq,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, true, false].map(Some)),
            &mut SESSION.create_execution_ctx()
        );

        let result = compare_constant(
            lhs.as_view(),
            12i32,
            Nullability::NonNullable,
            CompareOperator::NotEq,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, true, false].map(Some)),
            &mut SESSION.create_execution_ctx()
        );

        // The minimum of the encoded values is non-negative, so ordering engages.
        for (op, expected) in [
            (CompareOperator::Lt, [true, false, true]),
            (CompareOperator::Lte, [true, true, true]),
            (CompareOperator::Gt, [false, false, false]),
            (CompareOperator::Gte, [false, true, false]),
        ] {
            let result = compare_constant(
                lhs.as_view(),
                30i32,
                Nullability::NonNullable,
                op,
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap();
            assert_arrays_eq!(
                result,
                BoolArray::from_iter(expected.map(Some)),
                &mut SESSION.create_execution_ctx()
            );
        }
    }

    #[test]
    fn negative_signed_encoded_falls_back() {
        // A negative encoded value breaks the order of the encoded domain.
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, -5, 2), Validity::AllValid).into_array(),
            Scalar::from(10),
        );

        for op in [
            CompareOperator::Lt,
            CompareOperator::Lte,
            CompareOperator::Gt,
            CompareOperator::Gte,
        ] {
            assert!(
                compare_constant(
                    lhs.as_view(),
                    30i32,
                    Nullability::NonNullable,
                    op,
                    &mut SESSION.create_execution_ctx(),
                )
                .unwrap()
                .is_none()
            );
        }
    }

    #[test]
    fn test_compare_nullable_constant() {
        let reference = Scalar::from(0);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 20, 2), Validity::NonNullable).into_array(),
            reference,
        );

        assert_eq!(
            compare_constant(
                lhs.as_view(),
                30i32,
                Nullability::Nullable,
                CompareOperator::Eq,
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap()
            .dtype(),
            &DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            compare_constant(
                lhs.as_view(),
                30i32,
                Nullability::NonNullable,
                CompareOperator::Eq,
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap()
            .unwrap()
            .dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn compare_non_encodable_constant() {
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = for_arr(
            PrimitiveArray::new(buffer!(0i32, 10, 1), Validity::AllValid).into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            -1i32,
            Nullability::NonNullable,
            CompareOperator::Eq,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, false].map(Some)),
            &mut SESSION.create_execution_ctx()
        );

        let result = compare_constant(
            lhs.as_view(),
            -1i32,
            Nullability::NonNullable,
            CompareOperator::NotEq,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, true, true].map(Some)),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn compare_large_constant() {
        let reference = Scalar::from(-9219218377546224477i64);
        let lhs = for_arr(
            PrimitiveArray::new(
                buffer![0i64, 9654309310445864926u64 as i64],
                Validity::AllValid,
            )
            .into_array(),
            reference,
        );

        let result = compare_constant(
            lhs.as_view(),
            435090932899640449i64,
            Nullability::Nullable,
            CompareOperator::Eq,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(false), Some(true)]),
            &mut SESSION.create_execution_ctx()
        );

        let result = compare_constant(
            lhs.as_view(),
            435090932899640449i64,
            Nullability::Nullable,
            CompareOperator::NotEq,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), Some(false)]),
            &mut SESSION.create_execution_ctx()
        );
    }

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// Builds a FoR array over a bit-packed child, the shape the compressor produces, and
    /// asserts the kernel engages for every operator and matches the primitive comparison.
    fn assert_bit_packed_matches_primitive<T>(
        values: PrimitiveArray,
        bit_width: u8,
        constants: impl IntoIterator<Item = T>,
    ) -> VortexResult<()>
    where
        T: NativePType + WrappingSub + Into<PValue> + Into<Scalar>,
        T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let reference = values
            .clone()
            .into_array()
            .statistics()
            .compute_min::<T>(&mut ctx)
            .vortex_expect("min");
        let encoded = PrimitiveArray::new(
            values
                .as_slice::<T>()
                .iter()
                .map(|v| v.wrapping_sub(&reference))
                .collect::<vortex_buffer::Buffer<T>>(),
            values.validity()?,
        );
        let packed = BitPackedData::encode(&encoded.into_array(), bit_width, &mut ctx)?;
        let array = FoR::try_new(packed.into_array(), reference.into())?;

        for constant in constants {
            let rhs = ConstantArray::new(constant, values.len()).into_array();
            for operator in ALL_OPERATORS {
                let actual =
                    <FoR as CompareKernel>::compare(array.as_view(), &rhs, operator, &mut ctx)?
                        .unwrap_or_else(|| panic!("FoR compare must engage for {operator:?}"));
                let expected = values
                    .clone()
                    .into_array()
                    .binary(rhs.clone(), Operator::from(operator))?;
                assert_arrays_eq!(actual, expected, &mut ctx);
            }
        }
        Ok(())
    }

    #[test]
    fn signed_bit_packed_matches_primitive() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter((0..2000i32).map(|i| {
            if i % 97 == 0 {
                -1_000 + i * 50
            } else {
                -1_000 + i % 40
            }
        }));
        let constants = [
            i32::MIN,
            -1_001,
            -1_000,
            -999,
            -980,
            0,
            98_900,
            98_901,
            i32::MAX,
        ];
        assert_bit_packed_matches_primitive(values, 6, constants)
    }

    #[test]
    fn unsigned_bit_packed_matches_primitive() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter((0..1500u16).map(|i| 1_000 + i % 70));
        let constants = [0u16, 999, 1_000, 1_001, 1_035, 1_069, 1_070, u16::MAX];
        assert_bit_packed_matches_primitive(values, 7, constants)
    }

    #[test]
    fn nullable_bit_packed_matches_primitive() -> VortexResult<()> {
        let values = PrimitiveArray::from_option_iter(
            (0..1200i64).map(|i| (i % 5 != 0).then_some(-50 + i % 30)),
        );
        let constants = [i64::MIN, -51, -50, -40, -21, -20, i64::MAX];
        assert_bit_packed_matches_primitive(values, 5, constants)
    }
}
