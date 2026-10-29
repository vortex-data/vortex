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
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;

use crate::FoR;
use crate::FoRReferences;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;

impl CompareKernel for FoR {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
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
                );
            })
        }

        Ok(None)
    }
}

/// Compares the array to a constant through its encoded values.
///
/// Equality survives the wrapping subtraction. Ordering compares unsigned distances from the
/// reference.
fn compare_constant<T>(
    lhs: ArrayView<'_, FoR>,
    rhs: T,
    nullability: Nullability,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + PrimInt + WrappingSub,
    T: TryFrom<PValue, Error = VortexError>,
    PValue: From<T>,
{
    // TODO(mk): support many references.
    let FoRReferences::Global(reference) = lhs.references() else {
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
            // No value is below the reference.
            if rhs < reference {
                let answer = matches!(operator, CompareOperator::Gt | CompareOperator::Gte);
                return constant_answer(lhs, answer, nullability).map(Some);
            }

            let distance = rhs.wrapping_sub(&reference);
            if T::PTYPE.is_unsigned_int() {
                compare_encoded(lhs, distance, nullability, operator).map(Some)
            } else {
                compare_signed_encoded(lhs, distance, nullability, operator).map(Some)
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

/// Orders signed encoded values as unsigned distances.
///
/// Distances above `T::MAX` read as negative, so each operator is one signed range or its
/// complement: `[0, distance)` for a non-negative `distance`, `[distance, -1]` otherwise.
fn compare_signed_encoded<T>(
    lhs: ArrayView<'_, FoR>,
    distance: T,
    nullability: Nullability,
    operator: CompareOperator,
) -> VortexResult<ArrayRef>
where
    T: NativePType + PrimInt + Into<PValue>,
{
    let below = matches!(operator, CompareOperator::Lt | CompareOperator::Lte);
    let (lower, upper, options, negate) = if distance >= T::zero() {
        let upper_strict = matches!(operator, CompareOperator::Lt | CompareOperator::Gte);
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: strictness(upper_strict),
        };
        (T::zero(), distance, options, !below)
    } else {
        let lower_strict = matches!(operator, CompareOperator::Lte | CompareOperator::Gt);
        let options = BetweenOptions {
            lower_strict: strictness(lower_strict),
            upper_strict: StrictComparison::NonStrict,
        };
        (distance, !T::zero(), options, below)
    };

    let bound = |value: T| {
        ConstantArray::new(Scalar::primitive(value, nullability), lhs.len()).into_array()
    };
    let in_range = lhs
        .encoded()
        .clone()
        .between(bound(lower), bound(upper), options)?;

    if negate { in_range.not() } else { Ok(in_range) }
}

fn strictness(strict: bool) -> StrictComparison {
    if strict {
        StrictComparison::Strict
    } else {
        StrictComparison::NonStrict
    }
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
        let session = VortexSession::empty();
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
        )
        .unwrap()
        .unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, true, false].map(Some)),
            &mut SESSION.create_execution_ctx()
        );

        for (op, expected) in [
            (CompareOperator::Lt, [true, false, true]),
            (CompareOperator::Lte, [true, true, true]),
            (CompareOperator::Gt, [false, false, false]),
            (CompareOperator::Gte, [false, true, false]),
        ] {
            let result = compare_constant(lhs.as_view(), 30i32, Nullability::NonNullable, op)
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

    /// Encodes the values against their minimum.
    fn encode_with_min<T>(
        values: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(T, PrimitiveArray)>
    where
        T: NativePType + WrappingSub,
        T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        let reference = values
            .clone()
            .into_array()
            .statistics()
            .compute_min::<T>(ctx)
            .vortex_expect("min");
        let encoded = PrimitiveArray::new(
            values
                .as_slice::<T>()
                .iter()
                .map(|v| v.wrapping_sub(&reference))
                .collect::<vortex_buffer::Buffer<T>>(),
            values.validity()?,
        );
        Ok((reference, encoded))
    }

    /// Asserts the kernel engages and matches the primitive comparison.
    fn assert_matches_primitive<T: Into<Scalar>>(
        array: &FoRArray,
        values: &PrimitiveArray,
        constants: impl IntoIterator<Item = T>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for constant in constants {
            let rhs = ConstantArray::new(constant, values.len()).into_array();
            for operator in ALL_OPERATORS {
                let actual = <FoR as CompareKernel>::compare(array.as_view(), &rhs, operator, ctx)?
                    .unwrap_or_else(|| panic!("FoR compare must engage for {operator:?}"));
                let expected = values
                    .clone()
                    .into_array()
                    .binary(rhs.clone(), Operator::from(operator))?;
                assert_arrays_eq!(actual, expected, ctx);
            }
        }
        Ok(())
    }

    /// Asserts FoR over a bit-packed child matches the primitive comparison.
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
        let (reference, encoded) = encode_with_min::<T>(&values, &mut ctx)?;
        let packed = BitPackedData::encode(&encoded.into_array(), bit_width, &mut ctx)?;
        let array = FoR::try_new(packed.into_array(), reference.into())?;

        assert_matches_primitive(&array, &values, constants, &mut ctx)
    }

    /// Distances above `i8::MAX` read as negative encoded values.
    #[test]
    fn signed_distances_above_max_match_primitive() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter(
            (i8::MIN..=i8::MAX).map(|v| (v % 7 != 3).then_some(v)),
        );
        let (reference, encoded) = encode_with_min::<i8>(&values, &mut ctx)?;
        let array = FoR::try_new(encoded.into_array(), reference.into())?;

        assert_matches_primitive(&array, &values, i8::MIN..=i8::MAX, &mut ctx)
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
