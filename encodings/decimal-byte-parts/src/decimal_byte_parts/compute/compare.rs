// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use Sign::Negative;
use num_traits::NumCast;
use num_traits::ToPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::ToI256;
use vortex_array::dtype::i256;
use vortex_array::match_each_decimal_value;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use crate::decimal_byte_parts::LOWER_PART_BITS;
use crate::decimal_byte_parts::compute::compare::Sign::Positive;

impl CompareKernel for DecimalByteParts {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let rhs_decimal = rhs_const
            .as_decimal()
            .decimal_value()
            .vortex_expect("checked for null in entry func");

        if lhs.lower_parts().is_empty() {
            compare_narrow(lhs, rhs_decimal, operator, nullability).map(Some)
        } else {
            compare_wide(lhs, rhs_decimal, operator, nullability).map(Some)
        }
    }
}

/// Compares a decimal held entirely in the most significant part against the constant.
fn compare_narrow(
    lhs: ArrayView<'_, DecimalByteParts>,
    rhs_decimal: DecimalValue,
    operator: CompareOperator,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let scalar_type = lhs.msp().dtype().with_nullability(nullability);
    match decimal_value_wrapper_to_primitive(rhs_decimal, lhs.msp().dtype().as_ptype()) {
        Ok(value) => {
            let encoded_scalar = Scalar::try_new(scalar_type, Some(value))?;
            let encoded_const = ConstantArray::new(encoded_scalar, lhs.len());
            lhs.msp()
                .binary(encoded_const.into_array(), Operator::from(operator))
        }
        Err(sign) => constant_answer(lhs, unconvertible_value(sign, operator, nullability)),
    }
}

/// Compares a decimal split across the most significant part and its lower parts.
///
/// The most significant part is the signed high word and the lower parts are unsigned words, so
/// the value order is the lexicographic order of the parts: the first part that differs decides,
/// and the last part carries the operator's strictness.
fn compare_wide(
    lhs: ArrayView<'_, DecimalByteParts>,
    rhs_decimal: DecimalValue,
    operator: CompareOperator,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let lower_parts = lhs.lower_parts().to_vec();
    let msp_ptype = lhs.msp().dtype().as_ptype();
    let constant = match_each_decimal_value!(rhs_decimal, |value| {
        value
            .to_i256()
            .vortex_expect("every DecimalValue fits i256")
    });

    let split = match split_constant(constant, lower_parts.len(), msp_ptype) {
        Ok(split) => split,
        Err(sign) => {
            return constant_answer(lhs, unconvertible_value(sign, operator, nullability));
        }
    };

    let last = lower_parts.len() - 1;
    let mut answer = compare_lower_part(&lower_parts[last], split.words[last], operator)?;
    for (part, word) in lower_parts[..last].iter().zip(&split.words).rev() {
        answer = lexicographic_step(part, Scalar::from(*word), operator, answer)?;
    }

    let msp_scalar = Scalar::primitive(split.msp, nullability)
        .cast(&lhs.msp().dtype().with_nullability(nullability))?;
    let answer = lexicographic_step(lhs.msp(), msp_scalar, operator, answer)?;

    // Only the most significant part is nullable, so combining it with the lower parts under
    // Kleene logic can turn a null row into `false` or `true`. Restore the array's validity.
    Ok(match lhs.validity()?.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => answer,
        Validity::AllInvalid => {
            ConstantArray::new(Scalar::null(DType::Bool(nullability)), lhs.len()).into_array()
        }
        Validity::Array(validity) => answer.mask(validity)?,
    })
}

/// Combines the comparison of one part with `rest`, the answer over the less significant parts.
fn lexicographic_step(
    part: &ArrayRef,
    constant: Scalar,
    operator: CompareOperator,
    rest: ArrayRef,
) -> VortexResult<ArrayRef> {
    let compare = |operator: Operator| {
        part.binary(
            ConstantArray::new(constant.clone(), part.len()).into_array(),
            operator,
        )
    };
    match operator {
        CompareOperator::Eq => compare(Operator::Eq)?.binary(rest, Operator::And),
        CompareOperator::NotEq => compare(Operator::NotEq)?.binary(rest, Operator::Or),
        CompareOperator::Lt | CompareOperator::Lte => compare(Operator::Lt)?.binary(
            compare(Operator::Eq)?.binary(rest, Operator::And)?,
            Operator::Or,
        ),
        CompareOperator::Gt | CompareOperator::Gte => compare(Operator::Gt)?.binary(
            compare(Operator::Eq)?.binary(rest, Operator::And)?,
            Operator::Or,
        ),
    }
}

/// Compares a non-nullable unsigned lower part against one word of the constant.
///
/// A part narrower than 64 bits cannot hold a word beyond its range, in which case every value
/// in the part is below the word.
fn compare_lower_part(
    part: &ArrayRef,
    word: u64,
    operator: CompareOperator,
) -> VortexResult<ArrayRef> {
    match Scalar::from(word).cast(part.dtype()) {
        Ok(constant) => part.binary(
            ConstantArray::new(constant, part.len()).into_array(),
            Operator::from(operator),
        ),
        Err(_) => {
            let below = matches!(
                operator,
                CompareOperator::Lt | CompareOperator::Lte | CompareOperator::NotEq
            );
            Ok(ConstantArray::new(below, part.len()).into_array())
        }
    }
}

/// The constant broken into the array's parts.
struct SplitConstant {
    msp: i64,
    words: Vec<u64>,
}

/// Splits `constant` into a most significant part that fits `msp_ptype` and `lower_parts`
/// 64-bit words, or reports on which side of the representable range it lies.
fn split_constant(
    constant: i256,
    lower_parts: usize,
    msp_ptype: PType,
) -> Result<SplitConstant, Sign> {
    let sign = if constant > i256::from(0i64) {
        Positive
    } else {
        Negative
    };
    let shifted = |bits: usize| constant >> i256::from(bits as i64);

    let msp = shifted(LOWER_PART_BITS * lower_parts)
        .to_i64()
        .ok_or(sign)?;
    let fits =
        match_each_signed_integer_ptype!(msp_ptype, |P| { <P as NumCast>::from(msp).is_some() });
    if !fits {
        return Err(sign);
    }

    let words = (0..lower_parts)
        .map(|index| {
            let (low, _) = shifted(LOWER_PART_BITS * (lower_parts - 1 - index)).to_parts();
            u64::try_from(low & 0xFFFF_FFFF_FFFF_FFFF).vortex_expect("masked to 64 bits")
        })
        .collect();
    Ok(SplitConstant { msp, words })
}

/// A constant answer for every row that still carries the array's validity.
fn constant_answer(lhs: ArrayView<'_, DecimalByteParts>, value: Scalar) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    Ok(
        match lhs
            .validity()?
            .union_nullability(value.dtype().nullability())
        {
            Validity::NonNullable | Validity::AllValid => {
                ConstantArray::new(value, len).into_array()
            }
            Validity::AllInvalid => {
                ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len)
                    .into_array()
            }
            Validity::Array(validity) => {
                ConstantArray::new(value, len).into_array().mask(validity)?
            }
        },
    )
}

#[derive(Debug, Clone, Copy)]
enum Sign {
    Positive,
    Negative,
}

fn unconvertible_value(sign: Sign, operator: CompareOperator, nullability: Nullability) -> Scalar {
    match operator {
        CompareOperator::Eq => Scalar::bool(false, nullability),
        CompareOperator::NotEq => Scalar::bool(true, nullability),
        CompareOperator::Gt | CompareOperator::Gte => {
            Scalar::bool(matches!(sign, Negative), nullability)
        }
        CompareOperator::Lt | CompareOperator::Lte => {
            Scalar::bool(matches!(sign, Positive), nullability)
        }
    }
}

fn decimal_value_wrapper_to_primitive(
    decimal_value: DecimalValue,
    ptype: PType,
) -> Result<ScalarValue, Sign> {
    match_each_integer_ptype!(ptype, |P| {
        decimal_value_to_primitive::<P>(decimal_value)
    })
}

fn decimal_value_to_primitive<P>(decimal_value: DecimalValue) -> Result<ScalarValue, Sign>
where
    P: IntegerPType + ToI256,
    ScalarValue: From<P>,
{
    match_each_decimal_value!(decimal_value, |decimal_v| {
        let Some(encoded) = <P as NumCast>::from(decimal_v) else {
            let decimal_i256 = decimal_v
                .to_i256()
                .vortex_expect("i256 is big enough for any DecimalValue");
            return if decimal_i256
                > P::max_value()
                    .to_i256()
                    .vortex_expect("i256 is big enough for any PType")
            {
                Err(Positive)
            } else {
                assert!(
                    decimal_i256
                        < P::min_value()
                            .to_i256()
                            .vortex_expect("i256 is big enough for any PType")
                );
                Err(Negative)
            };
        };
        Ok(ScalarValue::from(encoded))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::i256;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DecimalByteParts;
    use crate::DecimalBytePartsArray;
    use crate::decimal_byte_parts::testing::i128_parts;
    use crate::decimal_byte_parts::testing::i256_parts;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn compare_decimal_const() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).into_array(),
            decimal_dtype,
        )
        .unwrap()
        .into_array();
        let rhs = ConstantArray::new(
            Scalar::try_new(dtype, Some(DecimalValue::I64(400).into())).unwrap(),
            lhs.len(),
        );

        let res = lhs.binary(rhs.into_array(), Operator::Eq).unwrap();

        let expected = BoolArray::from_iter([Some(false), Some(false), Some(true)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_byteparts_compare_nullable() -> VortexResult<()> {
        let decimal_type = DecimalDType::new(19, -11);
        let lhs = DecimalByteParts::try_new(
            PrimitiveArray::new(
                buffer![1i64, 2i64, 3i64, 4i64],
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
            )
            .into_array(),
            decimal_type,
        )?;

        let rhs = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(289888198),
                decimal_type,
                Nullability::NonNullable,
            ),
            4,
        )
        .into_array();

        let res = lhs.into_array().binary(rhs, Operator::Lte)?;
        let expected =
            BoolArray::from_iter([None, Some(true), Some(true), Some(true)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn compare_decimal_const_with_lower_parts() -> VortexResult<()> {
        // The MSP-only pushdown is invalid once lower parts carry part of the value, so this
        // must fall back to the canonical comparison rather than compare MSPs.
        let values = vec![1i128 << 70, (1i128 << 70) + 1, 5, -(1i128 << 70)];
        let lhs = i128_parts(values.clone(), Validity::NonNullable).into_array();
        let decimal_dtype = *lhs
            .dtype()
            .as_decimal_opt()
            .vortex_expect("decimal byte parts array");

        let pivot = (1i128 << 70) + 1;
        let rhs = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(pivot),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            lhs.len(),
        )
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        for (operator, predicate) in [
            (Operator::Eq, (|v, p| v == p) as fn(i128, i128) -> bool),
            (Operator::NotEq, |v, p| v != p),
            (Operator::Lt, |v, p| v < p),
            (Operator::Lte, |v, p| v <= p),
            (Operator::Gt, |v, p| v > p),
            (Operator::Gte, |v, p| v >= p),
        ] {
            let res = lhs.clone().binary(rhs.clone(), operator)?;
            let expected =
                BoolArray::from_iter(values.iter().map(|v| predicate(*v, pivot))).into_array();
            assert_arrays_eq!(res, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn compare_decimal_const_unconvertible_comparison() {
        let decimal_dtype = DecimalDType::new(40, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let lhs = DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200i32, 400i32], Validity::AllValid).into_array(),
            decimal_dtype,
        )
        .unwrap()
        .into_array();
        // This cannot be converted to a i32.
        let rhs = ConstantArray::new(
            Scalar::try_new(
                dtype.clone(),
                Some(DecimalValue::I128(-9999999999999965304).into()),
            )
            .unwrap(),
            lhs.len(),
        );

        let res = lhs.binary(rhs.clone().into_array(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());

        let res = lhs.binary(rhs.clone().into_array(), Operator::Gt).unwrap();
        let expected = BoolArray::from_iter([Some(true), Some(true), Some(true)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());

        let res = lhs.binary(rhs.into_array(), Operator::Lt).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());

        // This cannot be converted to a i32.
        let rhs = ConstantArray::new(
            Scalar::try_new(dtype, Some(DecimalValue::I128(9999999999999965304).into())).unwrap(),
            lhs.len(),
        );

        let res = lhs.binary(rhs.clone().into_array(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());

        let res = lhs.binary(rhs.clone().into_array(), Operator::Gt).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(false)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());

        let res = lhs.binary(rhs.into_array(), Operator::Lt).unwrap();
        let expected = BoolArray::from_iter([Some(true), Some(true), Some(true)]).into_array();
        assert_arrays_eq!(res, expected, &mut SESSION.create_execution_ctx());
    }

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// Asserts the kernel engages for every operator and matches the comparison over the
    /// decimal array the parts were split from.
    fn assert_matches_decimal(
        array: &DecimalBytePartsArray,
        baseline: &DecimalArray,
        constants: impl IntoIterator<Item = DecimalValue>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        for constant in constants {
            let rhs = ConstantArray::new(
                Scalar::decimal(constant, baseline.decimal_dtype(), Nullability::NonNullable),
                array.len(),
            )
            .into_array();
            for operator in ALL_OPERATORS {
                let actual = <DecimalByteParts as CompareKernel>::compare(
                    array.as_view(),
                    &rhs,
                    operator,
                    &mut ctx,
                )?
                .unwrap_or_else(|| {
                    panic!("decimal byte parts compare must engage for {operator:?}")
                });
                let expected = baseline
                    .clone()
                    .into_array()
                    .binary(rhs.clone(), Operator::from(operator))?;
                assert_arrays_eq!(actual, expected, &mut ctx);
            }
        }
        Ok(())
    }

    fn i128_cases() -> (Vec<i128>, Vec<DecimalValue>) {
        // Precision 38 admits magnitudes below 10^38.
        let limit = 10i128.pow(38) - 1;
        let values = vec![
            1i128 << 70,
            (1i128 << 70) + 1,
            5,
            -(1i128 << 70),
            -1,
            0,
            1 << 64,
            limit,
            -limit,
        ];
        let mut constants = Vec::new();
        for value in &values {
            constants.extend([value - 1, *value, value + 1]);
        }
        constants.extend([(1i128 << 64) - 1, -(1i128 << 64), u64::MAX.into()]);
        (
            values,
            constants
                .into_iter()
                .filter(|constant| constant.abs() <= limit)
                .map(DecimalValue::I128)
                .collect(),
        )
    }

    #[test]
    fn wide_i128_matches_decimal() -> VortexResult<()> {
        let (values, constants) = i128_cases();
        let array = i128_parts(values.clone(), Validity::NonNullable);
        let baseline = DecimalArray::new(
            Buffer::from(values),
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        );
        assert_matches_decimal(&array, &baseline, constants)
    }

    #[test]
    fn wide_i128_nullable_matches_decimal() -> VortexResult<()> {
        let (values, constants) = i128_cases();
        let validity = Validity::from_iter((0..values.len()).map(|i| i % 3 != 1));
        let array = i128_parts(values.clone(), validity.clone());
        let baseline = DecimalArray::new(Buffer::from(values), DecimalDType::new(38, 2), validity);
        assert_matches_decimal(&array, &baseline, constants)
    }

    #[test]
    fn wide_i256_matches_decimal() -> VortexResult<()> {
        // Precision 76 admits magnitudes below 10^76, which is above 2^252.
        let values = vec![
            i256::from_parts(0, 1),
            i256::from_parts(0, -1),
            i256::from_parts(u128::MAX, 0),
            i256::from_parts(7, 1 << 64),
            i256::from_parts(0, 0),
            i256::from_parts(u128::MAX, -1),
            i256::from_parts(u128::MAX, (1 << 120) - 1),
            i256::from_parts(0, -(1 << 120)),
        ];
        let mut constants: Vec<i256> = values.clone();
        constants.extend([
            i256::from_parts(1, 1),
            i256::from_parts(u128::MAX - 1, 0),
            i256::from_parts(8, 1 << 64),
            i256::from_parts(1, -1),
            i256::from_parts(u128::MAX, -2),
            i256::from_parts(1, -(1 << 120)),
        ]);
        let validity = Validity::from_iter((0..values.len()).map(|i| i % 4 != 2));
        let array = i256_parts(values.clone(), validity.clone());
        let baseline = DecimalArray::new(Buffer::from(values), DecimalDType::new(76, 2), validity);
        assert_matches_decimal(
            &array,
            &baseline,
            constants.into_iter().map(DecimalValue::I256),
        )
    }

    /// Narrow parts cannot hold every constant of their precision: a constant above the parts'
    /// range answers by its sign, and a word above a lower part's range sits above every value
    /// in that part.
    #[test]
    fn narrow_parts_match_decimal() -> VortexResult<()> {
        let dtype = DecimalDType::new(38, 2);
        let validity = Validity::from_iter([true, false, true, true]);
        let msp = PrimitiveArray::new(buffer![1i16, 0, -1, 3], validity.clone()).into_array();
        let lower = buffer![5u16, 7, 9, u16::MAX].into_array();
        let array = DecimalByteParts::try_new_with_lower_parts(msp, vec![lower], dtype)?;

        let values: Vec<i128> = vec![
            (1i128 << 64) + 5,
            7,
            -(1i128 << 64) + 9,
            (3i128 << 64) + i128::from(u16::MAX),
        ];
        let baseline = DecimalArray::new(Buffer::from(values.clone()), dtype, validity);

        let mut constants = values;
        constants.extend([
            1i128 << 100,
            -(1i128 << 100),
            (1i128 << 64) + 70_000,
            (3i128 << 64) + 70_000,
            -(1i128 << 64) + 70_000,
            (1i128 << 64) + 4,
            (1i128 << 64) + 6,
            0,
            -1,
        ]);
        assert_matches_decimal(
            &array,
            &baseline,
            constants.into_iter().map(DecimalValue::I128),
        )
    }
}
