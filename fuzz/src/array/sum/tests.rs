// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_decimal_value_type;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::sum_canonical_array;
use crate::Action;
use crate::CompressorStrategy;
use crate::ExpectedValue;
use crate::FuzzArrayAction;
use crate::SESSION;
use crate::run_fuzz_action;

#[test]
fn test_sum_ignores_cached_statistics() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for (values, expected) in [
        ([1i64, 2, 3], Some(Scalar::from(6i64))),
        ([i64::MAX, -1, 1], None),
    ] {
        let array = PrimitiveArray::from_iter(values).into_array();
        array
            .statistics()
            .set(Stat::Sum, Precision::Exact(ScalarValue::from(999i64)));
        assert_eq!(sum_canonical_array(&array, &mut ctx)?, expected);
    }
    Ok(())
}

#[rstest]
#[case::i8(buffer![i8::MAX, i8::MAX, -1].into_array(), 253i64.into())]
#[case::i16(buffer![i16::MAX, i16::MAX, -1].into_array(), 65_533i64.into())]
#[case::i32(buffer![i32::MAX, i32::MAX, -1].into_array(), 4_294_967_293i64.into())]
#[case::i64(buffer![1i64, 2, 3].into_array(), 6i64.into())]
#[case::u8(buffer![u8::MAX, u8::MAX].into_array(), 510u64.into())]
#[case::u16(buffer![u16::MAX, u16::MAX].into_array(), 131_070u64.into())]
#[case::u32(buffer![u32::MAX, u32::MAX].into_array(), 8_589_934_590u64.into())]
#[case::u64(buffer![u64::MAX, 0].into_array(), u64::MAX.into())]
#[case::unsigned_overflow(buffer![u64::MAX, 1].into_array(), Scalar::from(None::<u64>))]
#[case::unsigned_nulls(
    PrimitiveArray::new(buffer![u64::MAX, u64::MAX], Validity::from_iter([true, false])).into_array(),
    u64::MAX.into()
)]
#[case::unsigned_all_null(PrimitiveArray::from_option_iter([None::<u64>, None]).into_array(), 0u64.into())]
#[case::unsigned_empty(Buffer::<u64>::empty().into_array(), 0u64.into())]
#[case::signed_empty(Buffer::<i32>::empty().into_array(), 0i64.into())]
#[case::bool(BoolArray::from_iter([true, false, true]).into_array(), 2u64.into())]
#[case::bool_nulls(BoolArray::from_iter([Some(true), None, Some(false)]).into_array(), 1u64.into())]
#[case::bool_all_null(BoolArray::from_iter([None::<bool>, None]).into_array(), 0u64.into())]
#[case::bool_empty(BoolArray::from_iter([] as [bool; 0]).into_array(), 0u64.into())]
fn test_sum_integer_and_bool_values(
    #[case] array: ArrayRef,
    #[case] expected: Scalar,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        sum_canonical_array(&array, &mut ctx)?,
        Some(expected.clone())
    );
    assert_eq!(sum(&array, &mut ctx)?, expected);
    Ok(())
}

#[rstest]
#[case::overflow_first([i64::MAX, 1, -1])]
#[case::cancellation_first([i64::MAX, -1, 1])]
#[case::negative_overflow([i64::MIN, -1, 1])]
#[case::negative_cancellation_first([i64::MIN, 1, -1])]
fn test_sum_rejects_ambiguous_native_overflow(#[case] values: [i64; 3]) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for batch_size in 1..=values.len() {
        let array = ChunkedArray::try_new(
            values
                .chunks(batch_size)
                .map(|values| PrimitiveArray::from_iter(values.iter().copied()).into_array()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .into_array();
        assert_eq!(sum_canonical_array(&array, &mut ctx)?, None);
        let canonical = array.execute::<Canonical>(&mut ctx)?.into_array();
        assert_eq!(sum_canonical_array(&canonical, &mut ctx)?, None);
    }
    Ok(())
}

#[rstest]
#[case::positive_overflow(vec![Some(i64::MAX), Some(1), Some(0)], None)]
#[case::negative_overflow(vec![Some(i64::MIN), Some(-1), Some(0)], None)]
#[case::native_extremes(vec![Some(i64::MAX), Some(i64::MIN)], Some(-1))]
#[case::nullable(vec![Some(i64::MAX), None, Some(-1)], Some(i64::MAX - 1))]
#[case::all_null(vec![None, None], Some(0))]
fn test_sum_signed_unambiguous(
    #[case] values: Vec<Option<i64>>,
    #[case] expected: Option<i64>,
) -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter(values).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let expected = Scalar::from(expected);
    assert_eq!(
        sum_canonical_array(&array, &mut ctx)?,
        Some(expected.clone())
    );
    assert_eq!(sum(&array, &mut ctx)?, expected);
    Ok(())
}

#[rstest]
#[case(DecimalType::I8)]
#[case(DecimalType::I16)]
#[case(DecimalType::I32)]
#[case(DecimalType::I64)]
#[case(DecimalType::I128)]
#[case(DecimalType::I256)]
fn test_sum_decimal_storage(#[case] values_type: DecimalType) -> VortexResult<()> {
    let array = match_each_decimal_value_type!(values_type, |D| {
        let value = DecimalValue::I8(99)
            .cast::<D>()
            .ok_or_else(|| vortex_err!("99 fits in every decimal storage type"))?;
        DecimalArray::new(
            buffer![value, value, -value],
            DecimalDType::new(2, 0),
            Validity::NonNullable,
        )
        .into_array()
    });
    let expected = Scalar::decimal(
        DecimalValue::I64(99),
        DecimalDType::new(12, 0),
        Nullability::Nullable,
    );
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        sum_canonical_array(&array, &mut ctx)?,
        Some(expected.clone())
    );
    assert_eq!(sum(&array, &mut ctx)?, expected);
    Ok(())
}

#[test]
fn test_sum_decimal_widens_beyond_i128() -> VortexResult<()> {
    let value = 10i128.pow(38) - 1;
    let array = DecimalArray::new(
        buffer![value, value],
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    )
    .into_array();
    assert_eq!(
        sum_canonical_array(&array, &mut SESSION.create_execution_ctx())?,
        Some(Scalar::decimal(
            DecimalValue::I256(i256::from_i128(value) * i256::from_i128(2)),
            DecimalDType::new(48, 2),
            Nullability::Nullable,
        ))
    );
    Ok(())
}

fn decimal_chunks(groups: &[&[i8]]) -> VortexResult<ArrayRef> {
    let unit = i256::from_i128(10).wrapping_pow(75);
    let dtype = DecimalDType::new(76, -76);
    ChunkedArray::try_new(
        groups.iter().map(|values| {
            DecimalArray::new(
                values
                    .iter()
                    .map(|&value| unit * i256::from_i128(i128::from(value)))
                    .collect::<Buffer<_>>(),
                dtype,
                Validity::NonNullable,
            )
            .into_array()
        }),
        DType::Decimal(dtype, Nullability::NonNullable),
    )
    .map(IntoArray::into_array)
}

#[rstest]
#[case::overflow_first(decimal_chunks(&[&[6, 6], &[-6]]))]
#[case::cancellation_first(decimal_chunks(&[&[6, -6], &[6]]))]
#[case::negative_overflow(decimal_chunks(&[&[-6, -6], &[6]]))]
#[case::negative_cancellation_first(decimal_chunks(&[&[-6, 6], &[-6]]))]
#[case::zero_total(decimal_chunks(&[&[6, 6], &[-6, -6]]))]
fn test_sum_rejects_ambiguous_decimal_precision(
    #[case] array: VortexResult<ArrayRef>,
) -> VortexResult<()> {
    let array = array?;
    let nested = ChunkedArray::try_new(vec![array.clone()], array.dtype().clone())?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = array.clone().execute::<Canonical>(&mut ctx)?.into_array();
    for array in [array, nested, canonical] {
        assert_eq!(sum_canonical_array(&array, &mut ctx)?, None);
    }
    Ok(())
}

#[rstest]
fn test_sum_decimal_native_overflow(#[values(false, true)] negative: bool) -> VortexResult<()> {
    let dtype = DecimalDType::new(76, 0);
    let value =
        i256::from_i128(10).wrapping_pow(75) * i256::from_i128(if negative { -6 } else { 6 });
    let mut ctx = SESSION.create_execution_ctx();
    for mixed in [false, true] {
        let values = iter::repeat_n(value, 10)
            .chain(iter::repeat_n(-value, if mixed { 10 } else { 0 }))
            .collect::<Buffer<_>>();
        let array = DecimalArray::new(values, dtype, Validity::NonNullable).into_array();
        let expected = (!mixed).then(|| Scalar::null(DType::Decimal(dtype, Nullability::Nullable)));
        assert_eq!(sum_canonical_array(&array, &mut ctx)?, expected);
    }
    Ok(())
}

#[rstest]
#[case::boundary(Validity::from_iter([true, false, false]), true)]
#[case::cancellation(Validity::from_iter([true, false, true]), true)]
#[case::ambiguous(Validity::AllValid, false)]
#[case::all_null(Validity::AllInvalid, true)]
fn test_sum_decimal_precision_boundary_and_nulls(
    #[case] validity: Validity,
    #[case] accepted: bool,
) -> VortexResult<()> {
    let dtype = DecimalDType::new(76, 0);
    let max = i256::from_i128(10).wrapping_pow(76) - i256::ONE;
    let array = DecimalArray::new(buffer![max, i256::ONE, -max], dtype, validity).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let result = sum_canonical_array(&array, &mut ctx)?;
    assert_eq!(result.is_some(), accepted);
    if let Some(expected) = result {
        assert_eq!(sum(&array, &mut ctx)?, expected);
    }
    Ok(())
}

#[rstest]
#[case::positive(6)]
#[case::negative(-6)]
fn test_sum_constant_decimal_definite_overflow(#[case] value: i128) -> VortexResult<()> {
    let dtype = DecimalDType::new(76, 0);
    let scalar = Scalar::decimal(
        DecimalValue::I256(i256::from_i128(10).wrapping_pow(75) * i256::from_i128(value)),
        dtype,
        Nullability::NonNullable,
    );
    let array = ConstantArray::new(scalar, 2).into_array();
    assert_eq!(
        sum_canonical_array(&array, &mut SESSION.create_execution_ctx())?,
        Some(Scalar::null(DType::Decimal(dtype, Nullability::Nullable)))
    );
    Ok(())
}

#[rstest]
#[case::safe_cancellation(decimal_chunks(&[&[4, 4], &[-4]]), Some(4))]
#[case::negative_cancellation(decimal_chunks(&[&[-4, -4], &[4]]), Some(-4))]
#[case::positive_overflow(decimal_chunks(&[&[6, 5]]), None)]
#[case::negative_overflow(decimal_chunks(&[&[-6, -5]]), None)]
#[case::empty(decimal_chunks(&[&[], &[]]), Some(0))]
fn test_sum_action_accepts_unambiguous_inputs(
    #[case] array: VortexResult<ArrayRef>,
    #[case] expected: Option<i8>,
    #[values(false, true)] compress: bool,
) -> VortexResult<()> {
    let array = array?;
    let expected = match expected {
        Some(value) => Scalar::decimal(
            DecimalValue::I256(
                i256::from_i128(10).wrapping_pow(75) * i256::from_i128(i128::from(value)),
            ),
            DecimalDType::new(76, -76),
            Nullability::Nullable,
        ),
        None => Scalar::null(array.dtype().as_nullable()),
    };
    assert_eq!(
        sum_canonical_array(&array, &mut SESSION.create_execution_ctx())?,
        Some(expected.clone())
    );
    let mut actions = Vec::new();
    if compress {
        actions.push((
            Action::Compress(CompressorStrategy::Default),
            ExpectedValue::Array(array.clone()),
        ));
    }
    actions.push((Action::Sum, ExpectedValue::Scalar(expected)));
    assert!(
        run_fuzz_action(FuzzArrayAction { array, actions })
            .map_err(|error| vortex_err!("{error}"))?
    );
    Ok(())
}

#[test]
fn test_sum_rejects_floats() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter([1.0f64, 2.0]).into_array();
    assert_eq!(
        sum_canonical_array(&array, &mut SESSION.create_execution_ctx())?,
        None
    );
    Ok(())
}
