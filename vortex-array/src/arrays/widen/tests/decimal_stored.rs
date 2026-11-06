// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stored decimal conversion, buffer reuse, and range statistics.
//!
//! Compares converted values with canonical arrays and checks the selected child representation.

use std::ops::RangeInclusive;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::DecimalDataParts;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::i256;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Returns `values` as a buffer of `values_type`, with zero in each null slot.
fn stored_buffer(values: &[Option<i128>], values_type: DecimalType) -> ByteBuffer {
    match_each_decimal_value_type!(values_type, |T| {
        values
            .iter()
            .map(|value| {
                <T as BigCast>::from(value.unwrap_or_default())
                    .vortex_expect("fixture values fit the stored width")
            })
            .collect::<Buffer<T>>()
            .into_byte_buffer()
    })
}

/// Uses non-nullable validity when every value is present.
fn validity_of(values: &[Option<i128>]) -> Validity {
    if values.iter().all(Option::is_some) {
        return Validity::NonNullable;
    }

    Validity::from_iter(values.iter().map(Option::is_some))
}

/// Returns the parts of decimal `values` stored as `values_type`.
fn stored_parts(
    values: &[Option<i128>],
    values_type: DecimalType,
    decimal_dtype: DecimalDType,
) -> DecimalDataParts {
    DecimalDataParts {
        decimal_dtype,
        values: BufferHandle::new_host(stored_buffer(values, values_type)),
        values_type,
        validity: validity_of(values),
    }
}

/// Returns `values` as the canonical array of `decimal_dtype`.
fn canonical_reference(values: &[Option<i128>], decimal_dtype: DecimalDType) -> DecimalArray {
    let values_type = decimal_dtype.canonical_values_type();
    DecimalArray::new_handle(
        BufferHandle::new_host(stored_buffer(values, values_type)),
        values_type,
        decimal_dtype,
        validity_of(values),
    )
}

/// Returns the decimal storage, through a [`Widen`] if present.
fn decimal_leaf(array: &ArrayRef) -> DecimalArray {
    let leaf = match array.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => array.clone(),
    };

    leaf.downcast::<Decimal>()
}

fn i256_range(min: i128, max: i128) -> RangeInclusive<i256> {
    i256::from_i128(min)..=i256::from_i128(max)
}

/// Returns two `i32` zeros, with the alignment of `i32`.
fn two_i32_values() -> BufferHandle {
    BufferHandle::new_host(Buffer::<i32>::zeroed(2).into_byte_buffer())
}

/// Returns a validity array of `valid`. `Validity::from_iter` would turn an all-true input into
/// `Validity::AllValid`, which has no length to check.
fn validity_array(valid: &[bool]) -> Validity {
    Validity::Array(BoolArray::from_iter(valid.iter().copied()).into_array())
}

#[rstest]
#[case::canonical(
    vec![Some(-999_999_999), None, Some(999_999_999)],
    DecimalType::I32, DecimalDType::new(9, 2), None, true,
)]
#[case::stored_as_child(
    vec![Some(-9_999), Some(9_999)],
    DecimalType::I16, DecimalDType::new(18, 2), Some(4), true,
)]
#[case::narrowing_copy(
    vec![Some(-99), None, Some(99)],
    DecimalType::I32, DecimalDType::new(18, 2), Some(2), false,
)]
#[case::widening_copy(
    vec![Some(32_767)],
    DecimalType::I16, DecimalDType::new(10, 2), Some(5), false,
)]
#[case::no_narrower_child(
    vec![Some(1_000_000_000)],
    DecimalType::I32, DecimalDType::new(18, 2), None, false,
)]
#[case::scale_bound(
    vec![Some(1)],
    DecimalType::I8, DecimalDType::new(18, 10), None, false,
)]
#[case::negative_boundary(
    vec![Some(-128)],
    DecimalType::I8, DecimalDType::new(10, 0), Some(3), false,
)]
#[case::negative_scale(
    vec![Some(-99), Some(99)],
    DecimalType::I32, DecimalDType::new(18, -3), Some(2), false,
)]
#[case::i128_narrowing_copy(
    vec![Some(-99), Some(99)],
    DecimalType::I128, DecimalDType::new(18, 2), Some(2), false,
)]
#[case::i128_canonical_copy(
    vec![Some(100_000_000_000_000_000)],
    DecimalType::I128, DecimalDType::new(18, 2), None, false,
)]
#[case::i128_stored_as_child(
    vec![Some(10i128.pow(37))],
    DecimalType::I128, DecimalDType::new(40, 2), Some(38), true,
)]
#[case::i128_max_widens_to_i256(
    vec![Some(i128::MAX)],
    DecimalType::I128, DecimalDType::new(40, 0), None, false,
)]
#[case::i256_narrowing_copy(
    vec![Some(-99), None, Some(99)],
    DecimalType::I256, DecimalDType::new(18, 2), Some(2), false,
)]
#[case::i256_canonical_copy(
    vec![Some(10i128.pow(37))],
    DecimalType::I256, DecimalDType::new(38, 0), None, false,
)]
#[case::all_null(
    vec![None, None],
    DecimalType::I32, DecimalDType::new(18, 2), Some(2), false,
)]
#[case::empty(vec![], DecimalType::I32, DecimalDType::new(18, 2), Some(2), false)]
fn test_from_stored_decimal(
    #[case] values: Vec<Option<i128>>,
    #[case] values_type: DecimalType,
    #[case] decimal_dtype: DecimalDType,
    #[case] child_precision: Option<u8>,
    #[case] must_reuse_buffer: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = stored_parts(&values, values_type, decimal_dtype);
    let input_ptr = parts.values.as_host().as_ptr();

    let result = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)?;

    assert_eq!(result.is::<Widen>(), child_precision.is_some());
    let leaf = decimal_leaf(&result);
    let leaf_dtype = leaf.decimal_dtype();
    let expected_leaf_dtype = match child_precision {
        Some(precision) => DecimalDType::new(precision, decimal_dtype.scale()),
        None => decimal_dtype,
    };
    assert_eq!(leaf_dtype, expected_leaf_dtype);
    assert_eq!(leaf.values_type(), leaf_dtype.canonical_values_type());
    if must_reuse_buffer {
        assert_eq!(leaf.buffer_handle().as_host().as_ptr(), input_ptr);
    }
    assert_arrays_eq!(
        result,
        canonical_reference(&values, decimal_dtype),
        &mut ctx
    );

    Ok(())
}

/// Canonical storage is reused without checking values against the declared precision.
#[test]
fn test_from_stored_decimal_trusts_canonical_width() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = stored_parts(&[Some(100)], DecimalType::I8, DecimalDType::new(2, 0));
    let input_ptr = parts.values.as_host().as_ptr();

    let result = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)?;

    assert!(result.is::<Decimal>());
    let decimal = result.as_::<Decimal>();
    assert_eq!(decimal.buffer_handle().as_host().as_ptr(), input_ptr);
    assert_eq!(decimal.buffer::<i8>().as_slice(), [100]);

    Ok(())
}

#[rstest]
#[case::value_outside_precision(
    vec![Some(1), Some(100)], Precision::Absent, "Decimal range must fit",
)]
#[case::reversed_range(
    vec![Some(1), Some(2)], Precision::Inexact(i256_range(2, 1)), "ordered extrema",
)]
#[case::range_outside_precision(
    vec![Some(1), Some(2)], Precision::Inexact(i256_range(1, 100)), "Decimal range must fit",
)]
fn test_from_stored_decimal_rejects_invalid_range(
    #[case] values: Vec<Option<i128>>,
    #[case] range: Precision<RangeInclusive<i256>>,
    #[case] expected: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = stored_parts(&values, DecimalType::I16, DecimalDType::new(2, 0));

    let error = WidenArray::from_stored_decimal(parts, range, &mut ctx)
        .expect_err("an invalid value range must fail");
    assert!(error.to_string().contains(expected), "{error}");

    Ok(())
}

#[test]
fn test_from_stored_decimal_ignores_null_slots() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(9, 0);
    let validity = Validity::from_iter([false, true, false]);
    let stored = Buffer::copy_from([i128::MIN, 5, i128::MAX]);
    let parts = DecimalDataParts {
        decimal_dtype,
        values: BufferHandle::new_host(stored.into_byte_buffer()),
        values_type: DecimalType::I128,
        validity,
    };

    let result = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)?;

    let child = decimal_leaf(&result);
    assert_eq!(child.decimal_dtype(), DecimalDType::new(1, 0));
    assert_eq!(child.buffer::<i8>().as_slice(), [0, 5, 0]);
    assert_arrays_eq!(
        result,
        canonical_reference(&[None, Some(5), None], decimal_dtype),
        &mut ctx
    );

    Ok(())
}

/// Supplied ranges retain their exactness. An absent range records scanned extrema.
#[rstest]
#[case::scanned(
    vec![Some(-5), None, Some(7)], DecimalType::I128, Precision::Absent,
    2, Precision::Exact((-5, 7)),
)]
#[case::exact(
    vec![Some(-5), None, Some(7)], DecimalType::I32, Precision::Exact(i256_range(-5, 7)),
    2, Precision::Exact((-5, 7)),
)]
#[case::inexact(
    vec![Some(1), Some(2)], DecimalType::I32, Precision::Inexact(i256_range(-9_999, 9_999)), 4,
    Precision::Inexact((-9_999, 9_999)),
)]
fn test_from_stored_decimal_records_extrema(
    #[case] values: Vec<Option<i128>>,
    #[case] values_type: DecimalType,
    #[case] known_range: Precision<RangeInclusive<i256>>,
    #[case] child_precision: u8,
    #[case] expected: Precision<(i16, i16)>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(18, 2);
    let parts = stored_parts(&values, values_type, decimal_dtype);

    let result = WidenArray::from_stored_decimal(parts, known_range, &mut ctx)?;

    let child_dtype = DecimalDType::new(child_precision, 2);
    let child = decimal_leaf(&result);
    assert_eq!(child.decimal_dtype(), child_dtype);
    let statistics = child.statistics();
    let extremum = |value: i16| {
        let value = DecimalValue::try_from_i256(i256::from_i128(value.into()), child_dtype)
            .vortex_expect("expected extrema fit the child precision");
        Scalar::decimal(value, child_dtype, validity_of(&values).nullability())
    };
    assert_eq!(
        statistics.get(Stat::Min),
        expected.map(|(min, _)| extremum(min))
    );
    assert_eq!(
        statistics.get(Stat::Max),
        expected.map(|(_, max)| extremum(max))
    );
    assert_arrays_eq!(
        result,
        canonical_reference(&values, decimal_dtype),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::length_not_a_multiple_of_width(
    BufferHandle::new_host(ByteBuffer::copy_from([0u8; 5])),
    Validity::NonNullable,
    DecimalDType::new(9, 0),
    "not divisible",
)]
#[case::misaligned(
    BufferHandle::new_host(ByteBuffer::copy_from([0u8; 9]).slice_unaligned(1..9)),
    Validity::NonNullable,
    DecimalDType::new(9, 0),
    "alignment",
)]
#[case::validity_too_short(
    two_i32_values(),
    validity_array(&[true]),
    DecimalDType::new(9, 0),
    "one entry per value",
)]
#[case::validity_too_long(
    two_i32_values(),
    validity_array(&[true, false, true]),
    DecimalDType::new(18, 0),
    "one entry per value",
)]
fn test_from_stored_decimal_rejects_malformed_buffer(
    #[case] values: BufferHandle,
    #[case] validity: Validity,
    #[case] decimal_dtype: DecimalDType,
    #[case] expected: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = DecimalDataParts {
        decimal_dtype,
        values,
        values_type: DecimalType::I32,
        validity,
    };

    let error = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)
        .expect_err("malformed parts must fail");
    assert!(error.to_string().contains(expected), "{error}");

    Ok(())
}
