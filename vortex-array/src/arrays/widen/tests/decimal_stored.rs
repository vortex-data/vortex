// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads of decimal values stored at any width with [`WidenArray::from_stored_decimal`].
//!
//! Each fixture lists its values as `i128`, the stored width and the decimal dtype. A `None`
//! value is a null slot that holds zero, and a fixture without `None` values is non-nullable.

use std::ops::RangeInclusive;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::stored_decimal_dtype;
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
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
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

/// Returns the validity of `values`: non-nullable without a `None` value, and a validity array
/// otherwise.
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

/// Returns the decimal leaf that holds the values of `array`: the child of a `Widen`, or `array`.
fn decimal_leaf(array: &ArrayRef) -> ArrayRef {
    match array.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => array.clone(),
    }
}

/// Returns the address of the host buffer of the decimal leaf of `array`.
fn leaf_buffer_ptr(array: &ArrayRef) -> *const u8 {
    decimal_leaf(array)
        .as_::<Decimal>()
        .buffer_handle()
        .as_host()
        .as_ptr()
}

fn i256_range(min: i128, max: i128) -> RangeInclusive<i256> {
    i256::from_i128(min)..=i256::from_i128(max)
}

#[rstest]
#[case::canonical(
    vec![Some(-999_999_999), None, Some(999_999_999)],
    DecimalType::I32,
    DecimalDType::new(9, 2),
    None,
    true,
)]
#[case::stored_as_child(
    vec![Some(-9_999), Some(9_999)],
    DecimalType::I16,
    DecimalDType::new(18, 2),
    Some(4),
    true,
)]
#[case::narrowing_copy(
    vec![Some(-99), None, Some(99)],
    DecimalType::I32,
    DecimalDType::new(18, 2),
    Some(2),
    false,
)]
#[case::widening_copy(
    vec![Some(32_767)],
    DecimalType::I16,
    DecimalDType::new(10, 2),
    Some(5),
    false,
)]
#[case::no_narrower_child(
    vec![Some(1_000_000_000)],
    DecimalType::I32,
    DecimalDType::new(18, 2),
    None,
    false,
)]
#[case::scale_bound(
    vec![Some(1)],
    DecimalType::I8,
    DecimalDType::new(18, 10),
    None,
    false,
)]
#[case::negative_boundary(
    vec![Some(-128)],
    DecimalType::I8,
    DecimalDType::new(10, 0),
    Some(3),
    false,
)]
#[case::i128_narrowing_copy(
    vec![Some(-99), Some(99)],
    DecimalType::I128,
    DecimalDType::new(18, 2),
    Some(2),
    false,
)]
#[case::i128_canonical_copy(
    vec![Some(100_000_000_000_000_000)],
    DecimalType::I128,
    DecimalDType::new(18, 2),
    None,
    false,
)]
#[case::i128_stored_as_child(
    vec![Some(10i128.pow(37))],
    DecimalType::I128,
    DecimalDType::new(40, 2),
    Some(38),
    true,
)]
#[case::i128_max_widens_to_i256(
    vec![Some(i128::MAX)],
    DecimalType::I128,
    DecimalDType::new(40, 0),
    None,
    false,
)]
#[case::i256_narrowing_copy(
    vec![Some(-99), None, Some(99)],
    DecimalType::I256,
    DecimalDType::new(18, 2),
    Some(2),
    false,
)]
#[case::i256_canonical_copy(
    vec![Some(10i128.pow(37))],
    DecimalType::I256,
    DecimalDType::new(38, 0),
    None,
    false,
)]
#[case::all_null(
    vec![None, None],
    DecimalType::I32,
    DecimalDType::new(18, 2),
    Some(2),
    false,
)]
#[case::empty(vec![], DecimalType::I32, DecimalDType::new(18, 2), Some(2), false)]
fn test_from_stored_decimal(
    #[case] values: Vec<Option<i128>>,
    #[case] values_type: DecimalType,
    #[case] decimal_dtype: DecimalDType,
    #[case] child_precision: Option<u8>,
    #[case] reuses_buffer: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = stored_parts(&values, values_type, decimal_dtype);
    let input_ptr = parts.values.as_host().as_ptr();

    let result = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)?;

    assert_eq!(result.is::<Widen>(), child_precision.is_some());
    let leaf_dtype = stored_decimal_dtype(&result);
    let expected_leaf_dtype = match child_precision {
        Some(precision) => DecimalDType::new(precision, decimal_dtype.scale()),
        None => decimal_dtype,
    };
    assert_eq!(leaf_dtype, expected_leaf_dtype);
    assert_eq!(
        decimal_leaf(&result).as_::<Decimal>().values_type(),
        leaf_dtype.canonical_values_type()
    );
    if reuses_buffer {
        assert_eq!(leaf_buffer_ptr(&result), input_ptr);
    }
    assert_arrays_eq!(
        result,
        canonical_reference(&values, decimal_dtype),
        &mut ctx
    );

    Ok(())
}

/// A canonical stored width is trusted, so a valid value outside the precision is no error.
#[test]
fn test_from_stored_decimal_trusts_canonical_width() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = stored_parts(&[Some(100)], DecimalType::I8, DecimalDType::new(2, 0));
    let input_ptr = parts.values.as_host().as_ptr();

    let result = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)?;

    assert!(result.is::<Decimal>());
    assert_eq!(leaf_buffer_ptr(&result), input_ptr);
    assert_eq!(result.as_::<Decimal>().buffer::<i8>().as_slice(), [100]);

    Ok(())
}

#[rstest]
#[case::i16_at_p2(DecimalType::I16, DecimalDType::new(2, 0))]
#[case::i64_at_p2(DecimalType::I64, DecimalDType::new(2, 0))]
#[case::i128_at_p18(DecimalType::I128, DecimalDType::new(18, 0))]
#[case::i256_at_p38(DecimalType::I256, DecimalDType::new(38, 0))]
fn test_from_stored_decimal_wider_rejects_value_outside_precision(
    #[case] values_type: DecimalType,
    #[case] decimal_dtype: DecimalDType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let outside = 10i128.pow(decimal_dtype.precision().into());
    let parts = stored_parts(&[Some(1), Some(outside)], values_type, decimal_dtype);

    let error = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)
        .expect_err("a valid value outside the precision must fail");
    assert!(error.to_string().contains(&outside.to_string()), "{error}");

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

    assert_eq!(stored_decimal_dtype(&result), DecimalDType::new(1, 0));
    let child = decimal_leaf(&result);
    assert_eq!(child.as_::<Decimal>().buffer::<i8>().as_slice(), [0, 5, 0]);
    assert_arrays_eq!(
        result,
        canonical_reference(&[None, Some(5), None], decimal_dtype),
        &mut ctx
    );

    Ok(())
}

/// The child takes the precision of an inexact known range, not of the values, and keeps its
/// bounds as inexact extrema.
#[test]
fn test_from_stored_decimal_uses_known_range() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(18, 2);
    let values = [Some(1), Some(2)];
    let parts = stored_parts(&values, DecimalType::I32, decimal_dtype);
    let known_range = i256_range(-9_999, 9_999);

    let result = WidenArray::from_stored_decimal(parts, Precision::Inexact(known_range), &mut ctx)?;

    let child_dtype = DecimalDType::new(4, 2);
    assert_eq!(stored_decimal_dtype(&result), child_dtype);
    let child = decimal_leaf(&result);
    let statistics = child.statistics();
    let bound = |value: i16| Scalar::decimal(DecimalValue::I16(value), child_dtype, NonNullable);
    assert_eq!(statistics.get(Stat::Min), Precision::Inexact(bound(-9_999)));
    assert_eq!(statistics.get(Stat::Max), Precision::Inexact(bound(9_999)));
    assert_arrays_eq!(
        result,
        canonical_reference(&values, decimal_dtype),
        &mut ctx
    );

    Ok(())
}

#[test]
fn test_from_stored_decimal_records_extrema() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = stored_parts(
        &[Some(-5), None, Some(7)],
        DecimalType::I128,
        DecimalDType::new(18, 2),
    );

    let result = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)?;

    let child_dtype = DecimalDType::new(2, 2);
    assert_eq!(stored_decimal_dtype(&result), child_dtype);
    let child = decimal_leaf(&result);
    let statistics = child.statistics();
    let extremum = |value: i8| Scalar::decimal(DecimalValue::I8(value), child_dtype, Nullable);
    assert_eq!(statistics.get(Stat::Min), Precision::Exact(extremum(-5)));
    assert_eq!(statistics.get(Stat::Max), Precision::Exact(extremum(7)));

    Ok(())
}

/// An exact known range becomes the exact extrema of the child, so that `min_max` reads them
/// without a scan.
#[test]
fn test_from_stored_decimal_records_exact_known_range() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(18, 2);
    let values = [Some(-5), None, Some(7)];
    let parts = stored_parts(&values, DecimalType::I32, decimal_dtype);
    let known_range = Precision::Exact(i256_range(-5, 7));

    let result = WidenArray::from_stored_decimal(parts, known_range, &mut ctx)?;

    let child_dtype = DecimalDType::new(2, 2);
    assert_eq!(stored_decimal_dtype(&result), child_dtype);
    let child = decimal_leaf(&result);
    let statistics = child.statistics();
    let extremum = |value: i8| Scalar::decimal(DecimalValue::I8(value), child_dtype, Nullable);
    assert_eq!(statistics.get(Stat::Min), Precision::Exact(extremum(-5)));
    assert_eq!(statistics.get(Stat::Max), Precision::Exact(extremum(7)));
    assert_arrays_eq!(
        result,
        canonical_reference(&values, decimal_dtype),
        &mut ctx
    );

    Ok(())
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

/// Each case fails at its own check, so each check is tested alone.
#[rstest]
#[case::length_not_a_multiple_of_width(
    BufferHandle::new_host(ByteBuffer::copy_from([0u8; 5])),
    Validity::NonNullable,
    "not divisible",
)]
#[case::misaligned(
    BufferHandle::new_host(ByteBuffer::copy_from([0u8; 9]).slice_unaligned(1..9)),
    Validity::NonNullable,
    "alignment",
)]
#[case::validity_too_short(
    two_i32_values(),
    validity_array(&[true]),
    "one entry per value",
)]
#[case::validity_too_long(
    two_i32_values(),
    validity_array(&[true, false, true]),
    "one entry per value",
)]
fn test_from_stored_decimal_rejects_malformed_buffer(
    #[case] values: BufferHandle,
    #[case] validity: Validity,
    #[case] expected: &str,
    #[values(DecimalDType::new(9, 0), DecimalDType::new(18, 0))] decimal_dtype: DecimalDType,
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

/// Both scan the values when the stored width is not canonical, so they pick the same child.
/// They differ only when no child is narrower: `encode_decimal` then returns its input, and
/// `from_stored_decimal` returns the canonical array.
#[rstest]
#[case::narrower_child(vec![Some(-99), None, Some(99)], DecimalType::I32)]
#[case::stored_as_child(vec![Some(-9_999), Some(9_999)], DecimalType::I16)]
#[case::wider_stored(vec![Some(-99), Some(99)], DecimalType::I128)]
#[case::i256_stored(vec![Some(-99), Some(99)], DecimalType::I256)]
#[case::all_null(vec![None, None], DecimalType::I128)]
fn test_encode_decimal_matches_from_stored_decimal(
    #[case] values: Vec<Option<i128>>,
    #[case] values_type: DecimalType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(18, 2);
    let parts = stored_parts(&values, values_type, decimal_dtype);
    // SAFETY: the fixture values fit the precision, and the validity has one entry per value.
    let input = unsafe {
        DecimalArray::new_unchecked_handle(
            parts.values.clone(),
            values_type,
            decimal_dtype,
            parts.validity.clone(),
        )
    };

    let encoded = WidenArray::encode_decimal(input, &mut ctx)?;
    let stored = WidenArray::from_stored_decimal(parts, Precision::Absent, &mut ctx)?;

    assert_eq!(encoded.is::<Widen>(), stored.is::<Widen>());
    assert_eq!(
        stored_decimal_dtype(&encoded),
        stored_decimal_dtype(&stored)
    );
    assert_eq!(
        decimal_leaf(&encoded).as_::<Decimal>().values_type(),
        decimal_leaf(&stored).as_::<Decimal>().values_type()
    );
    assert_eq!(
        decimal_leaf(&encoded).statistics().get(Stat::Min),
        decimal_leaf(&stored).statistics().get(Stat::Min)
    );
    assert_arrays_eq!(encoded, stored, &mut ctx);

    Ok(())
}
