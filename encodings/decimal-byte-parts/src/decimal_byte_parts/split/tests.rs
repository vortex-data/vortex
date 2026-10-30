// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lower-part selection and checked splitting of valid decimal values.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::lower_part_count;
use super::split_decimal;
use crate::decimal_byte_parts::DecimalByteParts;
use crate::decimal_byte_parts::LOWER_PART_DTYPE;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// 2^127, the smallest positive value that does not fit `i128`.
const I256_ABOVE_I128: i256 = i256::from_parts(1 << 127, 0);

/// A decimal of `values` at the largest precision of their storage width.
fn decimal<T: NativeDecimalType>(values: Buffer<T>, validity: Validity) -> DecimalArray {
    DecimalArray::new(values, DecimalDType::new(T::MAX_PRECISION, 0), validity)
}

#[rstest]
#[case::i8(decimal(buffer![i8::MIN, i8::MAX], Validity::NonNullable), 0)]
#[case::i16(decimal(buffer![i16::MIN, i16::MAX], Validity::NonNullable), 0)]
#[case::i32(decimal(buffer![i32::MIN, i32::MAX], Validity::NonNullable), 0)]
#[case::i64(decimal(buffer![i64::MIN, i64::MAX], Validity::NonNullable), 0)]
#[case::i128_fits_i64(
    decimal(buffer![i128::from(i64::MIN), i128::from(i64::MAX)], Validity::NonNullable),
    0
)]
#[case::i128_below_i64_min(
    decimal(buffer![i128::from(i64::MIN) - 1, 0], Validity::NonNullable),
    1
)]
#[case::i128_above_i64_max(
    decimal(buffer![0, i128::from(i64::MAX) + 1], Validity::NonNullable),
    1
)]
#[case::i128_full_range(decimal(buffer![i128::MIN, i128::MAX], Validity::NonNullable), 1)]
#[case::i256_fits_i64(
    decimal(buffer![i256::from_i128(i128::from(i64::MIN))], Validity::NonNullable),
    0
)]
#[case::i256_below_i64_min(
    decimal(buffer![i256::from_i128(i128::from(i64::MIN) - 1)], Validity::NonNullable),
    1
)]
#[case::i256_minus_2_pow_127(
    decimal(buffer![i256::from_i128(i128::MIN)], Validity::NonNullable),
    1
)]
#[case::i256_below_minus_2_pow_127(
    decimal(buffer![i256::from_i128(i128::MIN) - i256::ONE], Validity::NonNullable),
    3
)]
#[case::i256_plus_2_pow_127(decimal(buffer![I256_ABOVE_I128], Validity::NonNullable), 3)]
#[case::null_slot_holds_2_pow_100(
    decimal(buffer![1i128, 1 << 100], Validity::from_iter([true, false])),
    0
)]
#[case::all_null(decimal(buffer![1i128 << 100], Validity::AllInvalid), 0)]
#[case::empty(decimal(Buffer::<i256>::empty(), Validity::NonNullable), 0)]
fn test_lower_part_count(
    #[case] decimal: DecimalArray,
    #[case] expected: usize,
) -> VortexResult<()> {
    let count = lower_part_count(&decimal, &mut SESSION.create_execution_ctx())?;
    assert_eq!(count, expected);
    Ok(())
}

/// The count reads only the stored values, so a value outside the precision neither panics nor
/// lowers the count.
#[test]
fn test_lower_part_count_ignores_precision() -> VortexResult<()> {
    let decimal = DecimalArray::new(
        buffer![1i128 << 70],
        DecimalDType::new(2, 0),
        Validity::NonNullable,
    );
    let count = lower_part_count(&decimal, &mut SESSION.create_execution_ctx())?;
    assert_eq!(count, 1);
    Ok(())
}

/// The copy into the narrower type checks every valid value, so a count below the need of a
/// valid value fails.
#[rstest]
#[case::i128_into_msp(decimal(buffer![0i128, 1 << 70], Validity::NonNullable), 0)]
#[case::i256_into_msp(decimal(buffer![i256::ZERO, I256_ABOVE_I128], Validity::NonNullable), 0)]
#[case::i256_into_one_lower_part(
    decimal(buffer![i256::ZERO, I256_ABOVE_I128], Validity::NonNullable),
    1
)]
fn test_split_decimal_rejects_too_small_count(
    #[case] decimal: DecimalArray,
    #[case] lower_part_count: usize,
) {
    let result = split_decimal(
        &decimal,
        lower_part_count,
        &mut SESSION.create_execution_ctx(),
    );
    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains("does not fit")),
        "expected the narrowing copy to fail"
    );
}

/// Only 0 lower parts, or the count of `i128` or `i256` up to the storage width, are supported.
/// The count is checked before an array without valid values takes its shortcut.
#[rstest]
#[case::i8_one(decimal(buffer![0i8], Validity::NonNullable), 1)]
#[case::i64_one(decimal(buffer![0i64], Validity::NonNullable), 1)]
#[case::i128_two(decimal(buffer![0i128], Validity::NonNullable), 2)]
#[case::i128_three(decimal(buffer![0i128], Validity::NonNullable), 3)]
#[case::i256_two(decimal(buffer![i256::ZERO], Validity::NonNullable), 2)]
#[case::i256_four(decimal(buffer![i256::ZERO], Validity::NonNullable), 4)]
#[case::all_null(decimal(buffer![0i128], Validity::AllInvalid), 3)]
#[case::empty(decimal(Buffer::<i64>::empty(), Validity::NonNullable), 1)]
fn test_split_decimal_rejects_invalid_count(
    #[case] decimal: DecimalArray,
    #[case] lower_part_count: usize,
) {
    let result = split_decimal(
        &decimal,
        lower_part_count,
        &mut SESSION.create_execution_ctx(),
    );
    let expected = format!("lower parts, got {lower_part_count}");
    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains(&expected)),
        "expected the shape check to reject {lower_part_count} lower parts"
    );
}

/// An array without valid values splits into constant parts with the requested count, not the
/// count of its storage width.
#[rstest]
#[case::i128_all_null_msp_only(decimal(buffer![1i128 << 100; 3], Validity::AllInvalid), 0)]
#[case::i128_empty_msp_only(decimal(Buffer::<i128>::empty(), Validity::NonNullable), 0)]
#[case::i256_all_null_msp_only(decimal(buffer![I256_ABOVE_I128; 3], Validity::AllInvalid), 0)]
#[case::i256_empty_msp_only(decimal(Buffer::<i256>::empty(), Validity::NonNullable), 0)]
#[case::i256_all_null_one_lower_part(
    decimal(buffer![I256_ABOVE_I128; 3], Validity::AllInvalid),
    1
)]
#[case::i256_empty_one_lower_part(decimal(Buffer::<i256>::empty(), Validity::NonNullable), 1)]
fn test_split_without_valid_rows_uses_requested_count(
    #[case] decimal: DecimalArray,
    #[case] lower_part_count: usize,
) -> VortexResult<()> {
    let parts = split_decimal(
        &decimal,
        lower_part_count,
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(parts.msp.is::<Constant>());
    assert_eq!(parts.msp.dtype().as_ptype(), PType::I64);
    assert_eq!(
        parts.msp.dtype().nullability(),
        decimal.dtype().nullability()
    );
    assert_eq!(parts.lower_parts.len(), lower_part_count);
    assert!(parts.lower_parts.iter().all(|part| part.is::<Constant>()));
    Ok(())
}

/// `i256` values that fit `i128` split into an `i64` MSP and one lower part. The null slot
/// holds a value outside `i128`, which the narrowing copy ignores.
#[test]
fn test_split_i256_into_one_lower_part_round_trips() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalArray::new(
        buffer![
            i256::from_i128(i128::MIN),
            I256_ABOVE_I128,
            i256::from_i128(-1),
            i256::from_i128((1 << 64) | 5),
            i256::from_i128(i128::MAX),
        ],
        DecimalDType::new(76, 0),
        Validity::from_iter([true, false, true, true, true]),
    );
    let parts = split_decimal(&decimal, 1, &mut ctx)?;
    assert_arrays_eq!(
        parts.msp,
        PrimitiveArray::new(
            buffer![i64::MIN, 0, -1, 1, i64::MAX],
            Validity::from_iter([true, false, true, true, true]),
        ),
        &mut ctx
    );
    assert_eq!(parts.lower_parts.len(), 1);
    assert_arrays_eq!(
        parts.lower_parts[0],
        PrimitiveArray::new(
            buffer![0u64, 0, u64::MAX, 5, u64::MAX],
            Validity::NonNullable
        ),
        &mut ctx
    );

    let encoded = DecimalByteParts::try_new_with_lower_parts(
        parts.msp,
        parts.lower_parts,
        decimal.decimal_dtype(),
    )?;
    assert_arrays_eq!(encoded, decimal, &mut ctx);
    Ok(())
}

/// Wide values that fit `i64` split into an `i64` MSP without lower parts. The null slot holds
/// a value outside `i64`, which the narrowing copy ignores.
#[rstest]
#[case::i128(decimal(
    buffer![i128::from(i64::MIN), 1 << 100, 0, i128::from(i64::MAX)],
    Validity::from_iter([true, false, true, true]),
))]
#[case::i256(decimal(
    buffer![
        i256::from_i128(i128::from(i64::MIN)),
        I256_ABOVE_I128,
        i256::ZERO,
        i256::from_i128(i128::from(i64::MAX)),
    ],
    Validity::from_iter([true, false, true, true]),
))]
fn test_split_wide_into_msp_only(#[case] decimal: DecimalArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = split_decimal(&decimal, 0, &mut ctx)?;
    assert!(parts.lower_parts.is_empty());
    assert_arrays_eq!(
        parts.msp,
        PrimitiveArray::new(
            buffer![i64::MIN, 0, 0, i64::MAX],
            Validity::from_iter([true, false, true, true]),
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_split_i256_part_count_and_types() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalArray::new(
        Buffer::from(vec![i256::from_i128(i128::MAX), i256::MIN]),
        DecimalDType::new(76, 0),
        Validity::NonNullable,
    );
    let parts = split_decimal(&decimal, MAX_LOWER_PARTS, &mut ctx)?;
    assert_eq!(parts.lower_parts.len(), MAX_LOWER_PARTS);
    assert_eq!(parts.msp.dtype().as_ptype(), PType::I64);
    for part in &parts.lower_parts {
        assert_eq!(part.dtype(), &LOWER_PART_DTYPE);
    }
    Ok(())
}

#[rstest]
fn test_split_i256_part_order(
    #[values(Validity::NonNullable, Validity::from_iter([true, false, true]))] validity: Validity,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalArray::new(
        buffer![
            i256::from_parts((2u128 << 64) | 3, (1i128 << 64) | 4),
            i256::ZERO,
            i256::from_parts((6u128 << 64) | 7, (-2i128 << 64) | 5),
        ],
        DecimalDType::new(76, 0),
        validity.clone(),
    );
    let parts = split_decimal(&decimal, MAX_LOWER_PARTS, &mut ctx)?;
    assert_arrays_eq!(
        parts.msp,
        PrimitiveArray::new(buffer![1i64, 0, -2], validity),
        &mut ctx
    );
    assert_eq!(parts.lower_parts.len(), 3);
    for (part, expected) in parts.lower_parts.into_iter().zip([
        buffer![4u64, 0, 5],
        buffer![2u64, 0, 6],
        buffer![3u64, 0, 7],
    ]) {
        assert_arrays_eq!(
            part,
            PrimitiveArray::new(expected, Validity::NonNullable),
            &mut ctx
        );
    }
    Ok(())
}
