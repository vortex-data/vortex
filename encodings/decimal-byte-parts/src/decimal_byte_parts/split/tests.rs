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
use vortex_array::dtype;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_decimal_value_type;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::lower_part_count;
use super::split_decimal;
use crate::decimal_byte_parts::DecimalByteParts;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;
use crate::decimal_byte_parts::testing::validity_mask;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// 2^127, the smallest positive value that does not fit `i128`.
const I256_ABOVE_I128: i256 = i256::from_parts(1 << 127, 0);

/// A decimal of `values` at the largest precision of their storage width.
fn decimal<T: NativeDecimalType>(values: Buffer<T>, validity: Validity) -> DecimalArray {
    DecimalArray::new(values, DecimalDType::new(T::MAX_PRECISION, 0), validity)
}

/// Runs splitting through validation and checks the reported requirement.
#[track_caller]
fn assert_split_error(decimal: &DecimalArray, lower_part_count: usize, expected: &str) {
    let mask = validity_mask(decimal).vortex_expect("the fixture validity executes");
    let error = match split_decimal(decimal, &mask, lower_part_count) {
        Ok(_) => panic!("the fixture must violate a split requirement"),
        Err(error) => error,
    };
    assert!(error.to_string().contains(expected), "{error}");
}

#[rstest]
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
    let count = lower_part_count(&decimal, &validity_mask(&decimal)?)?;
    assert_eq!(count, expected);

    Ok(())
}

/// Both functions reject a validity mask that does not have one bit per value.
#[test]
fn test_rejects_mask_of_other_length() {
    let decimal = decimal(buffer![1i128, 1 << 100], Validity::NonNullable);
    let mask = Mask::new_true(3);
    let expected_error = "validity mask length\n  left: 3\n right: 2";

    let count = lower_part_count(&decimal, &mask);
    assert!(
        count
            .as_ref()
            .is_err_and(|err| err.to_string().contains(expected_error)),
        "expected the mask length check to reject the count, got {count:?}"
    );

    let parts = split_decimal(&decimal, &mask, 1);
    assert!(
        parts
            .as_ref()
            .is_err_and(|err| err.to_string().contains(expected_error)),
        "expected the mask length check to reject the split, got {:?}",
        parts.as_ref().err()
    );
}

/// The checked copy into the MSP or the `i128` values rejects a valid value that does not fit.
#[rstest]
#[case::i128_into_msp(decimal(buffer![0i128, 1 << 70], Validity::NonNullable), 0)]
#[case::i256_into_one_lower_part(
    decimal(buffer![i256::ZERO, I256_ABOVE_I128], Validity::NonNullable),
    1
)]
#[case::legacy_wider_i64_at_p4(
    DecimalArray::new(buffer![0i64, i64::MAX], DecimalDType::new(4, 0), Validity::NonNullable),
    0
)]
fn test_split_decimal_rejects_too_small_count(
    #[case] decimal: DecimalArray,
    #[case] lower_part_count: usize,
) -> VortexResult<()> {
    assert_split_error(&decimal, lower_part_count, "does not fit");

    Ok(())
}

/// The lower-part count must be at most the bound of the precision, and the storage width and the
/// count must be one of the shapes that [`split_decimal`] supports. Both checks run before an array
/// without valid values takes its shortcut.
#[rstest]
#[case::i64_one(
    decimal(buffer![0i64], Validity::NonNullable),
    1,
    "allows at most 0 lower parts, got 1"
)]
#[case::i128_two(
    decimal(buffer![0i128], Validity::NonNullable),
    2,
    "allows at most 1 lower parts, got 2"
)]
#[case::i256_two(
    decimal(buffer![i256::ZERO], Validity::NonNullable),
    2,
    "i256 values split into 0, 1, or 3 lower parts, got 2"
)]
#[case::i256_four(
    decimal(buffer![i256::ZERO], Validity::NonNullable),
    4,
    "allows at most 3 lower parts, got 4"
)]
#[case::legacy_wider_storage(
    DecimalArray::new(buffer![0i128], DecimalDType::new(4, 0), Validity::NonNullable),
    1,
    "allows at most 0 lower parts, got 1"
)]
#[case::legacy_wider_all_null(
    DecimalArray::new(buffer![i128::MAX], DecimalDType::new(4, 0), Validity::AllInvalid),
    1,
    "allows at most 0 lower parts, got 1"
)]
#[case::i64_one_at_p38(
    DecimalArray::new(buffer![0i64], DecimalDType::new(38, 0), Validity::NonNullable),
    1,
    "i64 values split into 0 lower parts, got 1"
)]
#[case::i128_three_at_p76(
    DecimalArray::new(buffer![0i128], DecimalDType::new(76, 0), Validity::NonNullable),
    3,
    "i128 values split into 0 or 1 lower parts, got 3"
)]
#[case::empty(
    decimal(Buffer::<i64>::empty(), Validity::NonNullable),
    1,
    "allows at most 0 lower parts, got 1"
)]
fn test_split_decimal_rejects_invalid_count(
    #[case] decimal: DecimalArray,
    #[case] lower_part_count: usize,
    #[case] expected: &str,
) -> VortexResult<()> {
    assert_split_error(&decimal, lower_part_count, expected);

    Ok(())
}

/// An array without valid values splits into constant parts with the requested count, not the
/// count of its storage width.
#[rstest]
#[case::i128_all_null_msp_only(decimal(buffer![1i128 << 100; 3], Validity::AllInvalid), 0)]
#[case::i128_empty_msp_only(decimal(Buffer::<i128>::empty(), Validity::NonNullable), 0)]
#[case::i256_all_null_one_lower_part(
    decimal(buffer![I256_ABOVE_I128; 3], Validity::AllInvalid),
    1
)]
fn test_split_without_valid_rows_uses_requested_count(
    #[case] decimal: DecimalArray,
    #[case] lower_part_count: usize,
) -> VortexResult<()> {
    let parts = split_decimal(&decimal, &validity_mask(&decimal)?, lower_part_count)?;
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
    let parts = split_decimal(&decimal, &validity_mask(&decimal)?, 1)?;
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
    let parts = split_decimal(&decimal, &validity_mask(&decimal)?, MAX_LOWER_PARTS)?;
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

/// A valid value outside the precision can need more lower parts than the precision allows.
#[test]
fn test_encode_rejects_lower_parts_past_canonical_width() {
    let decimal = DecimalArray::new(
        buffer![-99i128, 0, 1 << 70],
        DecimalDType::new(18, 0),
        Validity::NonNullable,
    );
    let result = DecimalByteParts::encode(&decimal, &mut SESSION.create_execution_ctx());
    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains("allows at most 0 lower parts")),
        "expected the lower-part bound to reject the parts, got {result:?}"
    );
}

#[rstest]
#[case::narrower_i32_at_p38(DecimalType::I32, 38, PType::I64)]
#[case::wider_i64_at_p4(DecimalType::I64, 4, PType::I16)]
#[case::wider_i128_at_p4(DecimalType::I128, 4, PType::I16)]
#[case::wider_i256_at_p4(DecimalType::I256, 4, PType::I16)]
fn test_split_legacy_storage_uses_logical_msp(
    #[case] stored_type: DecimalType,
    #[case] precision: u8,
    #[case] expected_ptype: PType,
    #[values(
        Validity::NonNullable,
        Validity::from_iter([true, false, true]),
        Validity::AllInvalid
    )]
    validity: Validity,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(precision, 0);
    let decimal = match_each_decimal_value_type!(stored_type, |T| {
        let mut values = [-99i16, 0, 99].map(|value| {
            <T as dtype::BigCast>::from(value)
                .vortex_expect("fixture values fit every storage width")
        });
        if !matches!(validity, Validity::NonNullable) {
            values[1] = T::MAX;
        }

        DecimalArray::new(Buffer::copy_from(values), decimal_dtype, validity)
    });
    let parts = split_decimal(&decimal, &validity_mask(&decimal)?, 0)?;
    assert!(parts.lower_parts.is_empty());
    assert_eq!(parts.msp.dtype().as_ptype(), expected_ptype);
    assert_eq!(
        parts.msp.dtype().nullability(),
        decimal.dtype().nullability()
    );
    let encoded = DecimalByteParts::try_new(parts.msp, decimal_dtype)?;
    assert_arrays_eq!(encoded, decimal.clone(), &mut ctx);
    assert_arrays_eq!(
        DecimalByteParts::encode(&decimal, &mut ctx)?,
        decimal,
        &mut ctx
    );

    Ok(())
}
