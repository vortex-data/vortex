// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of part validation and execution.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::DecimalByteParts;
use super::DecimalBytePartsArraySlotsExt;
use super::DecimalBytePartsData;
use crate::decimal_byte_parts::LOWER_PART_DTYPE;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;
use crate::decimal_byte_parts::testing::i128_parts;
use crate::decimal_byte_parts::testing::i256_parts;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_scalar_at_decimal_parts() {
    let decimal_dtype = DecimalDType::new(8, 2);
    let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
    let array = DecimalByteParts::try_new(
        PrimitiveArray::new(
            buffer![100i32, 200i32, 400i32],
            Validity::Array(BoolArray::from_iter(vec![false, true, true]).into_array()),
        )
        .into_array(),
        decimal_dtype,
    )
    .unwrap()
    .into_array();

    assert_eq!(
        Scalar::null(dtype.clone()),
        array
            .execute_scalar(0, &mut SESSION.create_execution_ctx())
            .unwrap()
    );
    assert_eq!(
        Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Decimal(DecimalValue::I64(200)))
        )
        .unwrap(),
        array
            .execute_scalar(1, &mut SESSION.create_execution_ctx())
            .unwrap()
    );
    assert_eq!(
        Scalar::try_new(dtype, Some(ScalarValue::Decimal(DecimalValue::I64(400)))).unwrap(),
        array
            .execute_scalar(2, &mut SESSION.create_execution_ctx())
            .unwrap()
    );
}

#[test]
fn test_lower_part_layout_i128() -> VortexResult<()> {
    let array = i128_parts(vec![(3i128 << 64) | 7], Validity::NonNullable);
    assert_eq!(array.lower_parts().len(), 1);
    assert_eq!(array.msp().dtype().as_ptype(), PType::I64);
    assert_eq!(array.lower_parts()[0].dtype(), &LOWER_PART_DTYPE);

    let mut ctx = SESSION.create_execution_ctx();
    let msp = array.msp().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let lower = array.lower_parts()[0]
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(msp.as_slice::<i64>(), &[3]);
    assert_eq!(lower.as_slice::<u64>(), &[7]);
    Ok(())
}

#[test]
fn test_lower_part_layout_i256() -> VortexResult<()> {
    let array = i256_parts(
        vec![i256::from_parts((7u128 << 64) | 8, (5i128 << 64) | 6)],
        Validity::NonNullable,
    );
    assert_eq!(array.lower_parts().len(), MAX_LOWER_PARTS);

    let mut ctx = SESSION.create_execution_ctx();
    let msp = array.msp().clone().execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(msp.as_slice::<i64>(), &[5]);
    for (part, expected) in array.lower_parts().iter().zip([6u64, 7, 8]) {
        let part = part.clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(part.as_slice::<u64>(), &[expected]);
    }
    Ok(())
}

#[rstest]
fn test_scalar_at_matches_canonical_for_each_part_count(
    #[values(false, true)] narrow_msp: bool,
    #[values(0, 1, 2, 3)] lower_count: usize,
) -> VortexResult<()> {
    let validity = Validity::from_iter([false, true, true]);
    let msp = if narrow_msp {
        PrimitiveArray::new(buffer![0i8, 3, -3], validity)
    } else {
        PrimitiveArray::new(buffer![0i64, 3, -3], validity)
    };
    let lower = [4u64, 1, 2]
        .into_iter()
        .take(lower_count)
        .map(|word| PrimitiveArray::new(buffer![word; 3], Validity::NonNullable).into_array())
        .collect();
    let dtype = DecimalDType::new(if lower_count <= 1 { 38 } else { 76 }, 0);
    let array = DecimalByteParts::try_new_with_lower_parts(msp.into_array(), lower, dtype)?;
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = array
        .clone()
        .into_array()
        .execute::<DecimalArray>(&mut ctx)?;
    for row in 0..array.len() {
        assert_eq!(
            array.execute_scalar(row, &mut ctx)?,
            canonical.execute_scalar(row, &mut ctx)?
        );
    }
    Ok(())
}

#[test]
fn test_scalar_at_null_with_lower_parts() -> VortexResult<()> {
    let array = i128_parts(
        vec![1i128 << 100, 2, 3],
        Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        Scalar::null(array.dtype().clone())
    );
    assert_eq!(
        array.execute_scalar(1, &mut ctx)?,
        Scalar::decimal(
            DecimalValue::I128(2),
            DecimalDType::new(38, 2),
            Nullability::Nullable
        )
    );
    Ok(())
}

#[rstest]
fn test_narrow_msp_with_u64_lower_parts(
    #[values(PType::I8, PType::I16, PType::I32, PType::I64)] msp_ptype: PType,
    #[values(1, 2, 3)] lower_count: usize,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from_iter([false, true, true]);
    let msp = PrimitiveArray::new(buffer![0i64, 1, -1], validity.clone())
        .into_array()
        .cast(DType::Primitive(msp_ptype, Nullability::Nullable))?;
    // Set the highest bit of each word to catch accidental sign extension. The low bits
    // differ, so a swapped word shows up too.
    let words = [(1u64 << 63) | 1, (1 << 63) | 2, (1 << 63) | 3];
    let lower = words
        .into_iter()
        .take(lower_count)
        .map(|word| buffer![0u64, word, word].into_array())
        .collect();
    let dtype = DecimalDType::new(if lower_count == 1 { 38 } else { 76 }, 0);
    let encoded = DecimalByteParts::try_new_with_lower_parts(msp, lower, dtype)?;
    let expected = match lower_count {
        1 => DecimalArray::new(
            buffer![
                0i128,
                (1i128 << 64) | i128::from(words[0]),
                (-1i128 << 64) | i128::from(words[0])
            ],
            dtype,
            validity,
        ),
        2 => {
            let low = (u128::from(words[0]) << 64) | u128::from(words[1]);
            DecimalArray::new(
                buffer![
                    i256::ZERO,
                    i256::from_parts(low, 1),
                    i256::from_parts(low, -1)
                ],
                dtype,
                validity,
            )
        }
        _ => {
            let low = (u128::from(words[1]) << 64) | u128::from(words[2]);
            DecimalArray::new(
                buffer![
                    i256::ZERO,
                    i256::from_parts(low, (1i128 << 64) | i128::from(words[0])),
                    i256::from_parts(low, (-1i128 << 64) | i128::from(words[0])),
                ],
                dtype,
                validity,
            )
        }
    };
    assert_arrays_eq!(encoded, expected, &mut ctx);
    for row in 0..expected.len() {
        assert_eq!(
            encoded.execute_scalar(row, &mut ctx)?,
            expected.execute_scalar(row, &mut ctx)?
        );
    }
    Ok(())
}

fn msp() -> ArrayRef {
    buffer![1i64, 2, 3].into_array()
}

fn lower_part() -> ArrayRef {
    buffer![1u64, 2, 3].into_array()
}

#[rstest]
#[case::signed_lower_part(
    vec![buffer![1i64, 2, 3].into_array()],
    DecimalDType::new(38, 2),
    "must have dtype u64, got i64"
)]
#[case::nullable_lower_part(
    vec![PrimitiveArray::new(buffer![1u64, 2, 3], Validity::AllValid).into_array()],
    DecimalDType::new(38, 2),
    "must have dtype u64, got u64?"
)]
#[case::u8_lower_part(
    vec![buffer![1u8, 2, 3].into_array()],
    DecimalDType::new(38, 2),
    "must have dtype u64, got u8"
)]
#[case::u16_lower_part(
    vec![buffer![1u16, 2, 3].into_array()],
    DecimalDType::new(38, 2),
    "must have dtype u64, got u16"
)]
#[case::u32_lower_part(
    vec![buffer![1u32, 2, 3].into_array()],
    DecimalDType::new(38, 2),
    "must have dtype u64, got u32"
)]
#[case::mismatched_length(
    vec![buffer![1u64, 2].into_array()],
    DecimalDType::new(38, 2),
    "length mismatch"
)]
#[case::too_many_parts(
    vec![lower_part(), lower_part(), lower_part(), lower_part()],
    DecimalDType::new(76, 2),
    "expected 1..=4 slots, got 5"
)]
#[case::one_lower_part_at_p18(
    vec![lower_part()],
    DecimalDType::new(18, 2),
    "allows at most 0 lower parts, got 1"
)]
#[case::three_lower_parts_at_p38(
    vec![lower_part(), lower_part(), lower_part()],
    DecimalDType::new(38, 2),
    "allows at most 1 lower parts, got 3"
)]
fn test_rejects_invalid_parts(
    #[case] lower_parts: Vec<ArrayRef>,
    #[case] decimal_dtype: DecimalDType,
    #[case] expected_error: &str,
) {
    let result = DecimalByteParts::try_new_with_lower_parts(msp(), lower_parts, decimal_dtype);
    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains(expected_error)),
        "expected {expected_error}, got {result:?}"
    );
}

#[rstest]
#[case::no_slots(vec![])]
#[case::missing_msp(vec![None])]
#[case::missing_lower(vec![Some(msp()), None])]
#[case::gap_in_lower(vec![Some(msp()), None, Some(lower_part())])]
fn test_rejects_missing_slots(#[case] slots: Vec<Option<ArrayRef>>) {
    let parts = ArrayParts::new(
        DecimalByteParts,
        DType::Decimal(DecimalDType::new(76, 2), Nullability::NonNullable),
        3,
        DecimalBytePartsData,
        slots.into_iter().collect(),
    );
    assert!(Array::try_from_parts(parts).is_err());
}

#[test]
fn test_wide_decimal_buffer_types() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let i128_array = i128_parts(vec![1i128 << 100], Validity::NonNullable);
    let canonical = i128_array.into_array().execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I128);

    let i256_array = i256_parts(vec![i256::from_parts(0, 1 << 100)], Validity::NonNullable);
    let canonical = i256_array.into_array().execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I256);

    // A narrow MSP with a single lower part still fits 128 bits.
    let array = DecimalByteParts::try_new_with_lower_parts(
        buffer![1i8, -1, 0].into_array(),
        vec![buffer![7u64, 7, 7].into_array()],
        DecimalDType::new(38, 2),
    )?;
    let canonical = array.into_array().execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I128);
    assert_eq!(
        canonical.buffer::<i128>().as_slice(),
        &[(1i128 << 64) | 7, (-1i128 << 64) | 7, 7]
    );

    // Two lower parts under a narrow MSP overflow 128 bits, so the value widens.
    let array = DecimalByteParts::try_new_with_lower_parts(
        buffer![1i8].into_array(),
        vec![buffer![0u64].into_array(), buffer![9u64].into_array()],
        DecimalDType::new(76, 2),
    )?;
    let canonical = array.into_array().execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I256);
    assert_eq!(
        canonical.buffer::<i256>().as_slice(),
        &[i256::from_parts(9, 1)]
    );
    Ok(())
}

#[test]
fn test_unused_buffer_of_values_is_ignored_for_null_rows() -> VortexResult<()> {
    // Null rows may hold arbitrary bits in the lower parts; they must stay null.
    let array = DecimalByteParts::try_new_with_lower_parts(
        PrimitiveArray::new(
            buffer![0i64, 0, 0],
            Validity::Array(BoolArray::from_iter([false, false, true]).into_array()),
        )
        .into_array(),
        vec![buffer![7u64, 9, 11].into_array()],
        DecimalDType::new(38, 2),
    )?
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        Scalar::null(array.dtype().clone())
    );
    assert_arrays_eq!(
        array,
        DecimalArray::new(
            buffer![0i128, 0, 11],
            DecimalDType::new(38, 2),
            Validity::from_iter([false, false, true]),
        ),
        &mut ctx
    );
    Ok(())
}

/// Planted exact statistics select the child without scanning the encoded MSP's values.
/// A scan of the actual values would pick `Decimal(1, 0)` instead.
#[rstest]
fn test_execute_keeps_stats_of_encoded_msp(
    #[values(false, true)] single_step: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let msp = ConstantArray::new(5i32, 3).into_array();
    msp.statistics().set(Stat::Min, Precision::exact(-999i32));
    msp.statistics().set(Stat::Max, Precision::exact(999i32));
    let array = DecimalByteParts::try_new(msp, DecimalDType::new(18, 0))?.into_array();

    let decoded = if single_step {
        let mut decoded = array;
        // One step executes the MSP, and the next one decodes.
        for _ in 0..2 {
            decoded = decoded.execute::<ArrayRef>(&mut ctx)?;
        }
        decoded
    } else {
        array.execute_until::<Widen>(&mut ctx)?
    };

    let widen = decoded
        .as_opt::<Widen>()
        .unwrap_or_else(|| panic!("expected a Widen, got {}", decoded.encoding_id()));
    assert_eq!(
        widen.values().dtype(),
        &DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable)
    );
    Ok(())
}
