// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::*;

#[rstest]
#[case::non_nullable(Validity::NonNullable)]
#[case::all_valid(Validity::AllValid)]
#[case::all_null(Validity::AllInvalid)]
#[case::mixed(Validity::from_iter((0..263).map(|i| i % 3 != 1)))]
#[case::sparse(Validity::from_iter((0..263).map(|i| i % 16 == 0)))]
#[case::null_prefix_and_suffix(Validity::from_iter((0..263).map(|i| (67..196).contains(&i))))]
fn test_split_zeroes_null_words(
    #[case] validity: Validity,
    #[values(false, true)] wide_256: bool,
    #[values(0, 1, 63, 64, 65, 257)] len: usize,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal = if wide_256 {
        DecimalArray::new(
            buffer![i256::from_i128(-1); 263],
            DecimalDType::new(76, 2),
            validity,
        )
    } else {
        DecimalArray::new(buffer![-1i128; 263], DecimalDType::new(38, 2), validity)
    };
    let decimal = decimal
        .slice(3..len + 3)?
        .execute::<DecimalArray>(&mut ctx)?;
    let expected = PrimitiveArray::new(
        decimal
            .validity()?
            .execute_mask(len, &mut ctx)?
            .iter()
            .map(|valid| if valid { u64::MAX } else { 0 })
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );
    let parts = split_decimal(&decimal, &mut ctx)?;
    for lower in parts.lower_parts {
        assert_arrays_eq!(expected.clone(), lower, &mut ctx);
    }
    assert_arrays_eq!(decimal.clone(), round_trip(decimal)?, &mut ctx);
    Ok(())
}

fn round_trip(decimal: DecimalArray) -> VortexResult<DecimalArray> {
    let mut ctx = array_session().create_execution_ctx();
    let parts = split_decimal(&decimal, &mut ctx)?;
    let msp = parts.msp.execute::<PrimitiveArray>(&mut ctx)?;
    let lower = parts
        .lower_parts
        .into_iter()
        .map(|part| part.execute::<PrimitiveArray>(&mut ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    assemble_decimal(&msp, &lower, decimal.decimal_dtype())
}

#[rstest]
#[case::zero(0)]
#[case::one(1)]
#[case::minus_one(-1)]
#[case::limb_boundary(1i128 << 64)]
#[case::just_below_limb_boundary((1i128 << 64) - 1)]
#[case::negative_limb_boundary(-(1i128 << 64))]
#[case::max(i128::MAX)]
#[case::min(i128::MIN)]
fn test_split_assemble_i128(#[case] value: i128) -> VortexResult<()> {
    let decimal = DecimalArray::new(
        Buffer::from(vec![value]),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    );
    let round_tripped = round_trip(decimal)?;
    assert_eq!(round_tripped.buffer::<i128>().as_slice(), &[value]);
    Ok(())
}

#[rstest]
#[case::zero(i256::ZERO)]
#[case::one(i256::ONE)]
#[case::minus_one(i256::ZERO - i256::ONE)]
#[case::max(i256::MAX)]
#[case::min(i256::MIN)]
#[case::word_1(i256::from_parts(1u128 << 64, 0))]
#[case::word_2(i256::from_parts(0, 1))]
#[case::word_3(i256::from_parts(0, 1i128 << 64))]
#[case::mixed(i256::from_parts(u128::MAX, -3))]
fn test_split_assemble_i256(#[case] value: i256) -> VortexResult<()> {
    let decimal = DecimalArray::new(
        Buffer::from(vec![value]),
        DecimalDType::new(76, 2),
        Validity::NonNullable,
    );
    let round_tripped = round_trip(decimal)?;
    assert_eq!(round_tripped.buffer::<i256>().as_slice(), &[value]);
    Ok(())
}

#[rstest]
fn test_split_narrow_decimal_has_no_lower_parts(
    #[values(Validity::NonNullable, Validity::AllInvalid, Validity::from_iter([true, false, true]))]
    validity: Validity,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal = DecimalArray::new(buffer![1i32, 2, 3], DecimalDType::new(2, 0), validity);
    let parts = split_decimal(&decimal, &mut ctx)?;
    assert!(parts.lower_parts.is_empty());
    assert_eq!(parts.msp.dtype().as_ptype(), PType::I32);
    let msp = parts.msp.execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(
        msp.as_slice::<i32>().as_ptr(),
        decimal.buffer::<i32>().as_ptr()
    );
    assert_arrays_eq!(decimal.clone(), round_trip(decimal)?, &mut ctx);
    Ok(())
}

#[test]
fn test_split_i256_part_count_and_types() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal = DecimalArray::new(
        Buffer::from(vec![i256::from_i128(i128::MAX), i256::MIN]),
        DecimalDType::new(76, 0),
        Validity::NonNullable,
    );
    let parts = split_decimal(&decimal, &mut ctx)?;
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
    let mut ctx = array_session().create_execution_ctx();
    let decimal = DecimalArray::new(
        buffer![
            i256::from_parts((2u128 << 64) | 3, (1i128 << 64) | 4),
            i256::ZERO,
            i256::from_parts((6u128 << 64) | 7, (-2i128 << 64) | 5),
        ],
        DecimalDType::new(76, 0),
        validity.clone(),
    );
    let parts = split_decimal(&decimal, &mut ctx)?;
    assert_arrays_eq!(
        PrimitiveArray::new(buffer![1i64, 0, -2], validity),
        parts.msp,
        &mut ctx
    );
    assert_eq!(parts.lower_parts.len(), 3);
    for (part, expected) in parts.lower_parts.into_iter().zip([
        buffer![4u64, 0, 5],
        buffer![2u64, 0, 6],
        buffer![3u64, 0, 7],
    ]) {
        assert_arrays_eq!(
            PrimitiveArray::new(expected, Validity::NonNullable),
            part,
            &mut ctx
        );
    }
    Ok(())
}

#[rstest]
fn test_assemble_rejects_mismatched_lower_lengths(
    #[values(1, 2, 3)] lower_count: usize,
    #[values(0, 1, 3)] lower_len: usize,
) {
    let msp = PrimitiveArray::new(buffer![0i64; 2], Validity::NonNullable);
    let mut lower = vec![PrimitiveArray::new(buffer![0u64; 2], Validity::NonNullable); lower_count];
    lower[lower_count - 1] = PrimitiveArray::new(buffer![0u64; lower_len], Validity::NonNullable);
    let dtype = DecimalDType::new(if lower_count == 1 { 38 } else { 76 }, 0);
    assert!(assemble_decimal(&msp, &lower, dtype).is_err());
}

#[rstest]
fn test_assemble_i256_part_order_and_sign_extension(
    #[values(false, true)] narrow_msp: bool,
    #[values(2, 3)] lower_count: usize,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let msp = if narrow_msp {
        PrimitiveArray::new(buffer![3i8, -3], Validity::NonNullable)
    } else {
        PrimitiveArray::new(buffer![3i64, -3], Validity::NonNullable)
    };
    let lower =
        [4u64, 1, 2].map(|word| PrimitiveArray::new(buffer![word; 2], Validity::NonNullable));
    let dtype = DecimalDType::new(76, 0);
    let actual = assemble_decimal(&msp, &lower[3 - lower_count..], dtype)?;
    let low = (1u128 << 64) | 2;
    let expected = if lower_count == 2 {
        buffer![i256::from_parts(low, 3), i256::from_parts(low, -3)]
    } else {
        buffer![
            i256::from_parts(low, (3i128 << 64) | 4),
            i256::from_parts(low, (-3i128 << 64) | 4),
        ]
    };
    assert_arrays_eq!(
        DecimalArray::new(expected, dtype, Validity::NonNullable),
        actual,
        &mut ctx
    );
    Ok(())
}
