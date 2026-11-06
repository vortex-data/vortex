// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of assembly widths and buffer reuse.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_decimal_value_type;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::assemble_decimal;
use crate::decimal_byte_parts::split_decimal;
use crate::decimal_byte_parts::testing::full_width_lower_part_count;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

fn round_trip(decimal: DecimalArray) -> VortexResult<DecimalArray> {
    let mut ctx = SESSION.create_execution_ctx();
    let lower_part_count = full_width_lower_part_count(decimal.values_type());
    let parts = split_decimal(&decimal, lower_part_count, &mut ctx)?;
    assemble_decimal(
        &parts.msp,
        &parts.lower_parts,
        decimal.decimal_dtype(),
        &mut ctx,
    )?
    .execute::<DecimalArray>(&mut ctx)
}

#[rstest]
#[case::empty_non_nullable(0, Validity::NonNullable)]
#[case::empty_nullable(0, Validity::AllValid)]
#[case::empty_all_null(0, Validity::AllInvalid)]
#[case::all_null(3, Validity::AllInvalid)]
#[case::all_null_array(3, Validity::Array(BoolArray::from_iter([false; 3]).into_array()))]
fn test_split_without_valid_rows(
    #[case] len: usize,
    #[case] validity: Validity,
    #[values(
        DecimalType::I8,
        DecimalType::I16,
        DecimalType::I32,
        DecimalType::I64,
        DecimalType::I128,
        DecimalType::I256
    )]
    values_type: DecimalType,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = match_each_decimal_value_type!(values_type, |T| {
        DecimalArray::new(
            Buffer::<T>::zeroed(len),
            DecimalDType::new(T::MAX_PRECISION, 0),
            validity,
        )
    });
    let parts = split_decimal(&decimal, full_width_lower_part_count(values_type), &mut ctx)?;
    assert!(parts.msp.is::<Constant>());
    assert!(parts.lower_parts.iter().all(|part| part.is::<Constant>()));
    assert_eq!(parts.msp.len(), len);
    assert_eq!(
        parts.msp.dtype().nullability(),
        decimal.dtype().nullability()
    );
    let round_tripped = round_trip(decimal.clone())?;
    assert_eq!(round_tripped.values_type(), values_type);
    assert_arrays_eq!(round_tripped, decimal, &mut ctx);

    Ok(())
}

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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mask = decimal.validity()?.execute_mask(len, &mut ctx)?;
    let expected = PrimitiveArray::new(
        mask.iter()
            .map(|valid| if valid { u64::MAX } else { 0 })
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );
    let lower_part_count = if wide_256 { 3 } else { 1 };
    let parts = split_decimal(&decimal, lower_part_count, &mut ctx)?;
    assert_eq!(parts.lower_parts.len(), lower_part_count);
    assert_eq!(
        parts.msp.dtype(),
        &DType::Primitive(PType::I64, decimal.dtype().nullability())
    );
    for lower in parts.lower_parts {
        assert_arrays_eq!(lower, expected.clone(), &mut ctx);
    }
    assert_arrays_eq!(round_trip(decimal.clone())?, decimal, &mut ctx);

    Ok(())
}

#[rstest]
#[case::i128_zero(0i128)]
#[case::i128_one(1i128)]
#[case::i128_minus_one(-1i128)]
#[case::i128_limb_boundary(1i128 << 64)]
#[case::i128_just_below_limb_boundary((1i128 << 64) - 1)]
#[case::i128_negative_limb_boundary(-(1i128 << 64))]
#[case::i128_max(i128::MAX)]
#[case::i128_min(i128::MIN)]
#[case::i256_zero(i256::ZERO)]
#[case::i256_one(i256::ONE)]
#[case::i256_minus_one(i256::ZERO - i256::ONE)]
#[case::i256_max(i256::MAX)]
#[case::i256_min(i256::MIN)]
#[case::i256_word_1(i256::from_parts(1u128 << 64, 0))]
#[case::i256_word_2(i256::from_parts(0, 1))]
#[case::i256_word_3(i256::from_parts(0, 1i128 << 64))]
#[case::i256_mixed(i256::from_parts(u128::MAX, -3))]
fn test_split_assemble<T: NativeDecimalType>(#[case] value: T) -> VortexResult<()> {
    let decimal = DecimalArray::new(
        Buffer::from(vec![value]),
        DecimalDType::new(T::MAX_PRECISION, 2),
        Validity::NonNullable,
    );
    let round_tripped = round_trip(decimal)?;
    assert_eq!(round_tripped.buffer::<T>().as_slice(), &[value]);

    Ok(())
}

#[rstest]
fn test_split_narrow_decimal_reuses_values(
    #[values(Validity::NonNullable, Validity::from_iter([true, false, true]))] validity: Validity,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalArray::new(buffer![1i32, 2, 3], DecimalDType::new(9, 0), validity);
    let values_ptr = decimal.buffer::<i32>().as_ptr();

    let parts = split_decimal(&decimal, 0, &mut ctx)?;
    assert!(parts.lower_parts.is_empty());
    assert_eq!(parts.msp.dtype().as_ptype(), PType::I32);
    let msp = parts.msp.clone().execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(msp.as_slice::<i32>().as_ptr(), values_ptr);

    let decoded = assemble_decimal(
        &parts.msp,
        &parts.lower_parts,
        decimal.decimal_dtype(),
        &mut ctx,
    )?
    .execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(decoded.buffer::<i32>().as_ptr(), values_ptr);
    assert_arrays_eq!(decoded, decimal, &mut ctx);

    Ok(())
}

#[test]
fn test_assemble_validates_parts() {
    let msp = buffer![0i64; 2].into_array();
    let lower = [buffer![0u32; 2].into_array()];

    let result = assemble_decimal(
        &msp,
        &lower,
        DecimalDType::new(38, 0),
        &mut SESSION.create_execution_ctx(),
    );

    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains("must have dtype u64")),
        "expected the lower-part dtype check to reject a u32 part, got {result:?}"
    );
}
