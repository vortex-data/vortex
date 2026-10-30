// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of assembly widths, buffer reuse, and cached extrema.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::assemble_decimal;
use crate::DecimalByteParts;
use crate::decimal_byte_parts::split_decimal;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// The largest magnitude of a `Decimal(18, _)` value.
const MAX_P18: i64 = 999_999_999_999_999_999;

/// Decode an MSP with no lower parts, without executing the result.
fn decode_msp(msp: PrimitiveArray, decimal_dtype: DecimalDType) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    assemble_decimal(&msp.into_array(), &[], decimal_dtype, &mut ctx)
}

/// Returns the rows of a fixture at `nullability`. A `None` is a null row, and a
/// non-nullable fixture leaves the `None` rows out.
fn rows_at(values: &[Option<i64>], nullability: Nullability) -> Vec<Option<i64>> {
    match nullability {
        Nullability::NonNullable => values.iter().flatten().map(|&value| Some(value)).collect(),
        Nullability::Nullable => values.to_vec(),
    }
}

/// Returns the validity of `rows` at `nullability`.
fn validity_of(rows: &[Option<i64>], nullability: Nullability) -> Validity {
    match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::from_iter(rows.iter().map(Option::is_some)),
    }
}

/// Returns an MSP of `ptype` with the given rows. Each null row holds the largest value of
/// `ptype`, so a range that counted it would pick a wider child.
fn msp_of(ptype: PType, rows: &[Option<i64>], validity: Validity) -> PrimitiveArray {
    match_each_signed_integer_ptype!(ptype, |P| {
        let values = rows
            .iter()
            .map(|row| match row {
                Some(value) => num_traits::cast::<i64, P>(*value)
                    .vortex_expect("fixture values fit the MSP ptype"),
                None => P::MAX,
            })
            .collect::<Buffer<P>>();
        PrimitiveArray::new(values, validity)
    })
}

/// Returns `rows` as the canonical array of `decimal_dtype`, with zero in each null slot.
fn canonical_decimal(
    rows: &[Option<i64>],
    decimal_dtype: DecimalDType,
    validity: Validity,
) -> DecimalArray {
    match_each_decimal_value_type!(decimal_dtype.canonical_values_type(), |T| {
        let values = rows
            .iter()
            .map(|row| {
                <T as BigCast>::from(row.unwrap_or_default())
                    .vortex_expect("fixture values fit the canonical width")
            })
            .collect::<Buffer<T>>();
        DecimalArray::new(values, decimal_dtype, validity)
    })
}

/// Returns the decimal array that holds the values of `decoded`: the child of a `Widen`, or
/// `decoded` itself.
fn decimal_leaf(decoded: &ArrayRef) -> DecimalArray {
    let leaf = match decoded.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => decoded.clone(),
    };
    leaf.downcast::<Decimal>()
}

/// Asserts that `decoded` is a `Widen` over a child at `child_precision`, or the canonical
/// array of `decimal_dtype` when `child_precision` is `None`. Either way, the decimal leaf
/// has the canonical width of its own dtype.
#[track_caller]
fn assert_form(decoded: &ArrayRef, decimal_dtype: DecimalDType, child_precision: Option<u8>) {
    let leaf_dtype = match child_precision {
        Some(precision) => DecimalDType::new(precision, decimal_dtype.scale()),
        None => decimal_dtype,
    };
    let leaf = decimal_leaf(decoded);

    assert_eq!(decoded.is::<Widen>(), child_precision.is_some());
    assert_eq!(leaf.decimal_dtype(), leaf_dtype);
    assert_eq!(leaf.values_type(), leaf_dtype.canonical_values_type());
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
    let parts = split_decimal(&decimal, &mut ctx)?;
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
    let parts = split_decimal(&decimal, &mut ctx)?;
    assert_eq!(parts.lower_parts.len(), if wide_256 { 3 } else { 1 });
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

fn round_trip(decimal: DecimalArray) -> VortexResult<DecimalArray> {
    let mut ctx = SESSION.create_execution_ctx();
    let parts = split_decimal(&decimal, &mut ctx)?;
    assemble_decimal(
        &parts.msp,
        &parts.lower_parts,
        decimal.decimal_dtype(),
        &mut ctx,
    )?
    .execute::<DecimalArray>(&mut ctx)
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
fn test_split_narrow_decimal_reuses_values(
    #[values(Validity::NonNullable, Validity::from_iter([true, false, true]))] validity: Validity,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalArray::new(buffer![1i32, 2, 3], DecimalDType::new(9, 0), validity);
    let values_ptr = decimal.buffer::<i32>().as_ptr();

    let parts = split_decimal(&decimal, &mut ctx)?;
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

#[rstest]
#[case::signed(PrimitiveArray::new(buffer![0i64; 2], Validity::NonNullable))]
#[case::narrow(PrimitiveArray::new(buffer![0u32; 2], Validity::NonNullable))]
#[case::float(PrimitiveArray::new(buffer![0f32; 2], Validity::NonNullable))]
#[case::nullable_all_valid(PrimitiveArray::new(buffer![0u64; 2], Validity::AllValid))]
#[case::nullable_all_null(PrimitiveArray::new(buffer![0u64; 2], Validity::AllInvalid))]
#[case::nullable_mixed(PrimitiveArray::new(buffer![0u64; 2], Validity::from_iter([true, false])))]
fn test_assemble_rejects_invalid_lower_dtype(
    #[case] invalid_lower: PrimitiveArray,
    #[values(1, 2, 3)] lower_count: usize,
) {
    let msp = PrimitiveArray::new(buffer![0i64; 2], Validity::NonNullable);
    let mut lower = vec![PrimitiveArray::new(buffer![0u64; 2], Validity::NonNullable); lower_count];
    lower[lower_count - 1] = invalid_lower;
    let dtype = DecimalDType::new(if lower_count == 1 { 38 } else { 76 }, 0);
    let lower = lower
        .into_iter()
        .map(IntoArray::into_array)
        .collect::<Vec<_>>();
    let mut ctx = SESSION.create_execution_ctx();
    assert!(assemble_decimal(&msp.into_array(), &lower, dtype, &mut ctx).is_err());
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
    let lower = lower
        .into_iter()
        .map(IntoArray::into_array)
        .collect::<Vec<_>>();
    let mut ctx = SESSION.create_execution_ctx();
    assert!(assemble_decimal(&msp.into_array(), &lower, dtype, &mut ctx).is_err());
}

#[rstest]
fn test_assemble_i256_part_order_and_sign_extension(
    #[values(false, true)] narrow_msp: bool,
    #[values(2, 3)] lower_count: usize,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let msp = if narrow_msp {
        PrimitiveArray::new(buffer![3i8, -3], Validity::NonNullable)
    } else {
        PrimitiveArray::new(buffer![3i64, -3], Validity::NonNullable)
    };
    let lower = [4u64, 1, 2].map(|word| buffer![word; 2].into_array());
    let dtype = DecimalDType::new(76, 0);
    let actual = assemble_decimal(
        &msp.into_array(),
        &lower[3 - lower_count..],
        dtype,
        &mut ctx,
    )?;
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
        actual,
        DecimalArray::new(expected, dtype, Validity::NonNullable),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::canonical_reuses(
    PType::I32,
    vec![Some(-999_999_999), None, Some(999_999_999)],
    DecimalDType::new(9, 2),
    None,
    true,
)]
#[case::canonical_trusts_precision(
    PType::I32,
    vec![Some(2_000_000_000), None],
    DecimalDType::new(9, 2),
    None,
    true,
)]
#[case::narrow_reuses(
    PType::I32,
    vec![Some(-999_999_999), None, Some(999_999_999)],
    DecimalDType::new(18, 2),
    Some(9),
    true,
)]
#[case::gap_widens_child(
    PType::I16,
    vec![Some(10_000), None, Some(32_767)],
    DecimalDType::new(10, 2),
    Some(5),
    false,
)]
#[case::gap_needs_canonical(
    PType::I32,
    vec![Some(1_000_000_000), None, Some(i32::MAX.into())],
    DecimalDType::new(18, 2),
    None,
    false,
)]
#[case::high_scale(
    PType::I8,
    vec![Some(-1), None, Some(1)],
    DecimalDType::new(18, 10),
    None,
    false,
)]
#[case::negative_min_counts(
    PType::I8,
    vec![Some(-128), None, Some(5)],
    DecimalDType::new(10, 0),
    Some(3),
    false,
)]
#[case::wider_to_canonical(
    PType::I64,
    vec![Some(-9_999), None, Some(9_999)],
    DecimalDType::new(4, 2),
    None,
    false,
)]
#[case::wider_to_widen(
    PType::I64,
    vec![Some(-99), None, Some(99)],
    DecimalDType::new(9, 2),
    Some(2),
    false,
)]
#[case::i8_at_p76(
    PType::I8,
    vec![Some(-99), None, Some(99)],
    DecimalDType::new(76, 2),
    Some(2),
    true,
)]
#[case::all_null(PType::I32, vec![None, None], DecimalDType::new(18, 2), Some(2), false)]
#[case::empty(PType::I32, vec![], DecimalDType::new(18, 2), Some(2), false)]
fn test_assemble_result_form(
    #[case] msp_ptype: PType,
    #[case] values: Vec<Option<i64>>,
    #[case] decimal_dtype: DecimalDType,
    #[case] child_precision: Option<u8>,
    #[case] reuses_msp: bool,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = rows_at(&values, nullability);
    let validity = validity_of(&rows, nullability);
    let msp = msp_of(msp_ptype, &rows, validity.clone());
    let msp_ptr = msp.buffer_handle().as_host().as_ptr();

    let decoded = decode_msp(msp, decimal_dtype)?;

    assert_form(&decoded, decimal_dtype, child_precision);
    if reuses_msp {
        let leaf = decimal_leaf(&decoded);
        assert_eq!(leaf.buffer_handle().as_host().as_ptr(), msp_ptr);
    }
    assert_arrays_eq!(
        decoded,
        canonical_decimal(&rows, decimal_dtype, validity),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_assemble_narrowing_copy_zeroes_null_slots() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let msp = PrimitiveArray::new(
        buffer![1i64, 1000, -1],
        Validity::from_iter([true, false, true]),
    );

    let decoded = decode_msp(msp, DecimalDType::new(2, 0))?.execute::<DecimalArray>(&mut ctx)?;

    assert_eq!(decoded.values_type(), DecimalType::I8);
    assert_eq!(decoded.buffer::<i8>().as_slice(), &[1, 0, -1]);
    Ok(())
}

#[test]
fn test_assemble_narrowing_copy_rejects_value_outside_precision() {
    // 100 fits `i8`, but not precision 2.
    let msp = PrimitiveArray::new(buffer![1i64, 100], Validity::NonNullable);

    let result = decode_msp(msp, DecimalDType::new(2, 0));

    assert!(
        result
            .as_ref()
            .is_err_and(|err| err.to_string().contains("Decimal range must fit")),
        "expected the range of the MSP to reject 100, got {result:?}"
    );
}

#[test]
fn test_assemble_widening_sign_extends() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = DecimalDType::new(40, 0);
    let msp = PrimitiveArray::new(buffer![-1i8, 1], Validity::NonNullable);

    let decoded = decode_msp(msp, decimal_dtype)?;

    assert_form(&decoded, decimal_dtype, Some(1));
    assert_eq!(decimal_leaf(&decoded).values_type(), DecimalType::I8);
    let canonical = decoded.execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I256);
    assert_eq!(
        canonical.buffer::<i256>().as_slice(),
        &[i256::from_i128(-1), i256::ONE]
    );
    Ok(())
}

#[rstest]
#[case::one_lower_part_at_p38(DecimalArray::new(
    buffer![(3i128 << 64) | 2, (-3i128 << 64) | 2],
    DecimalDType::new(38, 0),
    Validity::NonNullable,
), 1, None)]
#[case::one_lower_part_at_p40(DecimalArray::new(
    buffer![i256::from_i128((3i128 << 64) | 2), i256::from_i128((-3i128 << 64) | 2)],
    DecimalDType::new(40, 0),
    Validity::NonNullable,
), 1, Some(20))]
#[case::two_lower_parts_at_p76(DecimalArray::new(
    buffer![
        i256::from_parts((1u128 << 64) | 2, 3),
        i256::from_parts((1u128 << 64) | 2, -3),
    ],
    DecimalDType::new(76, 0),
    Validity::NonNullable,
), 2, None)]
#[case::three_lower_parts_at_p76(DecimalArray::new(
    buffer![
        i256::from_parts((1u128 << 64) | 2, (3i128 << 64) | 4),
        i256::from_parts((1u128 << 64) | 2, (-3i128 << 64) | 4),
    ],
    DecimalDType::new(76, 0),
    Validity::NonNullable,
), 3, None)]
fn test_assemble_lower_parts_result_form(
    #[case] expected: DecimalArray,
    #[case] lower_count: usize,
    #[case] child_precision: Option<u8>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal_dtype = expected.decimal_dtype();
    let msp = buffer![3i64, -3].into_array();
    let lower = [4u64, 1, 2].map(|word| buffer![word; 2].into_array());

    let decoded = assemble_decimal(&msp, &lower[3 - lower_count..], decimal_dtype, &mut ctx)?;

    assert_form(&decoded, decimal_dtype, child_precision);
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

/// Planted exact statistics select the child without scanning the MSP values.
#[test]
fn test_assemble_prefers_cached_msp_range() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(18, 0);
    let msp = PrimitiveArray::new(buffer![-5i32, 7], Validity::NonNullable);
    msp.statistics().set(Stat::Min, Precision::exact(-999i32));
    msp.statistics().set(Stat::Max, Precision::exact(999i32));

    let decoded = decode_msp(msp, decimal_dtype)?;

    assert_form(&decoded, decimal_dtype, Some(3));
    Ok(())
}

/// The range from `min_max` on the MSP is exact, so the child records it as exact extrema, and
/// a later `min_max` on the decoded array reads them without a scan.
#[test]
fn test_assemble_records_exact_child_extrema() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(18, 0);
    let msp = PrimitiveArray::new(buffer![-5i32, 7], Validity::NonNullable);

    let decoded = decode_msp(msp, decimal_dtype)?;

    assert_form(&decoded, decimal_dtype, Some(1));
    let child_dtype = DecimalDType::new(1, 0);
    let extremum = |value: i8| {
        Scalar::decimal(
            DecimalValue::I8(value),
            child_dtype,
            Nullability::NonNullable,
        )
    };
    let leaf = decimal_leaf(&decoded);
    let statistics = leaf.statistics();
    assert_eq!(statistics.get(Stat::Min), Precision::Exact(extremum(-5)));
    assert_eq!(statistics.get(Stat::Max), Precision::Exact(extremum(7)));
    Ok(())
}

#[test]
fn test_assemble_ignores_inexact_stats() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(18, 0);
    let msp = PrimitiveArray::new(buffer![-5i32, 7], Validity::NonNullable);
    msp.statistics().set(Stat::Min, Precision::inexact(-999i32));
    msp.statistics().set(Stat::Max, Precision::inexact(999i32));

    let decoded = decode_msp(msp, decimal_dtype)?;

    assert_form(&decoded, decimal_dtype, Some(1));
    Ok(())
}

/// Pins the forms that the `dbp_compute` benches and the `dbp_bp_i32` tree of
/// `common_encoding_tree_throughput` decode to. Each MSP holds the extremes of the bench
/// values. `dbp_bp_i32` draws its values from `0..10_000_000`, so its 2048 values reach
/// precision 7. Its MSP is bit-packed in the bench, which decodes to the same `i32` values.
#[rstest]
#[case::msp_i32_p18(
    PType::I32,
    vec![Some(-999_999_999), Some(999_999_999)],
    DecimalDType::new(18, 2),
    Some(9),
    true,
)]
#[case::msp_i32_p18_nullable(
    PType::I32,
    vec![Some(-999_999_999), None, Some(999_999_999)],
    DecimalDType::new(18, 2),
    Some(9),
    true,
)]
#[case::gap_msp_i16_p10(
    PType::I16,
    vec![Some(10_000), Some(32_767)],
    DecimalDType::new(10, 2),
    Some(5),
    false,
)]
#[case::msp_i64_p38(
    PType::I64,
    vec![Some(-MAX_P18), Some(MAX_P18)],
    DecimalDType::new(38, 2),
    Some(18),
    true,
)]
#[case::msp_i64_p18_canonical(
    PType::I64,
    vec![Some(-MAX_P18), Some(MAX_P18)],
    DecimalDType::new(18, 2),
    None,
    true,
)]
#[case::dbp_bp_i32(
    PType::I32,
    vec![Some(0), Some(9_999_999)],
    DecimalDType::new(12, 2),
    Some(7),
    true,
)]
fn test_execute_until_widen_matches_bench_cases(
    #[case] msp_ptype: PType,
    #[case] rows: Vec<Option<i64>>,
    #[case] decimal_dtype: DecimalDType,
    #[case] child_precision: Option<u8>,
    #[case] reuses_msp: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let nullability = if rows.iter().all(Option::is_some) {
        Nullability::NonNullable
    } else {
        Nullability::Nullable
    };
    let validity = validity_of(&rows, nullability);
    let msp = msp_of(msp_ptype, &rows, validity.clone());
    let msp_ptr = msp.buffer_handle().as_host().as_ptr();
    let array = DecimalByteParts::try_new(msp.into_array(), decimal_dtype)?.into_array();

    let decoded = array.execute_until::<Widen>(&mut ctx)?;

    assert_form(&decoded, decimal_dtype, child_precision);
    let leaf = decimal_leaf(&decoded);
    assert_eq!(
        leaf.buffer_handle().as_host().as_ptr() == msp_ptr,
        reuses_msp
    );
    assert_arrays_eq!(
        decoded,
        canonical_decimal(&rows, decimal_dtype, validity),
        &mut ctx
    );
    Ok(())
}
