// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use half::f16;
use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::ApproxDistinct;
use super::ApproxDistinctOptions;
use super::DEFAULT_PRECISION;
use super::HyperLogLog;
use super::MAX_PRECISION;
use super::MIN_PRECISION;
use super::approx_distinct;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateDTypes;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::ExtensionArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::Scalar;
use crate::test_harness::check_metadata;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// A string longer than the 12 bytes a view can inline, so it is stored in a data buffer.
const LONG: &str = "a value long enough to live outside the view";

fn estimate(array: &ArrayRef) -> VortexResult<u64> {
    approx_distinct(array, &mut SESSION.create_execution_ctx())
}

fn sketch_of_batches(batches: &[ArrayRef]) -> VortexResult<HyperLogLog> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut accumulator = Accumulator::try_new(
        ApproxDistinct,
        ApproxDistinctOptions::default(),
        batches[0].dtype().clone(),
    )?;
    for batch in batches {
        accumulator.accumulate(batch, &mut ctx)?;
    }
    let partial = accumulator.flush()?;
    let bytes = partial.as_binary().value().expect("partial is non-null");
    HyperLogLog::deserialize(bytes.as_slice())
}

fn sketch_of(array: &ArrayRef) -> VortexResult<HyperLogLog> {
    sketch_of_batches(std::slice::from_ref(array))
}

fn i64_dtypes() -> VortexResult<AggregateDTypes> {
    AggregateDTypes::try_new(
        &ApproxDistinct,
        &ApproxDistinctOptions::default(),
        DType::from(PType::I64),
    )
}

fn dictionary_with_unused_value() -> ArrayRef {
    DictArray::try_new(
        buffer![0u8, 0, 1].into_array(),
        VarBinViewArray::from_iter_str(["a", "b", "unused"]).into_array(),
    )
    .expect("valid dictionary")
    .into_array()
}

fn slice_of_four() -> ArrayRef {
    PrimitiveArray::from_iter(0i64..100)
        .into_array()
        .slice(10..14)
        .expect("valid slice")
}

fn chunked_zero_to_thousand() -> ArrayRef {
    ChunkedArray::try_new(
        [
            PrimitiveArray::from_iter(0i64..300).into_array(),
            PrimitiveArray::from_iter(300i64..301).into_array(),
            PrimitiveArray::from_iter(301i64..1_000).into_array(),
        ],
        DType::from(PType::I64),
    )
    .expect("valid chunks")
    .into_array()
}

fn timestamp(nullability: Nullability) -> DType {
    DType::Extension(Timestamp::new(TimeUnit::Milliseconds, nullability).erased())
}

#[test]
fn test_options() -> VortexResult<()> {
    assert_eq!(
        ApproxDistinctOptions::default().precision(),
        DEFAULT_PRECISION
    );
    assert_eq!(
        ApproxDistinctOptions::try_new(14)?.to_string(),
        "precision=14"
    );
    assert!(ApproxDistinctOptions::try_new(MIN_PRECISION - 1).is_err());
    assert!(ApproxDistinctOptions::try_new(MAX_PRECISION + 1).is_err());

    for precision in MIN_PRECISION..=MAX_PRECISION {
        let options = ApproxDistinctOptions::try_new(precision)?;
        let metadata = ApproxDistinct.serialize(&options)?.expect("serializable");
        assert_eq!(ApproxDistinct.deserialize(&metadata, &SESSION)?, options);
    }
    Ok(())
}

#[rstest]
#[case::empty(&[])]
#[case::too_long(&[DEFAULT_PRECISION, 0])]
#[case::precision_too_small(&[MIN_PRECISION - 1])]
#[case::precision_too_large(&[MAX_PRECISION + 1])]
fn test_options_deserialize_rejects_invalid(#[case] metadata: &[u8]) {
    assert!(ApproxDistinct.deserialize(metadata, &SESSION).is_err());
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_options_metadata_golden() -> VortexResult<()> {
    // A fixed precision rather than the default, so changing the default can't change the bytes.
    let options = ApproxDistinctOptions::try_new(12)?;
    let metadata = ApproxDistinct.serialize(&options)?.expect("serializable");
    check_metadata("approx_distinct_hll_options_p12.metadata", &metadata);
    Ok(())
}

#[rstest]
#[case::null(DType::Null, true)]
#[case::bool(DType::Bool(Nullability::Nullable), true)]
#[case::primitive(DType::Primitive(PType::F64, Nullability::NonNullable), true)]
#[case::utf8(DType::Utf8(Nullability::Nullable), true)]
#[case::binary(DType::Binary(Nullability::NonNullable), true)]
#[case::timestamp(timestamp(Nullability::Nullable), true)]
#[case::decimal(
    DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable),
    false
)]
#[case::struct_(
    DType::Struct(
        StructFields::from_iter([("a", DType::from(PType::I32))]),
        Nullability::NonNullable,
    ),
    false
)]
#[case::list(
    DType::List(Arc::new(DType::from(PType::I32)), Nullability::NonNullable),
    false
)]
#[case::fixed_size_list(
    DType::FixedSizeList(Arc::new(DType::from(PType::F32)), 4, Nullability::NonNullable),
    false
)]
fn test_dtype_support(#[case] dtype: DType, #[case] supported: bool) {
    let options = ApproxDistinctOptions::default();
    let expected_return =
        supported.then_some(DType::Primitive(PType::U64, Nullability::NonNullable));
    let expected_partial = supported.then_some(DType::Binary(Nullability::NonNullable));
    assert_eq!(
        ApproxDistinct.return_dtype(&options, &dtype),
        expected_return
    );
    assert_eq!(
        ApproxDistinct.partial_dtype(&options, &dtype),
        expected_partial
    );
    assert_eq!(
        Accumulator::try_new(ApproxDistinct, options, dtype).is_ok(),
        supported
    );
}

// A handful of values almost never share one of 4,096 registers, so these counts are exact.
#[rstest]
#[case::empty_primitive(PrimitiveArray::from_iter(Vec::<i32>::new()).into_array(), 0)]
#[case::null_array(NullArray::new(10).into_array(), 0)]
#[case::all_null_primitive(
    PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllInvalid).into_array(),
    0
)]
#[case::all_null_utf8(VarBinViewArray::from_iter_nullable_str([None::<&str>, None]).into_array(), 0)]
#[case::null_constant(
    ConstantArray::new(Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)), 5)
        .into_array(),
    0
)]
#[case::empty_constant(
    ConstantArray::new(Scalar::primitive(7i32, Nullability::NonNullable), 0).into_array(),
    0
)]
#[case::nulls_not_counted(
    PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), None, Some(1)]).into_array(),
    2
)]
#[case::many_duplicates(
    PrimitiveArray::from_iter(std::iter::repeat_n(42i64, 10_000)).into_array(),
    1
)]
#[case::bool_both(BoolArray::from_iter([true, false, true]).into_array(), 2)]
#[case::bool_only_true(BoolArray::from_iter([true; 5]).into_array(), 1)]
#[case::bool_with_null(BoolArray::from_iter([Some(false), None]).into_array(), 1)]
#[case::bool_false_hidden_by_null(
    BoolArray::new([true, false].into_iter().collect(), Validity::from_iter([true, false]))
        .into_array(),
    1
)]
#[case::negative_and_positive_zero(PrimitiveArray::from_iter([-0.0f64, 0.0]).into_array(), 2)]
#[case::same_nan_twice(PrimitiveArray::from_iter([f64::NAN, f64::NAN]).into_array(), 1)]
#[case::nan_payloads(
    PrimitiveArray::from_iter([f64::NAN, f64::from_bits(f64::NAN.to_bits() ^ 1)]).into_array(),
    2
)]
#[case::infinities(
    PrimitiveArray::from_iter([f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY]).into_array(),
    2
)]
#[case::integer_extremes(PrimitiveArray::from_iter([i64::MIN, i64::MAX, 0, -1]).into_array(), 4)]
#[case::utf8_inlined(VarBinViewArray::from_iter_str(["a", "b", "a"]).into_array(), 2)]
#[case::utf8_long(VarBinViewArray::from_iter_str([LONG, LONG, "short"]).into_array(), 2)]
#[case::utf8_twelve_and_thirteen_bytes(
    VarBinViewArray::from_iter_str(["twelve bytes", "thirteen byte", "twelve bytes"]).into_array(),
    2
)]
#[case::utf8_empty_string_is_a_value(VarBinViewArray::from_iter_str(["", "", "x"]).into_array(), 2)]
#[case::utf8_with_nulls(
    VarBinViewArray::from_iter_nullable_str([Some("a"), None, Some(LONG), None, Some("a")])
        .into_array(),
    2
)]
#[case::binary(
    VarBinViewArray::from_iter_bin([&[0u8, 1][..], &[], &[0, 1], &[255; 20]]).into_array(),
    3
)]
#[case::dictionary_ignores_unreferenced_values(dictionary_with_unused_value(), 2)]
#[case::slice(slice_of_four(), 4)]
#[case::extension(
    ExtensionArray::new(
        Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        PrimitiveArray::from_iter([1_000i64, 2_000, 1_000]).into_array(),
    )
    .into_array(),
    2
)]
fn test_distinct_count(#[case] array: ArrayRef, #[case] expected: u64) -> VortexResult<()> {
    assert_eq!(estimate(&array)?, expected);
    Ok(())
}

#[rstest]
#[case::u8(PrimitiveArray::from_iter(0u8..100))]
#[case::u16(PrimitiveArray::from_iter(0u16..100))]
#[case::u32(PrimitiveArray::from_iter(0u32..100))]
#[case::u64(PrimitiveArray::from_iter(0u64..100))]
#[case::i8(PrimitiveArray::from_iter(-50i8..50))]
#[case::i16(PrimitiveArray::from_iter(-50i16..50))]
#[case::i32(PrimitiveArray::from_iter(-50i32..50))]
#[case::i64(PrimitiveArray::from_iter(-50i64..50))]
#[case::f16(PrimitiveArray::from_iter((0..100u8).map(f16::from)))]
#[case::f32(PrimitiveArray::from_iter((0..100u8).map(f32::from)))]
#[case::f64(PrimitiveArray::from_iter((0..100u8).map(f64::from)))]
fn test_every_primitive_type(#[case] array: PrimitiveArray) -> VortexResult<()> {
    // 100 values can share registers, so check three standard errors (1.04 / sqrt(4096)).
    let estimate = estimate(&array.into_array())?;
    assert!(
        (95..=105).contains(&estimate),
        "estimate {estimate} for 100 values"
    );
    Ok(())
}

#[rstest]
#[case::values_hidden_by_nulls(
    PrimitiveArray::new(buffer![1i32, 100, 2, 200], Validity::from_iter([true, false, true, false]))
        .into_array(),
    PrimitiveArray::from_iter([1i32, 2]).into_array()
)]
#[case::utf8_and_binary_with_same_bytes(
    VarBinViewArray::from_iter_str(["same bytes"]).into_array(),
    VarBinViewArray::from_iter_bin([b"same bytes"]).into_array()
)]
#[case::extension_and_its_storage(
    ExtensionArray::new(
        Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        PrimitiveArray::from_iter([1_000i64, 2_000]).into_array(),
    )
    .into_array(),
    PrimitiveArray::from_iter([1_000i64, 2_000]).into_array()
)]
#[case::chunked_and_whole(
    chunked_zero_to_thousand(),
    PrimitiveArray::from_iter(0i64..1_000).into_array()
)]
fn test_same_values_build_the_same_sketch(
    #[case] left: ArrayRef,
    #[case] right: ArrayRef,
) -> VortexResult<()> {
    assert_eq!(sketch_of(&left)?, sketch_of(&right)?);
    Ok(())
}

#[rstest]
#[case::primitive(Scalar::primitive(7i32, Nullability::NonNullable))]
#[case::negative_zero(Scalar::primitive(-0.0f64, Nullability::NonNullable))]
#[case::bool_true(Scalar::bool(true, Nullability::NonNullable))]
#[case::bool_false(Scalar::bool(false, Nullability::NonNullable))]
#[case::short_utf8(Scalar::utf8("short", Nullability::NonNullable))]
#[case::long_utf8(Scalar::utf8(LONG, Nullability::NonNullable))]
#[case::binary(Scalar::binary(vec![1u8, 2, 3], Nullability::NonNullable))]
#[case::extension(Scalar::extension_ref(
    Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
    Scalar::primitive(1_000i64, Nullability::NonNullable),
))]
fn test_constant_matches_its_canonical_form(#[case] scalar: Scalar) -> VortexResult<()> {
    // A column can be constant in one chunk and canonical in another; both must hash alike.
    let constant = ConstantArray::new(scalar, 10).into_array();
    let canonical = constant
        .clone()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
        .into_array();
    assert_eq!(sketch_of(&constant)?, sketch_of(&canonical)?);
    assert_eq!(estimate(&constant)?, 1);
    Ok(())
}

#[test]
fn test_batches_merge_regardless_of_order_and_overlap() -> VortexResult<()> {
    let union = sketch_of(&PrimitiveArray::from_iter(0i64..1_000).into_array())?;
    let forward = sketch_of_batches(&[
        PrimitiveArray::from_iter(0i64..600).into_array(),
        PrimitiveArray::from_iter(400i64..1_000).into_array(),
    ])?;
    let reversed = sketch_of_batches(&[
        PrimitiveArray::from_iter((400i64..1_000).rev()).into_array(),
        PrimitiveArray::from_iter((0i64..600).rev()).into_array(),
    ])?;
    assert_eq!(forward, union);
    assert_eq!(reversed, union);
    Ok(())
}

#[test]
fn test_combining_a_partial_twice_does_not_double_count() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let options = ApproxDistinctOptions::default();
    let mut source = Accumulator::try_new(ApproxDistinct, options, DType::from(PType::I64))?;
    source.accumulate(
        &PrimitiveArray::from_iter(0i64..1_000).into_array(),
        &mut ctx,
    )?;
    let partial = source.partial_scalar()?;

    let mut combined = Accumulator::try_new(ApproxDistinct, options, DType::from(PType::I64))?;
    combined.combine_partials(&partial)?;
    combined.combine_partials(&partial)?;
    assert_eq!(combined.final_scalar()?, source.final_scalar()?);
    Ok(())
}

#[test]
fn test_partial_from_scalar() -> VortexResult<()> {
    let options = ApproxDistinctOptions::default();
    let dtypes = i64_dtypes()?;

    let from_null = ApproxDistinct.partial_from_scalar(
        dtypes.args(&options),
        &Scalar::null(DType::Binary(Nullability::Nullable)),
    )?;
    assert_eq!(from_null, HyperLogLog::try_new(DEFAULT_PRECISION)?);

    let other_precision = Scalar::binary(
        HyperLogLog::try_new(DEFAULT_PRECISION + 1)?.serialize(),
        Nullability::NonNullable,
    );
    assert!(
        ApproxDistinct
            .partial_from_scalar(dtypes.args(&options), &other_precision)
            .is_err()
    );
    Ok(())
}

#[test]
fn test_is_never_saturated() -> VortexResult<()> {
    let options = ApproxDistinctOptions::default();
    let dtypes = i64_dtypes()?;
    let mut sketch = HyperLogLog::try_new(DEFAULT_PRECISION)?;
    for index in 0..(1u64 << DEFAULT_PRECISION) {
        // A hash with every bit after the index zero puts the register at its largest rank.
        sketch.insert_hash(index << (64 - u32::from(DEFAULT_PRECISION)));
    }
    assert!(!ApproxDistinct.is_saturated(dtypes.args(&options), &sketch));
    Ok(())
}

#[test]
fn test_grouped_finalize_is_unsupported() -> VortexResult<()> {
    let options = ApproxDistinctOptions::default();
    let states = VarBinViewArray::from_iter_bin([HyperLogLog::try_new(12)?.serialize()]);
    assert!(
        ApproxDistinct
            .finalize(i64_dtypes()?.args(&options), states.into_array())
            .is_err()
    );
    Ok(())
}
