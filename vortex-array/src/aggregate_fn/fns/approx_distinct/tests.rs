// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::*;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::DynAccumulator;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::builders::dict::dict_encode;
use crate::dtype::DecimalDType;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// The estimate and the partial sketch of `array`.
fn approx_distinct(array: &ArrayRef) -> VortexResult<(u64, HllSketch)> {
    let mut ctx = SESSION.create_execution_ctx();
    let options = ApproxDistinctOptions::default();
    let mut accumulator = Accumulator::try_new(ApproxDistinct, options, array.dtype().clone())?;
    accumulator.accumulate(array, &mut ctx)?;
    let sketch = ApproxDistinct::sketch_from_partial(&options, &accumulator.partial_scalar()?)?;
    let estimate = accumulator
        .finish()?
        .as_primitive()
        .typed_value::<u64>()
        .ok_or_else(|| vortex_err!("approx_distinct result is non-null"))?;
    Ok((estimate, sketch))
}

#[test]
fn options_roundtrip() -> VortexResult<()> {
    let options = ApproxDistinctOptions::try_new(14)?;
    let metadata = ApproxDistinct
        .serialize(&options)?
        .ok_or_else(|| vortex_err!("serializable"))?;
    assert_eq!(metadata, [14]);
    assert_eq!(ApproxDistinct.deserialize(&metadata, &SESSION)?, options);
    assert!(ApproxDistinct.deserialize(&[2], &SESSION).is_err());
    assert!(ApproxDistinct.deserialize(&[], &SESSION).is_err());
    Ok(())
}

#[rstest]
#[case::struct_(DType::struct_([("a", DType::Bool(Nullability::NonNullable))], Nullability::NonNullable))]
#[case::null(DType::Null)]
fn unsupported_dtypes(#[case] dtype: DType) {
    let options = ApproxDistinctOptions::default();
    assert!(ApproxDistinct.return_dtype(&options, &dtype).is_none());
    assert!(ApproxDistinct.partial_dtype(&options, &dtype).is_none());
}

#[test]
fn nulls_are_not_counted() -> VortexResult<()> {
    let array =
        PrimitiveArray::from_option_iter((0..10_000i64).map(|i| (i % 3 != 0).then_some(i % 500)))
            .into_array();
    let (estimate, _) = approx_distinct(&array)?;
    // Every residue modulo 500 also appears in a valid row.
    assert!(estimate.abs_diff(500) <= 10, "estimate {estimate}");

    let all_null =
        PrimitiveArray::new(Buffer::from_iter(0..100i32), Validity::AllInvalid).into_array();
    assert_eq!(approx_distinct(&all_null)?.0, 0);
    Ok(())
}

#[rstest]
#[case(1_000)]
#[case(100_000)]
fn integers_within_expected_error(#[case] distinct: i64) -> VortexResult<()> {
    let array = PrimitiveArray::from_iter((0..2 * distinct).map(|i| i % distinct)).into_array();
    let (estimate, _) = approx_distinct(&array)?;
    let error = estimate.abs_diff(distinct.unsigned_abs()) as f64 / distinct as f64;
    assert!(error < 0.05, "estimate {estimate} for {distinct}");
    Ok(())
}

#[test]
fn strings_inlined_and_buffered() -> VortexResult<()> {
    let values = (0..3_000).map(|i| {
        if i % 2 == 0 {
            format!("s{}", i % 1_000)
        } else {
            format!("a string longer than the inline limit {}", i % 1_000)
        }
    });
    let array = VarBinViewArray::from_iter_str(values).into_array();
    let (estimate, _) = approx_distinct(&array)?;
    assert!(estimate.abs_diff(1_000) <= 30, "estimate {estimate}");
    Ok(())
}

#[test]
fn bools_count_at_most_two() -> VortexResult<()> {
    let array = BoolArray::from_iter([true, false, true]).into_array();
    assert_eq!(approx_distinct(&array)?.0, 2);
    let array = BoolArray::from_iter([Some(true), None, Some(true)]).into_array();
    assert_eq!(approx_distinct(&array)?.0, 1);
    Ok(())
}

#[test]
fn signed_zeros_and_nans_are_one_value_each() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter([0.0f64, -0.0, f64::NAN, -f64::NAN, 1.0]).into_array();
    assert_eq!(approx_distinct(&array)?.0, 3);
    Ok(())
}

/// Decimals hash by their unscaled value, so chunks narrowed to different widths agree.
#[test]
fn decimals_hash_independently_of_width() -> VortexResult<()> {
    let dtype = DecimalDType::new(10, 2);
    let narrow = DecimalArray::new(Buffer::from_iter(0..1_000i32), dtype, Validity::NonNullable)
        .into_array();
    let wide = DecimalArray::new(
        Buffer::from_iter((0..1_000).map(i128::from)),
        dtype,
        Validity::NonNullable,
    )
    .into_array();
    let (narrow_estimate, narrow_sketch) = approx_distinct(&narrow)?;
    assert_eq!(narrow_sketch, approx_distinct(&wide)?.1);
    assert!(narrow_estimate.abs_diff(1_000) <= 30);
    Ok(())
}

/// Constants, dictionaries and chunks give the sketch of their canonical form.
#[test]
fn encodings_match_canonical() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = PrimitiveArray::from_iter((0..5_000i32).map(|i| i % 700)).into_array();
    let (_, expected) = approx_distinct(&canonical)?;

    let dict = dict_encode(&canonical, &mut ctx)?;
    assert!(dict.has_all_values_referenced());
    assert_eq!(approx_distinct(&dict.into_array())?.1, expected);

    let chunked =
        ChunkedArray::from_iter([canonical.slice(0..2_000)?, canonical.slice(2_000..5_000)?])
            .into_array();
    assert_eq!(approx_distinct(&chunked)?.1, expected);

    let constant = ConstantArray::new(7i32, 100).into_array();
    let single = PrimitiveArray::from_iter([7i32]).into_array();
    assert_eq!(approx_distinct(&constant)?, approx_distinct(&single)?);
    Ok(())
}

#[test]
fn partials_merge_into_union() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let options = ApproxDistinctOptions::default();
    let mut left = Accumulator::try_new(ApproxDistinct, options, dtype.clone())?;
    left.accumulate(
        &PrimitiveArray::from_iter(0..30_000i64).into_array(),
        &mut ctx,
    )?;
    let mut right = Accumulator::try_new(ApproxDistinct, options, dtype)?;
    right.accumulate(
        &PrimitiveArray::from_iter(20_000..50_000i64).into_array(),
        &mut ctx,
    )?;
    left.combine_partials(&right.partial_scalar()?)?;
    let estimate = left
        .finish()?
        .as_primitive()
        .typed_value::<u64>()
        .ok_or_else(|| vortex_err!("non-null"))?;
    assert!(estimate.abs_diff(50_000) <= 2_500, "estimate {estimate}");
    Ok(())
}
