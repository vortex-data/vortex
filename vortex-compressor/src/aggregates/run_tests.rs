// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Contract tests for ordered runs and physical view-prefix summaries.

use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateDTypes;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::RunSummary;
use super::VarBinViewPrefixDistinct;

/// Accumulate an input and parse its partial scalar through the aggregate contract.
fn partial_for<V: AggregateFnVTable<Options = EmptyOptions>>(
    function: V,
    array: ArrayRef,
) -> VortexResult<V::Partial> {
    let dtypes = AggregateDTypes::try_new(&function, &EmptyOptions, array.dtype().clone())?;
    let mut accumulator =
        Accumulator::try_new(function.clone(), EmptyOptions, array.dtype().clone())?;
    let mut ctx = array_session().create_execution_ctx();
    accumulator.accumulate(&array, &mut ctx)?;

    function.partial_from_scalar(dtypes.args(&EmptyOptions), accumulator.flush()?)
}

#[rstest]
#[case::empty(vec![], 0, 0, 0)]
#[case::all_null(vec![None, None], 0, 0, 0)]
#[case::one_run(vec![Some(1.0), Some(1.0)], 2, 1, 2)]
#[case::null_gaps(vec![None, Some(1.0), None, Some(1.0), Some(2.0), None], 3, 2, 1)]
#[case::signed_zeros(vec![Some(-0.0), Some(0.0)], 2, 1, 2)]
#[case::first_nan(vec![Some(f64::NAN), Some(1.0), Some(1.0), Some(1.0)], 4, 2, 1)]
#[case::single_nan(vec![Some(f64::NAN)], 1, 1, 0)]
#[case::repeated_nans(vec![Some(1.0), Some(f64::NAN), None, Some(f64::NAN)], 3, 3, 1)]
fn runs_skip_nulls_with_numeric_equality(
    #[case] values: Vec<Option<f64>>,
    #[case] valid_count: u64,
    #[case] runs: u64,
    #[case] average: u32,
) -> VortexResult<()> {
    let partial = partial_for(
        RunSummary,
        PrimitiveArray::from_option_iter(values).into_array(),
    )?;

    assert_eq!(partial.valid_count(), valid_count);
    assert_eq!(partial.run_count(), runs);
    assert_eq!(partial.average_run_length_for_compression()?, average);
    Ok(())
}

#[rstest]
#[case::u8(PType::U8)]
#[case::u16(PType::U16)]
#[case::u32(PType::U32)]
#[case::u64(PType::U64)]
#[case::i8(PType::I8)]
#[case::i16(PType::I16)]
#[case::i32(PType::I32)]
#[case::i64(PType::I64)]
#[case::f16(PType::F16)]
#[case::f32(PType::F32)]
#[case::f64(PType::F64)]
fn native_runs_and_partial_round_trip(#[case] ptype: PType) -> VortexResult<()> {
    match_each_native_ptype!(ptype, |T| {
        let one = PValue::U8(1).cast::<T>()?;
        let two = PValue::U8(2).cast::<T>()?;
        let array = PrimitiveArray::from_option_iter([Some(one), None, Some(one), Some(two)]);
        let dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, array.dtype().clone())?;
        let partial = partial_for(RunSummary, array.into_array())?;
        let scalar = RunSummary.to_scalar(dtypes.args(&EmptyOptions), &partial)?;
        let restored =
            RunSummary.partial_from_scalar(dtypes.args(&EmptyOptions), scalar.clone())?;

        assert_eq!(partial.valid_count(), 3);
        assert_eq!(partial.run_count(), 2);
        assert_eq!(
            RunSummary.to_scalar(dtypes.args(&EmptyOptions), &restored)?,
            scalar
        );
        Ok(())
    })
}

/// Count runs from a compacted list of valid values, independently of chunked scanning.
fn reference_run_counts<T: Copy + PartialEq>(
    values: &[T],
    validity: &[bool],
) -> VortexResult<(u64, u64)> {
    let valid_values: Vec<T> = values
        .iter()
        .zip(validity)
        .filter_map(|(&value, &valid)| valid.then_some(value))
        .collect();
    let count = u64::try_from(valid_values.len())?;
    let transitions = valid_values
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    let runs = u64::try_from(transitions)? + u64::from(!valid_values.is_empty());

    Ok((count, runs))
}

#[rstest]
fn tuned_integer_runs_match_reference(
    #[values(0, 1, 63, 64, 65, 1297)] len: u32,
    #[values(1, 3, 64, 100)] run_len: u32,
    #[values(None, Some(3), Some(97))] null_every: Option<u32>,
) -> VortexResult<()> {
    let values: Vec<u32> = (0..len).map(|index| (index / run_len) % 16).collect();
    let valid: Vec<bool> = (0..len)
        .map(|index| null_every.is_none_or(|interval| index % interval != 0))
        .collect();
    let expected = reference_run_counts(&values, &valid)?;
    let validity = match null_every {
        None => Validity::NonNullable,
        Some(_) => Validity::from(BitBuffer::from(valid)),
    };
    let partial = partial_for(
        RunSummary,
        PrimitiveArray::new(Buffer::from(values), validity).into_array(),
    )?;

    assert_eq!((partial.valid_count(), partial.run_count()), expected);
    let average = u32::try_from(expected.0.checked_div(expected.1).unwrap_or(0))?;
    assert_eq!(partial.average_run_length_for_compression()?, average);
    Ok(())
}

#[rstest]
#[case::i8(PType::I8)]
#[case::i16(PType::I16)]
#[case::i32(PType::I32)]
#[case::i64(PType::I64)]
fn tuned_signed_extremes_match_reference(
    #[case] ptype: PType,
    #[values(false, true)] null_middle_chunk: bool,
) -> VortexResult<()> {
    match_each_integer_ptype!(ptype, |T| {
        let values: Vec<T> = (0..257)
            .map(|index| {
                if index < 64 || (128..192).contains(&index) {
                    T::MIN
                } else {
                    T::MAX
                }
            })
            .collect();
        let valid: Vec<bool> = (0..257)
            .map(|index| !null_middle_chunk || !(64..128).contains(&index))
            .collect();
        let expected = reference_run_counts(&values, &valid)?;
        let validity = if null_middle_chunk {
            Validity::from(BitBuffer::from(valid))
        } else {
            Validity::NonNullable
        };
        let partial = partial_for(
            RunSummary,
            PrimitiveArray::new(Buffer::from(values), validity).into_array(),
        )?;

        assert_eq!((partial.valid_count(), partial.run_count()), expected);
        Ok(())
    })
}

#[rstest]
#[case::f16(PType::F16)]
#[case::f32(PType::F32)]
#[case::f64(PType::F64)]
fn float_boundaries_use_numeric_equality(#[case] ptype: PType) -> VortexResult<()> {
    match_each_native_ptype!(ptype, |T| {
        let negative_zero = PValue::F64(-0.0).cast::<T>()?;
        let positive_zero = PValue::F64(0.0).cast::<T>()?;
        let nan = PValue::F64(f64::NAN).cast::<T>()?;
        let first = PrimitiveArray::from_option_iter([Some(negative_zero), None]);
        let second = PrimitiveArray::from_option_iter([Some(positive_zero), Some(nan)]);
        let third = PrimitiveArray::from_option_iter([None, Some(nan)]);
        let dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, first.dtype().clone())?;
        let args = dtypes.args(&EmptyOptions);
        let first = partial_for(RunSummary, first.into_array())?;
        let second = partial_for(RunSummary, second.into_array())?;
        let third = partial_for(RunSummary, third.into_array())?;
        let first_two = RunSummary.merge_partials(args, first, second)?;
        let merged = RunSummary.merge_partials(args, first_two, third)?;

        assert_eq!(merged.valid_count(), 4);
        assert_eq!(merged.run_count(), 3);
        assert!(!merged.first_is_nan());
        Ok(())
    })
}

#[test]
fn ordered_run_merges_preserve_boundaries_and_first_nan_once() -> VortexResult<()> {
    let values = [
        None,
        Some(f64::NAN),
        Some(1.0),
        None,
        None,
        Some(1.0),
        Some(-0.0),
        Some(0.0),
    ];
    let array = PrimitiveArray::from_option_iter(values);
    let dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let whole = partial_for(RunSummary, array.into_array())?;
    let expected = RunSummary.to_scalar(args, &whole)?;

    for first_end in 0..=values.len() {
        for second_end in first_end..=values.len() {
            let first = partial_for(
                RunSummary,
                PrimitiveArray::from_option_iter(values[..first_end].iter().copied()).into_array(),
            )?;
            let middle = partial_for(
                RunSummary,
                PrimitiveArray::from_option_iter(values[first_end..second_end].iter().copied())
                    .into_array(),
            )?;
            let last = partial_for(
                RunSummary,
                PrimitiveArray::from_option_iter(values[second_end..].iter().copied()).into_array(),
            )?;
            let left = RunSummary.merge_partials(args, first.clone(), middle.clone())?;
            let left = RunSummary.merge_partials(args, left, last.clone())?;
            let right = RunSummary.merge_partials(args, middle, last)?;
            let right = RunSummary.merge_partials(args, first, right)?;

            assert_eq!(RunSummary.to_scalar(args, &left)?, expected);
            assert_eq!(RunSummary.to_scalar(args, &right)?, expected);
            assert_eq!(
                left.average_run_length_for_compression()?,
                whole.average_run_length_for_compression()?
            );
        }
    }
    Ok(())
}

#[test]
fn float_endpoint_round_trip_preserves_nan_payload_and_zero_sign() -> VortexResult<()> {
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    let array = PrimitiveArray::from_option_iter([Some(nan), None, Some(-0.0)]);
    let dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let partial = partial_for(RunSummary, array.into_array())?;
    let scalar = RunSummary.to_scalar(args, &partial)?;
    let restored = RunSummary.partial_from_scalar(args, scalar.clone())?;

    assert_eq!(RunSummary.to_scalar(args, &restored)?, scalar);
    assert_eq!(
        scalar
            .as_struct()
            .field_by_idx(2)
            .unwrap()
            .as_primitive()
            .typed_value::<f64>()
            .unwrap()
            .to_bits(),
        nan.to_bits()
    );
    assert_eq!(
        scalar
            .as_struct()
            .field_by_idx(3)
            .unwrap()
            .as_primitive()
            .typed_value::<f64>()
            .unwrap()
            .to_bits(),
        (-0.0f64).to_bits()
    );
    Ok(())
}

#[rstest]
#[case::number(Scalar::primitive(7i32, NonNullable), 8, 1)]
#[case::nan(Scalar::primitive(f64::NAN, NonNullable), 8, 8)]
#[case::empty(Scalar::primitive(f64::NAN, NonNullable), 0, 0)]
#[case::null(Scalar::null(DType::Primitive(PType::F64, Nullable)), 8, 0)]
fn constant_runs(#[case] value: Scalar, #[case] len: usize, #[case] runs: u64) -> VortexResult<()> {
    let partial = partial_for(RunSummary, ConstantArray::new(value, len).into_array())?;
    assert_eq!(partial.run_count(), runs);
    Ok(())
}

#[test]
fn grouped_run_finalization_uses_logical_runs() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([Some(f64::NAN), Some(1.0), Some(1.0)]);
    let dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let partial = partial_for(RunSummary, array.into_array())?;
    let states = ConstantArray::new(RunSummary.to_scalar(args, &partial)?, 2).into_array();
    let results = RunSummary.finalize(args, states)?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(
        results.execute_scalar(0, &mut ctx)?,
        Scalar::primitive(2u64, NonNullable)
    );
    assert_eq!(
        results.execute_scalar(1, &mut ctx)?,
        Scalar::primitive(2u64, NonNullable)
    );
    Ok(())
}

#[test]
fn run_summary_rejects_inconsistent_partial_counts() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullable);
    let dtypes = AggregateDTypes::try_new(&RunSummary, &EmptyOptions, dtype)?;
    let invalid = Scalar::struct_(
        dtypes.partial_dtype.clone(),
        [
            Scalar::primitive(1u64, NonNullable),
            Scalar::primitive(1u64, NonNullable),
            Scalar::primitive(7i32, Nullable),
            Scalar::primitive(7i32, Nullable),
        ],
    );

    assert!(
        RunSummary
            .partial_from_scalar(dtypes.args(&EmptyOptions), invalid)
            .is_err()
    );
    Ok(())
}

#[test]
fn run_average_preserves_compressor_count_limit() -> VortexResult<()> {
    let len = usize::try_from(u64::from(u32::MAX) + 1)?;
    let array = ConstantArray::new(1i32, len).into_array();
    let partial = partial_for(RunSummary, array)?;

    assert_eq!(partial.valid_count(), u64::from(u32::MAX) + 1);
    assert_eq!(partial.run_count(), 1);
    assert!(partial.average_run_length_for_compression().is_err());
    Ok(())
}

#[test]
fn physical_prefix_counts_invalid_views_without_normalizing() -> VortexResult<()> {
    let views = buffer![
        BinaryView::new_inlined(b"first"),
        BinaryView::new_inlined(b"other")
    ];
    let array = VarBinViewArray::new_handle(
        BufferHandle::new_host(views.into_byte_buffer()),
        Arc::from([]),
        DType::Binary(Nullable),
        Validity::AllInvalid,
    );
    let partial = partial_for(VarBinViewPrefixDistinct, array.into_array())?;

    assert_eq!(partial.len(), 2);
    Ok(())
}

#[rstest]
#[case::empty(vec![], 0)]
#[case::all_null(vec![None, None], 1)]
#[case::prefix_collision(vec![Some("sameaaa"), Some("samebbb")], 1)]
#[case::invalid_slot(vec![Some("sameaaa"), Some("samebbb"), None], 2)]
fn physical_prefix_counts_view_fingerprints(
    #[case] values: Vec<Option<&str>>,
    #[case] distinct: usize,
) -> VortexResult<()> {
    let array = VarBinViewArray::from_iter_nullable_str(values);
    let dtypes = AggregateDTypes::try_new(
        &VarBinViewPrefixDistinct,
        &EmptyOptions,
        array.dtype().clone(),
    )?;
    let args = dtypes.args(&EmptyOptions);
    let partial = partial_for(VarBinViewPrefixDistinct, array.into_array())?;
    let scalar = VarBinViewPrefixDistinct.to_scalar(args, &partial)?;
    let restored = VarBinViewPrefixDistinct.partial_from_scalar(args, scalar)?;
    let merged = VarBinViewPrefixDistinct.merge_partials(args, partial.clone(), restored)?;
    let states =
        ConstantArray::new(VarBinViewPrefixDistinct.to_scalar(args, &merged)?, 1).into_array();
    let results = VarBinViewPrefixDistinct.finalize(args, states)?;
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(partial.len(), distinct);
    assert_eq!(merged, partial);
    assert_eq!(
        results.execute_scalar(0, &mut ctx)?,
        Scalar::primitive(u32::try_from(distinct)?, NonNullable)
    );
    assert_eq!(VarBinViewPrefixDistinct.serialize(&EmptyOptions)?, None);
    Ok(())
}

#[test]
fn physical_prefix_rejects_noncanonical_input() -> VortexResult<()> {
    let array = ConstantArray::new("one", 4).into_array();
    assert!(partial_for(VarBinViewPrefixDistinct, array).is_err());
    Ok(())
}
