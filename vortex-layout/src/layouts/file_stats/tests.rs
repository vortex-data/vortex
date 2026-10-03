// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
#[cfg(target_pointer_width = "64")]
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
#[cfg(target_pointer_width = "64")]
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::stats::AggregateResults;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::field::FieldAccumulator;

fn accumulate(
    dtype: &DType,
    aggregates: &[AggregateFnRef],
    chunks: &[ArrayRef],
    max_length: usize,
) -> VortexResult<AggregateResults> {
    let mut ctx = array_session().create_execution_ctx();
    let mut accumulator = FieldAccumulator::new(dtype, aggregates, max_length)?;
    for chunk in chunks {
        accumulator.push_chunk(chunk, &mut ctx)?;
    }
    accumulator.results()
}

#[rstest]
#[case::overflow_then_valid(vec![vec![i64::MAX, 1], vec![5]])]
#[case::overflow_then_zero(vec![vec![i64::MAX, 1], vec![0]])]
#[case::all_overflow(vec![vec![i64::MAX, 1], vec![i64::MAX, 1]])]
#[case::merged_overflow(vec![vec![i64::MAX], vec![1]])]
fn sum_retains_overflow(
    #[case] chunks: Vec<Vec<i64>>,
    #[values(false, true)] cached: bool,
) -> VortexResult<()> {
    let chunks = chunks
        .into_iter()
        .map(|values| PrimitiveArray::from_iter(values).into_array())
        .collect::<Vec<_>>();
    let mut ctx = array_session().create_execution_ctx();
    if cached {
        for chunk in &chunks {
            sum(chunk, &mut ctx)?;
        }
    }
    let request = Sum.bind(NumericalAggregateOpts::skip_nans());
    let results = accumulate(
        chunks[0].dtype(),
        std::slice::from_ref(&request),
        &chunks,
        64,
    )?;
    assert_eq!(
        results.get_result(&request),
        Precision::Exact(Scalar::null(DType::Primitive(
            PType::I64,
            Nullability::Nullable
        )))
    );
    Ok(())
}

#[test]
#[cfg(target_pointer_width = "64")]
fn decimal_sum_overflows_at_its_return_precision() -> VortexResult<()> {
    let value = Scalar::decimal(
        DecimalValue::I8(9),
        DecimalDType::new(1, 0),
        Nullability::NonNullable,
    );
    let chunks = [
        ConstantArray::new(value.clone(), 10_000_000_000).into_array(),
        ConstantArray::new(value, 10_000_000_000).into_array(),
    ];
    let request = Sum.bind(NumericalAggregateOpts::skip_nans());
    let results = accumulate(
        chunks[0].dtype(),
        std::slice::from_ref(&request),
        &chunks,
        64,
    )?;
    for chunk in &chunks {
        assert_eq!(
            chunk.statistics().get(Stat::Sum),
            Precision::Exact(Scalar::decimal(
                DecimalValue::I128(90_000_000_000),
                DecimalDType::new(11, 0),
                Nullability::Nullable
            ))
        );
    }
    assert_eq!(
        results.get_result(&request),
        Precision::Exact(Scalar::null(DType::Decimal(
            DecimalDType::new(11, 0),
            Nullability::Nullable
        )))
    );
    Ok(())
}

#[rstest]
#[case::non_null(vec![3, 4], Some(7))]
#[case::overflow(vec![i64::MAX, 1], None)]
fn sum_preserves_chunk_cache_publication(
    #[case] values: Vec<i64>,
    #[case] expected: Option<i64>,
) -> VortexResult<()> {
    let array = PrimitiveArray::from_iter(values).into_array();
    accumulate(
        array.dtype(),
        &[Sum.bind(NumericalAggregateOpts::skip_nans())],
        std::slice::from_ref(&array),
        64,
    )?;
    assert_eq!(
        array.statistics().get(Stat::Sum),
        expected.map_or(Precision::Absent, |value| Precision::Exact(
            Scalar::primitive(value, Nullability::Nullable)
        ))
    );
    Ok(())
}

#[rstest]
#[case::arithmetic_nan_then_finite(vec![1.0], 1.0)]
#[case::all_nan_finals(vec![f64::INFINITY, f64::NEG_INFINITY], 0.0)]
fn sum_skips_nan_chunk_finals(
    #[case] second: Vec<f64>,
    #[case] expected: f64,
    #[values(false, true)] cached: bool,
) -> VortexResult<()> {
    let chunks = [
        PrimitiveArray::from_iter([f64::INFINITY, f64::NEG_INFINITY]).into_array(),
        PrimitiveArray::from_iter(second).into_array(),
    ];
    let mut ctx = array_session().create_execution_ctx();
    if cached {
        for chunk in &chunks {
            sum(chunk, &mut ctx)?;
        }
    }
    let request = Sum.bind(NumericalAggregateOpts::skip_nans());
    let results = accumulate(
        chunks[0].dtype(),
        std::slice::from_ref(&request),
        &chunks,
        64,
    )?;
    assert!(
        chunks[0]
            .statistics()
            .get(Stat::Sum)
            .as_exact()
            .is_some_and(|value| value.as_primitive().is_nan())
    );
    assert_eq!(
        results.get_result(&request),
        Precision::Exact(Scalar::primitive(expected, Nullability::Nullable))
    );
    Ok(())
}

#[test]
fn sum_keeps_nan_from_merging_chunk_finals() -> VortexResult<()> {
    let chunks = [
        PrimitiveArray::from_iter([f64::INFINITY]).into_array(),
        PrimitiveArray::from_iter([f64::NEG_INFINITY]).into_array(),
    ];
    let request = Sum.bind(NumericalAggregateOpts::skip_nans());
    let results = accumulate(
        chunks[0].dtype(),
        std::slice::from_ref(&request),
        &chunks,
        64,
    )?;
    assert!(
        results
            .get_result(&request)
            .as_exact()
            .is_some_and(|value| value.as_primitive().is_nan())
    );
    Ok(())
}

#[test]
fn sum_keeps_chunk_float_grouping() -> VortexResult<()> {
    let chunks = [
        PrimitiveArray::from_iter([1e16f64]).into_array(),
        PrimitiveArray::from_iter([-1e16f64, 1.0]).into_array(),
    ];
    let request = Sum.bind(NumericalAggregateOpts::skip_nans());
    let results = accumulate(
        chunks[0].dtype(),
        std::slice::from_ref(&request),
        &chunks,
        64,
    )?;
    assert_eq!(
        results.get_result(&request),
        Precision::Exact(Scalar::primitive(0.0f64, Nullability::Nullable))
    );
    Ok(())
}

#[rstest]
#[case::no_chunks(vec![], None)]
#[case::empty_chunk(vec![vec![]], Some(0))]
#[case::all_null_chunk(vec![vec![None, None]], Some(0))]
#[case::nullable_chunks(vec![vec![None, Some(3)], vec![Some(4), None]], Some(7))]
fn sum_distinguishes_missing_input_from_zero(
    #[case] chunks: Vec<Vec<Option<i64>>>,
    #[case] expected: Option<i64>,
) -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    let chunks = chunks
        .into_iter()
        .map(|values| PrimitiveArray::from_option_iter(values).into_array())
        .collect::<Vec<_>>();
    let request = Sum.bind(NumericalAggregateOpts::skip_nans());
    let results = accumulate(&dtype, std::slice::from_ref(&request), &chunks, 64)?;
    assert_eq!(
        results.get_result(&request),
        expected.map_or(Precision::Absent, |value| Precision::Exact(
            Scalar::primitive(value, Nullability::Nullable)
        ))
    );
    Ok(())
}

#[rstest]
#[case::primitive(PrimitiveArray::empty::<f64>(Nullability::NonNullable).into_array())]
#[case::constant(ConstantArray::new(Scalar::from(42f64), 0).into_array())]
fn empty_chunk_preserves_counts(#[case] array: ArrayRef) -> VortexResult<()> {
    let requests = [
        NullCount.bind(EmptyOptions),
        NanCount.bind(EmptyOptions),
        UncompressedSizeInBytes.bind(EmptyOptions),
        Sum.bind(NumericalAggregateOpts::skip_nans()),
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
        IsConstant.bind(EmptyOptions),
        IsSorted.bind(IsSortedOptions { strict: false }),
    ];
    let results = accumulate(array.dtype(), &requests, std::slice::from_ref(&array), 64)?;
    assert_eq!(
        results.get_result(&requests[0]),
        Precision::Exact(Scalar::from(0u64))
    );
    assert_eq!(
        results.get_result(&requests[1]),
        Precision::Exact(Scalar::from(0u64))
    );
    assert_eq!(
        results.get_result(&requests[2]),
        Precision::Exact(Scalar::from(0u64))
    );
    assert_eq!(
        results.get_result(&requests[3]),
        Precision::Exact(Scalar::primitive(0f64, Nullability::Nullable))
    );
    for request in &requests[4..] {
        assert!(results.get_result(request).is_absent());
    }
    for stat in [Stat::Min, Stat::Max, Stat::IsConstant, Stat::IsSorted] {
        assert!(array.statistics().get(stat).is_absent());
    }
    Ok(())
}

#[rstest]
#[case::utf8(DType::Utf8(Nullability::NonNullable))]
#[case::binary(DType::Binary(Nullability::NonNullable))]
fn only_final_extrema_are_truncated(#[case] dtype: DType) -> VortexResult<()> {
    let chunks = [
        VarBinArray::from_vec(vec![b"middle long value".to_vec()], dtype.clone()).into_array(),
        VarBinArray::from_vec(vec![b"a".to_vec(), b"z".to_vec()], dtype.clone()).into_array(),
    ];
    let requests = [
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
    ];
    let results = accumulate(&dtype, &requests, &chunks, 2)?;
    let scalar = |value: &'static str| {
        if dtype.is_utf8() {
            Scalar::utf8(value, Nullability::Nullable)
        } else {
            Scalar::binary(ByteBuffer::copy_from(value), Nullability::Nullable)
        }
    };
    assert_eq!(
        results.get_result(&requests[0]),
        Precision::Exact(scalar("a"))
    );
    assert_eq!(
        results.get_result(&requests[1]),
        Precision::Exact(scalar("z"))
    );
    assert!(chunks[0].statistics().get(Stat::Min).is_exact());
    assert!(chunks[0].statistics().get(Stat::Max).is_exact());
    Ok(())
}

#[rstest]
#[case::utf8(DType::Utf8(Nullability::NonNullable))]
#[case::binary(DType::Binary(Nullability::NonNullable))]
fn final_extrema_bounds_are_inexact(#[case] dtype: DType) -> VortexResult<()> {
    let array = VarBinArray::from_vec(
        vec![b"aaxxxxxxxxx".to_vec(), b"zzxxxxxxxxx".to_vec()],
        dtype.clone(),
    )
    .into_array();
    let requests = [
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
    ];
    let results = accumulate(&dtype, &requests, &[array], 2)?;
    for request in &requests {
        assert!(matches!(results.get_result(request), Precision::Inexact(_)));
    }
    Ok(())
}

#[test]
fn unbounded_binary_max_is_absent() -> VortexResult<()> {
    let dtype = DType::Binary(Nullability::NonNullable);
    let chunks = [
        VarBinArray::from_vec(vec![vec![0xff; 8]], dtype.clone()).into_array(),
        VarBinArray::from_vec(vec![vec![1]], dtype.clone()).into_array(),
    ];
    let request = Max.bind(NumericalAggregateOpts::skip_nans());
    let results = accumulate(&dtype, std::slice::from_ref(&request), &chunks, 2)?;
    assert!(results.get_result(&request).is_absent());
    Ok(())
}

#[rstest]
#[case::constant(IsConstant.bind(EmptyOptions), [1, 1], [2, 2])]
#[case::sorted(IsSorted.bind(IsSortedOptions { strict: false }), [3, 4], [1, 2])]
#[case::strict_sorted(IsSorted.bind(IsSortedOptions { strict: true }), [1, 2], [2, 3])]
fn flags_merge_actual_chunk_boundaries(
    #[case] request: AggregateFnRef,
    #[case] first: [i32; 2],
    #[case] second: [i32; 2],
) -> VortexResult<()> {
    let chunks = [
        PrimitiveArray::from_iter(first).into_array(),
        PrimitiveArray::from_iter(second).into_array(),
    ];
    let results = accumulate(
        chunks[0].dtype(),
        std::slice::from_ref(&request),
        &chunks,
        64,
    )?;
    assert_eq!(
        results.get_result(&request),
        Precision::Exact(Scalar::from(false))
    );
    Ok(())
}

#[test]
fn unsupported_nested_extrema_are_omitted() -> VortexResult<()> {
    let array = ListArray::try_new(
        buffer![1i32].into_array(),
        buffer![0u32, 1].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let requests = [
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
        NullCount.bind(EmptyOptions),
    ];
    let results = accumulate(array.dtype(), &requests, std::slice::from_ref(&array), 64)?;
    assert!(results.get_result(&requests[0]).is_absent());
    assert!(results.get_result(&requests[1]).is_absent());
    assert_eq!(
        results.get_result(&requests[2]),
        Precision::Exact(Scalar::from(0u64))
    );
    Ok(())
}

#[test]
fn empty_chunk_preserves_retained_buffer_size() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let backing = ByteBuffer::copy_from(b"retained backing bytes");
    let expected = backing.len() as u64;
    let array = VarBinViewArray::try_new(
        buffer![],
        Arc::from([backing]),
        DType::Binary(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let request = UncompressedSizeInBytes.bind(EmptyOptions);
    let results = accumulate(
        array.dtype(),
        std::slice::from_ref(&request),
        std::slice::from_ref(&array),
        64,
    )?;
    assert_eq!(
        results.get_result(&request),
        Precision::Exact(Scalar::from(expected))
    );
    Ok(())
}

#[test]
fn all_null_extrema_remain_absent() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([None::<f64>, None]).into_array();
    let requests = [
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
        NullCount.bind(EmptyOptions),
        NanCount.bind(EmptyOptions),
        Sum.bind(NumericalAggregateOpts::skip_nans()),
    ];
    let results = accumulate(array.dtype(), &requests, std::slice::from_ref(&array), 64)?;
    assert!(results.get_result(&requests[0]).is_absent());
    assert!(results.get_result(&requests[1]).is_absent());
    assert_eq!(
        results.get_result(&requests[2]),
        Precision::Exact(Scalar::from(2u64))
    );
    assert_eq!(
        results.get_result(&requests[3]),
        Precision::Exact(Scalar::from(0u64))
    );
    assert_eq!(
        results.get_result(&requests[4]),
        Precision::Exact(Scalar::primitive(0f64, Nullability::Nullable))
    );
    Ok(())
}

#[derive(Debug)]
struct RejectMinMaxKernel;

impl DynAggregateKernel for RejectMinMaxKernel {
    fn aggregate(
        &self,
        _aggregate_fn: &AggregateFnRef,
        _batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        vortex_bail!("Cached extrema must precede the MinMax kernel")
    }
}

#[test]
fn fused_min_max_reuses_exact_chunk_extrema() -> VortexResult<()> {
    static KERNEL: RejectMinMaxKernel = RejectMinMaxKernel;
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let chunks = [buffer![3i32, 7].into_array(), buffer![2i32, 5].into_array()];
    for chunk in &chunks {
        min_max(chunk, &mut ctx, NumericalAggregateOpts::skip_nans())?;
        assert!(chunk.statistics().get(Stat::Min).as_exact().is_some());
        assert!(chunk.statistics().get(Stat::Max).as_exact().is_some());
    }
    session.aggregate_fns().register_aggregate_kernel(
        chunks[0].encoding_id(),
        Some(MinMax.id()),
        &KERNEL,
    );
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let mut accumulator =
        FieldAccumulator::new(chunks[0].dtype(), &[min.clone(), max.clone()], 64)?;
    for chunk in &chunks {
        accumulator.push_chunk(chunk, &mut ctx)?;
    }
    let results = accumulator.results()?;
    assert_eq!(
        results.get_result(&min),
        Precision::Exact(Scalar::primitive(2i32, Nullability::Nullable))
    );
    assert_eq!(
        results.get_result(&max),
        Precision::Exact(Scalar::primitive(7i32, Nullability::Nullable))
    );
    Ok(())
}

#[rstest]
#[case::min(vec![Min.bind(NumericalAggregateOpts::skip_nans())], vec![1, 2], vec![Scalar::primitive(1i32, Nullability::Nullable)])]
#[case::max(vec![Max.bind(NumericalAggregateOpts::skip_nans())], vec![1, 2], vec![Scalar::primitive(2i32, Nullability::Nullable)])]
#[case::fused(vec![Min.bind(NumericalAggregateOpts::skip_nans()), Max.bind(NumericalAggregateOpts::skip_nans())], vec![1, 2], vec![Scalar::primitive(1i32, Nullability::Nullable), Scalar::primitive(2i32, Nullability::Nullable)])]
#[case::constant(vec![IsConstant.bind(EmptyOptions)], vec![1, 1], vec![Scalar::from(true)])]
#[case::sorted(vec![IsSorted.bind(IsSortedOptions { strict: false })], vec![1, 2], vec![Scalar::from(true)])]
#[case::strict_sorted(vec![IsSorted.bind(IsSortedOptions { strict: true })], vec![1, 2], vec![Scalar::from(true)])]
fn empty_constants_do_not_add_values_or_boundaries(
    #[case] requests: Vec<AggregateFnRef>,
    #[case] values: Vec<i32>,
    #[case] expected: Vec<Scalar>,
) -> VortexResult<()> {
    let chunks = [
        ConstantArray::new(Scalar::from(42i32), 0).into_array(),
        PrimitiveArray::from_iter(values).into_array(),
        ConstantArray::new(Scalar::from(-42i32), 0).into_array(),
    ];
    let results = accumulate(chunks[0].dtype(), &requests, &chunks, 64)?;
    for (request, expected) in requests.iter().zip(expected) {
        assert_eq!(results.get_result(request), Precision::Exact(expected));
    }
    Ok(())
}
