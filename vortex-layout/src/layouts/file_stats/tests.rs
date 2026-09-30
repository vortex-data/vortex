// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::slice;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::array_session;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_array::stats::default_file_aggregates;
use vortex_array::validity::Validity;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::FieldAccumulator;

#[test]
fn combines_chunks() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let first = buffer![2i32, 3].into_array();
    let second = buffer![-1i32, 4].into_array();
    let mut acc = FieldAccumulator::new(first.dtype(), &default_file_aggregates(), 64)?;
    acc.push_chunk(&first, &mut ctx)?;
    acc.push_chunk(&second, &mut ctx)?;
    let results = acc.results()?;
    assert_eq!(
        results.get(&Min.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Exact(Scalar::primitive(-1i32, Nullability::Nullable))
    );
    assert_eq!(
        results.get(&Max.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Exact(Scalar::primitive(4i32, Nullability::Nullable))
    );
    assert_eq!(
        results.get(&Sum.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Exact(Scalar::primitive(8i64, Nullability::Nullable))
    );
    assert_eq!(
        results.get(&NullCount.bind(EmptyOptions)),
        Precision::Exact(0u64.into())
    );
    assert_eq!(results.get(&NanCount.bind(EmptyOptions)), Precision::Absent);
    Ok(())
}

#[test]
fn empty_input_has_no_summaries() -> VortexResult<()> {
    let array = PrimitiveArray::empty::<i32>(Nullability::Nullable).into_array();
    let mut acc = FieldAccumulator::new(array.dtype(), &default_file_aggregates(), 64)?;
    acc.push_chunk(&array, &mut array_session().create_execution_ctx())?;
    assert_eq!(acc.results()?.iter().count(), 0);
    Ok(())
}

#[test]
fn all_null_input_has_known_results() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([None::<i32>, None]).into_array();
    let mut acc = FieldAccumulator::new(array.dtype(), &default_file_aggregates(), 64)?;
    acc.push_chunk(&array, &mut array_session().create_execution_ctx())?;
    let results = acc.results()?;
    for aggregate in [
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
    ] {
        assert_eq!(
            results.get(&aggregate),
            Precision::Exact(Scalar::null(array.dtype().clone()))
        );
    }
    assert_eq!(
        results.get(&NullCount.bind(EmptyOptions)),
        Precision::Exact(2u64.into())
    );
    assert_eq!(
        results.get(&Sum.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Exact(Scalar::primitive(0i64, Nullability::Nullable))
    );
    Ok(())
}

#[test]
fn unsupported_extrema_are_omitted() -> VortexResult<()> {
    let array = ListArray::try_new(
        buffer![1i32, 2].into_array(),
        buffer![0u32, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = array_session().create_execution_ctx();
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    for aggregates in [vec![min.clone()], vec![max.clone()], vec![min, max]] {
        let mut acc = FieldAccumulator::new(array.dtype(), &aggregates, 64)?;
        acc.push_chunk(&array, &mut ctx)?;
        assert_eq!(acc.results()?.iter().count(), 0);
    }
    Ok(())
}

#[rstest]
#[case::minimum(vec![Min.bind(NumericalAggregateOpts::skip_nans())])]
#[case::maximum(vec![Max.bind(NumericalAggregateOpts::skip_nans())])]
#[case::both(vec![
    Min.bind(NumericalAggregateOpts::skip_nans()),
    Max.bind(NumericalAggregateOpts::skip_nans()),
])]
fn extension_extrema_keep_the_logical_type(
    #[case] aggregates: Vec<AggregateFnRef>,
) -> VortexResult<()> {
    let array = TemporalArray::new_date(
        PrimitiveArray::from_option_iter([Some(3i32), None, Some(-1)]).into_array(),
        TimeUnit::Days,
    )
    .into_array();
    let mut acc = FieldAccumulator::new(array.dtype(), &aggregates, 64)?;
    acc.push_chunk(&array, &mut array_session().create_execution_ctx())?;
    let results = acc.results()?;
    let ext = Date::new(TimeUnit::Days, Nullability::Nullable).erased();

    for aggregate in aggregates {
        let value = if aggregate.is::<Min>() { -1i32 } else { 3i32 };
        let expected =
            Scalar::extension_ref(ext.clone(), Scalar::primitive(value, Nullability::Nullable));
        assert_eq!(results.get(&aggregate), Precision::Exact(expected));
    }

    Ok(())
}

#[test]
fn nulls_and_nans_do_not_contribute_to_extrema_or_sum() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([None, Some(f64::NAN), Some(3.0)]).into_array();
    let mut acc = FieldAccumulator::new(array.dtype(), &default_file_aggregates(), 64)?;
    acc.push_chunk(&array, &mut array_session().create_execution_ctx())?;
    let results = acc.results()?;
    for aggregate in [
        Min.bind(NumericalAggregateOpts::skip_nans()),
        Max.bind(NumericalAggregateOpts::skip_nans()),
        Sum.bind(NumericalAggregateOpts::skip_nans()),
    ] {
        assert_eq!(
            results.get(&aggregate),
            Precision::Exact(Scalar::primitive(3.0f64, Nullability::Nullable))
        );
    }
    assert_eq!(
        results.get(&NullCount.bind(EmptyOptions)),
        Precision::Exact(1u64.into())
    );
    assert_eq!(
        results.get(&NanCount.bind(EmptyOptions)),
        Precision::Exact(1u64.into())
    );
    Ok(())
}

#[test]
fn overflowing_sum_stays_null_across_chunks() -> VortexResult<()> {
    let first = buffer![i64::MAX, 1].into_array();
    let second = buffer![2i64].into_array();
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let mut acc = FieldAccumulator::new(first.dtype(), slice::from_ref(&sum), 64)?;
    let mut ctx = array_session().create_execution_ctx();
    acc.push_chunk(&first, &mut ctx)?;
    acc.push_chunk(&second, &mut ctx)?;
    assert_eq!(
        acc.results()?.get(&sum),
        Precision::Exact(Scalar::null(DType::Primitive(
            PType::I64,
            Nullability::Nullable
        )))
    );
    Ok(())
}

#[rstest]
#[case::sorted(IsSorted.bind(IsSortedOptions { strict: false }), vec![1i32, 2], vec![0i32, 3])]
#[case::constant(IsConstant.bind(EmptyOptions), vec![1i32, 1], vec![2i32, 2])]
fn flags_include_chunk_boundaries(
    #[case] aggregate: AggregateFnRef,
    #[case] first: Vec<i32>,
    #[case] second: Vec<i32>,
) -> VortexResult<()> {
    let first = PrimitiveArray::from_iter(first).into_array();
    let second = PrimitiveArray::from_iter(second).into_array();
    let mut acc = FieldAccumulator::new(first.dtype(), slice::from_ref(&aggregate), 64)?;
    let mut ctx = array_session().create_execution_ctx();
    acc.push_chunk(&first, &mut ctx)?;
    acc.push_chunk(&second, &mut ctx)?;
    assert_eq!(
        acc.results()?.get(&aggregate),
        Precision::Exact(false.into())
    );
    Ok(())
}

#[rstest]
#[case::utf8(DType::Utf8(Nullability::NonNullable))]
#[case::binary(DType::Binary(Nullability::NonNullable))]
fn truncates_final_bounds(#[case] dtype: DType) -> VortexResult<()> {
    let mut builder = VarBinViewBuilder::with_capacity_in(
        dtype.clone(),
        2,
        BufferAllocatorRef::statically_allocated(),
    );
    builder.append_value("aaa123");
    builder.append_value("zzz999");
    let mut acc = FieldAccumulator::new(&dtype, &default_file_aggregates(), 3)?;
    acc.push_chunk(
        &builder.finish(),
        &mut array_session().create_execution_ctx(),
    )?;
    let results = acc.results()?;
    let expected = |s: &str| {
        if dtype.is_utf8() {
            Scalar::utf8(s.to_owned(), Nullability::Nullable)
        } else {
            Scalar::binary(ByteBuffer::copy_from(s.as_bytes()), Nullability::Nullable)
        }
    };
    assert_eq!(
        results.get(&Min.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Inexact(expected("aaa"))
    );
    assert_eq!(
        results.get(&Max.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Inexact(expected("zz{"))
    );
    Ok(())
}

#[test]
fn omits_maximum_when_truncation_has_no_upper_bound() -> VortexResult<()> {
    let dtype = DType::Binary(Nullability::NonNullable);
    let mut acc = FieldAccumulator::new(&dtype, &default_file_aggregates(), 3)?;
    let mut ctx = array_session().create_execution_ctx();
    for value in [b"aaa".as_slice(), &[0xff; 6]] {
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype.clone(),
            1,
            BufferAllocatorRef::statically_allocated(),
        );
        builder.append_value(value);
        acc.push_chunk(&builder.finish(), &mut ctx)?;
    }
    let results = acc.results()?;
    assert_eq!(
        results.get(&Max.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Absent
    );
    assert!(results.iter().all(|(aggregate, _)| !aggregate.is::<Max>()));
    assert!(
        results
            .get(&Min.bind(NumericalAggregateOpts::skip_nans()))
            .is_exact()
    );
    Ok(())
}
