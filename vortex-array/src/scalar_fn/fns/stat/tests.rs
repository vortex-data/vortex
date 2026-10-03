// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::stat_array;
use super::stat_dtype;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::all_nan::AllNan;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::nan_count::NAN_COUNT;
use crate::aggregate_fn::fns::null_count::NULL_COUNT;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::array_session;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;

fn stat_scalar(array: &ArrayRef, aggregate: &AggregateFnRef, len: usize) -> VortexResult<Scalar> {
    let dtype = stat_dtype(aggregate, array.dtype())?;
    let output = stat_array(array, aggregate, dtype, len)?;
    assert_eq!(output.len(), len);
    Ok(output.as_::<Constant>().scalar().clone())
}

#[test]
fn reads_generic_count_without_a_legacy_slot() -> VortexResult<()> {
    let array = buffer![1u32, 2, 3].into_array();
    let aggregate = Count.bind(NumericalAggregateOpts::skip_nans());
    let mut ctx = array_session().create_execution_ctx();
    array.aggregations().compute_result(&aggregate, &mut ctx)?;
    assert_eq!(
        stat_scalar(&array, &aggregate, array.len())?,
        Scalar::primitive(3u64, Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn metadata_lookup_preserves_numeric_options() -> VortexResult<()> {
    let array = buffer![1.0f64, f64::NAN, 2.0].into_array();
    let skip = Sum.bind(NumericalAggregateOpts::skip_nans());
    let include = Sum.bind(NumericalAggregateOpts::include_nans());
    let mut ctx = array_session().create_execution_ctx();
    array.aggregations().compute_result(&skip, &mut ctx)?;
    assert_eq!(
        f64::try_from(&stat_scalar(&array, &skip, array.len())?)?,
        3.0
    );
    assert!(stat_scalar(&array, &include, array.len())?.is_null());
    assert!(array.aggregations().get_result(&include).is_absent());

    array.aggregations().compute_result(&include, &mut ctx)?;
    assert!(f64::try_from(&stat_scalar(&array, &include, array.len())?)?.is_nan());
    Ok(())
}

#[test]
fn inexact_extrema_remain_available_as_bounds() -> VortexResult<()> {
    let array = buffer![3i32, 1, 2].into_array();
    let aggregate = Min.bind(NumericalAggregateOpts::skip_nans());
    let bound = Scalar::primitive(0i32, Nullability::Nullable);
    array
        .aggregations()
        .insert_result(aggregate.clone(), Precision::Inexact(bound.clone()));
    assert_eq!(stat_scalar(&array, &aggregate, array.len())?, bound);
    Ok(())
}

#[rstest]
#[case::missing_min(Min.bind(NumericalAggregateOpts::skip_nans()), None)]
#[case::sorted_final(IsSorted.bind(IsSortedOptions { strict: false }), Some(true.into()))]
#[case::sum_v2_final(SumV2.bind(NumericalAggregateOpts::skip_nans()), Some(Scalar::primitive(3u64, Nullability::Nullable)))]
fn missing_or_unrecoverable_results_return_typed_nulls(
    #[case] aggregate: AggregateFnRef,
    #[case] result: Option<Scalar>,
) -> VortexResult<()> {
    let array = buffer![1u32, 2].into_array();
    if let Some(result) = result {
        array
            .aggregations()
            .insert_result(aggregate.clone(), Precision::Exact(result));
    }
    let before = array.aggregations().snapshot_results();
    let scalar = stat_scalar(&array, &aggregate, array.len())?;
    assert!(scalar.is_null());
    assert_eq!(scalar.dtype(), &stat_dtype(&aggregate, array.dtype())?);
    assert_eq!(
        array.aggregations().snapshot_results().iter().count(),
        before.iter().count()
    );
    Ok(())
}

#[rstest]
#[case::all_null(false)]
#[case::all_nan(true)]
fn derived_all_properties_use_the_input_length(#[case] nan: bool) -> VortexResult<()> {
    let (array, aggregate, count) = if nan {
        (
            ConstantArray::new(f64::NAN, 3).into_array(),
            AllNan.bind(EmptyOptions),
            &*NAN_COUNT,
        )
    } else {
        (
            ConstantArray::new(
                Scalar::null(DType::Primitive(PType::F64, Nullability::Nullable)),
                3,
            )
            .into_array(),
            AllNull.bind(EmptyOptions),
            &*NULL_COUNT,
        )
    };
    array
        .aggregations()
        .insert_result(count.clone(), Precision::Exact(3u64.into()));
    assert_eq!(
        stat_scalar(&array, &aggregate, 7)?,
        Scalar::bool(true, Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn exact_null_sum_stays_cached_during_metadata_lookup() -> VortexResult<()> {
    let array = buffer![u64::MAX, 1].into_array();
    let aggregate = Sum.bind(NumericalAggregateOpts::skip_nans());
    let mut ctx = array_session().create_execution_ctx();
    assert!(
        array
            .aggregations()
            .compute_result(&aggregate, &mut ctx)?
            .is_null()
    );
    assert!(stat_scalar(&array, &aggregate, array.len())?.is_null());
    assert!(
        matches!(array.aggregations().get_result(&aggregate), Precision::Exact(result) if result.is_null())
    );
    Ok(())
}

#[test]
fn fused_extrema_use_the_explicit_partial_contract() -> VortexResult<()> {
    let array = buffer![3i32, 1, 2].into_array();
    let aggregate = MinMax.bind(NumericalAggregateOpts::skip_nans());
    let mut ctx = array_session().create_execution_ctx();
    let result = array.aggregations().compute_result(&aggregate, &mut ctx)?;
    assert_eq!(stat_scalar(&array, &aggregate, array.len())?, result);
    Ok(())
}
