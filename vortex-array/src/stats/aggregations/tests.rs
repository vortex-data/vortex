// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::stats::Aggregations;

#[test]
fn known_results_remain_distinct_from_missing() -> VortexResult<()> {
    let array = buffer![i64::MAX, 1i64].into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let count = NullCount.bind(EmptyOptions);
    let constant = IsConstant.bind(EmptyOptions);
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());

    assert_eq!(cache.get_result(&count), Precision::Absent);
    cache.insert_result(count.clone(), Precision::Exact(0u64.into()))?;
    cache.insert_result(constant.clone(), Precision::Exact(false.into()))?;
    cache.insert_result(
        sum.clone(),
        Precision::Exact(Scalar::null(sum.return_dtype(array.dtype()).unwrap())),
    )?;

    assert_eq!(cache.get_result(&count), Precision::Exact(0u64.into()));
    assert_eq!(cache.get_result(&constant), Precision::Exact(false.into()));
    assert!(cache.get_result(&sum).as_exact().unwrap().is_null());
    assert_eq!(cache.snapshot_results().iter().count(), 3);
    Ok(())
}

#[rstest]
#[case(false)]
#[case(true)]
fn numerical_options_are_distinct_cache_keys(#[case] include_first: bool) -> VortexResult<()> {
    let array = buffer![1.0f64, f64::NAN].into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let skipped = Sum.bind(NumericalAggregateOpts::skip_nans());
    let included = Sum.bind(NumericalAggregateOpts::include_nans());
    let mut ctx = array_session().create_execution_ctx();

    if include_first {
        assert!(cache.compute_as::<f64>(&included, &mut ctx)?.is_nan());
        assert_eq!(cache.get_result(&skipped), Precision::Absent);
    } else {
        assert_eq!(cache.compute_as::<f64>(&skipped, &mut ctx)?, 1.0);
        assert_eq!(cache.get_result(&included), Precision::Absent);
    }
    assert!(cache.compute_as::<f64>(&included, &mut ctx)?.is_nan());
    assert_eq!(cache.compute_as::<f64>(&skipped, &mut ctx)?, 1.0);
    Ok(())
}

#[test]
fn errors_do_not_populate_the_cache() {
    let array = ConstantArray::new("text", 3).into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let mut ctx = array_session().create_execution_ctx();

    assert!(cache.compute_result(&sum, &mut ctx).is_err());
    assert_eq!(cache.get_result(&sum), Precision::Absent);
    assert!(cache.snapshot_results().iter().next().is_none());
}

#[derive(Debug)]
struct FailingAfterDependency;

impl DynAggregateKernel for FailingAfterDependency {
    fn aggregate(
        &self,
        _aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        batch
            .aggregations()
            .compute_result(&NullCount.bind(EmptyOptions), ctx)?;
        vortex_bail!("Target failed after computing its dependency")
    }
}

#[test]
fn failed_target_preserves_successful_dependency() -> VortexResult<()> {
    let array = buffer![1i32, 2, 3].into_array();
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let count = NullCount.bind(EmptyOptions);
    assert_eq!(array.aggregations().get_result(&count), Precision::Absent);
    let session = array_session();
    session.aggregate_fns().register_aggregate_kernel(
        array.encoding_id(),
        Some(sum.id()),
        &FailingAfterDependency,
    );
    let mut ctx = session.create_execution_ctx();

    assert!(array.aggregations().compute_result(&sum, &mut ctx).is_err());
    assert_eq!(array.aggregations().get_result(&sum), Precision::Absent);
    assert_eq!(
        array.aggregations().get_result(&count),
        Precision::Exact(0u64.into())
    );
    assert_eq!(array.aggregations().snapshot_results().iter().count(), 1);
    Ok(())
}

#[test]
fn insertion_checks_result_dtype() {
    let array = buffer![1i32].into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let count = NullCount.bind(EmptyOptions);

    assert!(
        cache
            .insert_result(
                count.clone(),
                Precision::Exact(Scalar::primitive(1u64, Nullability::Nullable))
            )
            .is_err()
    );
    assert_eq!(cache.get_result(&count), Precision::Absent);
}

#[test]
fn concurrent_misses_and_errors_leave_known_results() -> VortexResult<()> {
    let array = buffer![1i32, 2, 3].into_array();
    let unsupported = ConstantArray::new("text", 3).into_array();
    let count = NullCount.bind(EmptyOptions);
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    std::thread::scope(|scope| {
        let workers = (0..8)
            .map(|_| {
                scope.spawn(|| -> VortexResult<()> {
                    let mut ctx = array_session().create_execution_ctx();
                    assert_eq!(array.aggregations().compute_as::<u64>(&count, &mut ctx)?, 0);
                    assert!(
                        unsupported
                            .aggregations()
                            .compute_result(&sum, &mut ctx)
                            .is_err()
                    );
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().expect("worker panicked")?;
        }
        Ok::<_, vortex_error::VortexError>(())
    })?;
    assert_eq!(
        array.aggregations().get_result(&count),
        Precision::Exact(0u64.into())
    );
    assert_eq!(
        unsupported.aggregations().get_result(&sum),
        Precision::Absent
    );
    Ok(())
}

#[test]
fn physical_size_is_not_inherited_across_representations() -> VortexResult<()> {
    let source = buffer![1i32, 2].into_array();
    let rewritten = buffer![1i32, 2].into_array();
    let count = NullCount.bind(EmptyOptions);
    let size = UncompressedSizeInBytes.bind(EmptyOptions);
    let mut ctx = array_session().create_execution_ctx();
    source.aggregations().compute_result(&count, &mut ctx)?;
    source.aggregations().compute_result(&size, &mut ctx)?;
    rewritten
        .aggregations()
        .inherit_from(source.aggregations())?;

    assert_eq!(
        rewritten.aggregations().get_result(&count),
        Precision::Exact(0u64.into())
    );
    assert_eq!(
        rewritten.aggregations().get_result(&size),
        Precision::Absent
    );
    assert!(source.aggregations().get_result(&size).is_exact());
    Ok(())
}

#[test]
fn dtype_changing_reduction_drops_cached_results() -> VortexResult<()> {
    let source = PrimitiveArray::from_option_iter([Some(7i32), Some(7)]).into_array();
    let reduced = ConstantArray::new(7i32, 2).into_array();
    let count = NullCount.bind(EmptyOptions);
    let mut ctx = array_session().create_execution_ctx();
    source.aggregations().compute_result(&count, &mut ctx)?;
    reduced.aggregations().inherit_from(source.aggregations())?;

    assert_eq!(reduced.aggregations().get_result(&count), Precision::Absent);
    assert_eq!(
        source.aggregations().get_result(&count),
        Precision::Exact(0u64.into())
    );
    Ok(())
}
