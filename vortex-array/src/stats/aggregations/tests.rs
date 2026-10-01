// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayInput;
use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
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
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::input::AggregateCacheMode;
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

#[rstest]
#[case(AggregateCacheMode::Array)]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn typed_first_reuses_state_for_finalization(#[case] mode: AggregateCacheMode) -> VortexResult<()> {
    let array = buffer![1i32, 2, 3].into_array();
    let input = ArrayInput::new(array.clone()).with_cache_mode(mode);
    let calls = Arc::new(AtomicUsize::new(0));
    let aggregate = tracked_rows::<0>(&calls, false);
    let key = tracked_rows::<0>(&calls, false).erased();
    let mut ctx = array_session().create_execution_ctx();

    let first = input.compute_partial(&aggregate, &mut ctx)?;
    let second = input.compute_partial(&aggregate, &mut ctx)?;
    assert_eq!(first.rows, 3);
    assert!(input.snapshot_results().iter().next().is_none());
    assert_eq!(input.compute_result(&key, &mut ctx)?, 3u64.into());
    if mode == AggregateCacheMode::Disabled {
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(input.get_result(&key), Precision::Absent);
    } else {
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(input.get_result(&key), Precision::Exact(3u64.into()));
    }

    let retained = Arc::downgrade(&first);
    drop(first);
    drop(second);
    drop(input);
    // The retained state contains the original array. Keeping that array alive must not keep
    // input-owned typed state alive through the array's cache.
    assert!(retained.upgrade().is_none());
    assert_eq!(array.len(), 3);
    Ok(())
}

#[test]
fn scoped_array_mode_finalizes_the_input_partial() -> VortexResult<()> {
    let input = ArrayInput::new(buffer![1i32, 2, 3].into_array())
        .with_cache_mode(AggregateCacheMode::Array);
    let calls = Arc::new(AtomicUsize::new(0));
    let aggregate = tracked_rows::<0>(&calls, false);
    let key = tracked_rows::<0>(&calls, false).erased();
    let mut ctx = array_session().create_execution_ctx();

    input.compute_partial(&aggregate, &mut ctx)?;
    let mut scoped = ctx.with_aggregate_input(&input);
    assert_eq!(
        scoped.compute_aggregate_result(input.array(), &key)?,
        3u64.into()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        input.array().aggregations().get_result(&key),
        Precision::Exact(3u64.into())
    );
    Ok(())
}

#[test]
fn retained_null_false_and_zero_are_known_results() -> VortexResult<()> {
    let input = ArrayInput::new(buffer![u64::MAX, 1].into_array());
    let sum = AggregateFn::new(Sum, NumericalAggregateOpts::skip_nans());
    let constant = AggregateFn::new(IsConstant, EmptyOptions);
    let count = AggregateFn::new(NullCount, EmptyOptions);
    let mut ctx = array_session().create_execution_ctx();

    input.compute_partial(&sum, &mut ctx)?;
    input.compute_partial(&constant, &mut ctx)?;
    input.compute_partial(&count, &mut ctx)?;
    let sum_key = sum.erased();
    let constant_key = constant.erased();
    let count_key = count.erased();
    assert!(input.compute_result(&sum_key, &mut ctx)?.is_null());
    assert_eq!(input.compute_result(&constant_key, &mut ctx)?, false.into());
    assert_eq!(input.compute_result(&count_key, &mut ctx)?, 0u64.into());
    assert!(input.get_result(&sum_key).as_exact().unwrap().is_null());
    assert_eq!(
        input.get_result(&constant_key),
        Precision::Exact(false.into())
    );
    assert_eq!(input.get_result(&count_key), Precision::Exact(0u64.into()));
    assert_eq!(input.snapshot_results().iter().count(), 3);
    Ok(())
}

#[test]
fn scalar_first_does_not_manufacture_typed_state() -> VortexResult<()> {
    let input = ArrayInput::new(buffer![1i32, 2, 3].into_array());
    let calls = Arc::new(AtomicUsize::new(0));
    let aggregate = tracked_rows::<0>(&calls, false);
    let key = tracked_rows::<0>(&calls, false).erased();
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(input.compute_result(&key, &mut ctx)?, 3u64.into());
    let partial = input.compute_partial(&aggregate, &mut ctx)?;
    assert_eq!(partial.rows, 3);
    assert!(partial.batch.is_some());
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    Ok(())
}

#[test]
fn typed_keys_include_options_and_concrete_vtable() -> VortexResult<()> {
    let input = ArrayInput::new(buffer![1i32, 2, 3].into_array());
    let calls = Arc::new(AtomicUsize::new(0));
    let skipped = tracked_rows::<0>(&calls, false);
    let included = AggregateFn::new(
        skipped.vtable().clone(),
        NumericalAggregateOpts::include_nans(),
    );
    let other_vtable = tracked_rows::<1>(&calls, false);
    let mut ctx = array_session().create_execution_ctx();

    let first = input.compute_partial(&skipped, &mut ctx)?;
    let included_partial = input.compute_partial(&included, &mut ctx)?;
    let other_partial = input.compute_partial(&other_vtable, &mut ctx)?;
    assert_eq!(first.rows, 3);
    assert_eq!(included_partial.rows, 103);
    assert_eq!(other_partial.rows, 4);
    assert!(!Arc::ptr_eq(&first, &included_partial));
    assert!(!Arc::ptr_eq(&first, &other_partial));
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert!(Arc::ptr_eq(
        &first,
        &input.compute_partial(&skipped, &mut ctx)?
    ));
    assert!(Arc::ptr_eq(
        &other_partial,
        &input.compute_partial(&other_vtable, &mut ctx)?
    ));
    Ok(())
}

#[test]
fn clear_removes_typed_states_and_snapshots_omit_them() -> VortexResult<()> {
    let array = buffer![1i32, 2, 3].into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let calls = Arc::new(AtomicUsize::new(0));
    let aggregate = tracked_rows::<0>(&calls, false);
    let mut ctx = array_session().create_execution_ctx();

    let partial = cache.compute_partial(&aggregate, &mut ctx)?;
    let retained = Arc::downgrade(&partial);
    drop(partial);
    assert!(retained.upgrade().is_some());
    assert!(cache.snapshot_results().iter().next().is_none());
    cache.clear();
    assert!(retained.upgrade().is_none());
    assert_eq!(cache.compute_partial(&aggregate, &mut ctx)?.rows, 3);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    Ok(())
}

#[test]
fn accumulation_errors_do_not_retain_typed_state() {
    let input = ArrayInput::new(buffer![1i32].into_array());
    let calls = Arc::new(AtomicUsize::new(0));
    let aggregate = tracked_rows::<0>(&calls, true);
    let mut ctx = array_session().create_execution_ctx();

    assert!(input.compute_partial(&aggregate, &mut ctx).is_err());
    assert!(input.compute_partial(&aggregate, &mut ctx).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert!(input.snapshot_results().iter().next().is_none());
}

fn tracked_rows<const KIND: u64>(
    calls: &Arc<AtomicUsize>,
    fail: bool,
) -> AggregateFn<TrackedRows<KIND>> {
    AggregateFn::new(
        TrackedRows {
            calls: Arc::clone(calls),
            fail,
        },
        NumericalAggregateOpts::skip_nans(),
    )
}

#[derive(Clone)]
struct TrackedRows<const KIND: u64> {
    calls: Arc<AtomicUsize>,
    fail: bool,
}

// Intentionally has no Clone implementation, and retains the batch to exercise input ownership.
struct RowsState {
    rows: u64,
    batch: Option<ArrayRef>,
}

impl<const KIND: u64> AggregateFnVTable for TrackedRows<KIND> {
    type Options = NumericalAggregateOpts;
    type Partial = RowsState;

    #[expect(
        clippy::disallowed_methods,
        reason = "test-only id shared by distinct vtable types"
    )]
    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.test.retained_rows")
    }

    fn return_dtype(&self, _options: &Self::Options, _dtype: &DType) -> Option<DType> {
        Some(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, dtype: &DType) -> Option<DType> {
        self.return_dtype(options, dtype)
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(RowsState {
            rows: 0,
            batch: None,
        })
    }

    fn partial_from_result(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _result: Scalar,
    ) -> VortexResult<Option<Self::Partial>> {
        panic!("a finalized row count cannot recover the retained batch")
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        Ok(RowsState {
            rows: u64::try_from(&scalar)?,
            batch: None,
        })
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        Ok(RowsState {
            rows: first.rows + second.rows,
            batch: second.batch.or(first.batch),
        })
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(partial.rows.into())
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn try_accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fail {
            vortex_bail!("intentional accumulation failure");
        }
        partial.rows += batch.len() as u64 + KIND + if args.options.skip_nans { 0 } else { 100 };
        partial.batch = Some(batch.clone());
        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_bail!("raw accumulation handles all batches")
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}
