// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::ArrayView;
use crate::Columnar;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::is_constant::IS_CONSTANT;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::fns::is_sorted::IS_SORTED;
use crate::aggregate_fn::fns::is_sorted::IS_STRICT_SORTED;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::max::MAX_SKIP_NANS;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::MIN_SKIP_NANS;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum::sum;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::session::AggregateFnSession;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::VTable;
use crate::array::vtable::NotSupported;
use crate::array::vtable::ValidityVTable;
use crate::array_session;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::propagate_take_results;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::stats::StatsSet;
use crate::validity::Validity;

#[test]
fn helpers_and_fixed_facade_share_the_store() -> VortexResult<()> {
    let array = buffer![1u32, 2, 3].into_array();
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let result = sum(&array, &mut ctx)?;
    let key = Stat::Sum.finalized_aggregate_fn();
    assert_eq!(
        array.aggregations().get_result(key),
        Precision::Exact(result.clone())
    );
    assert_eq!(
        array.statistics().get(Stat::Sum),
        Precision::Exact(result.clone())
    );
    assert_eq!(array.aggregations().snapshot_results().iter().count(), 1);

    array.statistics().clear(Stat::Sum);
    assert_eq!(array.aggregations().get_result(key), Precision::Absent);
    array
        .statistics()
        .set(Stat::Sum, Precision::Exact(result.into_value().unwrap()));
    assert_eq!(array.aggregations().compute_as::<u64>(key, &mut ctx)?, 6);
    Ok(())
}

#[test]
fn exact_null_survives_legacy_projection_and_overflow_reconstruction() -> VortexResult<()> {
    let array = buffer![u64::MAX, 1].into_array();
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let result = sum(&array, &mut ctx)?;
    assert!(result.is_null());
    assert_eq!(
        array
            .aggregations()
            .get_result(Stat::Sum.finalized_aggregate_fn()),
        Precision::Exact(result)
    );
    assert_eq!(array.statistics().get(Stat::Sum), Precision::Absent);
    let projected = array.statistics().to_owned();
    array
        .clone()
        .downcast::<Primitive>()
        .with_stats_set(projected);
    assert!(
        array
            .aggregations()
            .get_result(Stat::Sum.finalized_aggregate_fn())
            .is_exact()
    );

    let mut accumulator = Accumulator::try_new(
        Sum,
        NumericalAggregateOpts::skip_nans(),
        array.dtype().clone(),
    )?;
    accumulator.accumulate(&array, &mut ctx)?;
    accumulator.accumulate(&buffer![1u64].into_array(), &mut ctx)?;
    assert!(accumulator.finish()?.is_null());
    // Detached projections do not hold a cache lock through the callback.
    array.statistics().with_typed_stats_set(|stats| {
        assert_eq!(stats.get(Stat::Sum), Precision::Absent);
        array
            .statistics()
            .set(Stat::IsSorted, Precision::exact(false));
    });
    Ok(())
}

/// Returns twice the row count, so its final scalar is not its mergeable partial even though the
/// dtypes are equal. The default result-to-partial hook must decline it.
#[derive(Clone)]
struct TwiceRows {
    calls: Arc<AtomicUsize>,
    _drop_probe: Option<Arc<CacheDropProbe>>,
}

struct CacheDropProbe {
    owner: Weak<ArrayRef>,
    unlocked: Arc<AtomicBool>,
}

impl Drop for CacheDropProbe {
    fn drop(&mut self) {
        if let Some(array) = self.owner.upgrade() {
            self.unlocked.store(
                array
                    .aggregations()
                    .aggregations
                    .entries
                    .try_write()
                    .is_some(),
                Ordering::Relaxed,
            );
        }
    }
}

impl AggregateFnVTable for TwiceRows {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("test.twice_rows");
        *ID
    }

    fn return_dtype(&self, _options: &EmptyOptions, _input: &DType) -> Option<DType> {
        Some(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &EmptyOptions, input: &DType) -> Option<DType> {
        self.return_dtype(options, input)
    }

    fn empty_partial(&self, _args: AggregateArgs<'_, EmptyOptions>) -> VortexResult<u64> {
        Ok(0)
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, EmptyOptions>,
        scalar: Scalar,
    ) -> VortexResult<u64> {
        u64::try_from(&scalar)
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, EmptyOptions>,
        first: u64,
        second: u64,
    ) -> VortexResult<u64> {
        Ok(first + second)
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, EmptyOptions>,
        partial: &u64,
    ) -> VortexResult<Scalar> {
        Ok((*partial).into())
    }

    fn is_saturated(&self, _args: AggregateArgs<'_, EmptyOptions>, _partial: &u64) -> bool {
        false
    }

    fn try_accumulate(
        &self,
        _args: AggregateArgs<'_, EmptyOptions>,
        partial: &mut u64,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        *partial += batch.len() as u64;
        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, EmptyOptions>,
        _partial: &mut u64,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_bail!("row-count metadata handles every input")
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, EmptyOptions>,
        _states: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("grouped finalization is not used by this test aggregate")
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, EmptyOptions>,
        partial: &u64,
    ) -> VortexResult<Scalar> {
        Ok((2 * partial).into())
    }
}

#[test]
fn custom_results_share_clones_and_decline_partial_recovery() -> VortexResult<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let vtable = TwiceRows {
        calls: Arc::clone(&calls),
        _drop_probe: None,
    };
    let aggregate = vtable.bind(EmptyOptions);
    let array = buffer![1u32, 2].into_array();
    let clone = array.clone();
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    assert_eq!(
        array
            .aggregations()
            .compute_as::<u64>(&aggregate, &mut ctx)?,
        4
    );
    assert_eq!(
        clone
            .aggregations()
            .compute_as::<u64>(&aggregate, &mut ctx)?,
        4
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    clone.aggregations().clear();
    assert_eq!(
        array.aggregations().get_result(&aggregate),
        Precision::Absent
    );
    assert_eq!(
        array
            .aggregations()
            .compute_as::<u64>(&aggregate, &mut ctx)?,
        4
    );
    array
        .clone()
        .downcast::<Primitive>()
        .with_stats_set(StatsSet::default());
    assert!(clone.aggregations().get_result(&aggregate).is_exact());

    let mut accumulator = Accumulator::try_new(vtable, EmptyOptions, array.dtype().clone())?;
    accumulator.accumulate(&array, &mut ctx)?;
    accumulator.accumulate(&buffer![3u32, 4, 5].into_array(), &mut ctx)?;
    assert_eq!(u64::try_from(&accumulator.finish()?)?, 10);
    assert_eq!(calls.load(Ordering::Relaxed), 4);
    Ok(())
}

#[test]
fn new_representations_preserve_only_portable_results() -> VortexResult<()> {
    let array = buffer![1u32, 2].into_array();
    let aggregate = TwiceRows {
        calls: Arc::new(AtomicUsize::new(0)),
        _drop_probe: None,
    }
    .bind(EmptyOptions);
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    sum(&array, &mut ctx)?;
    let count = Count.bind(NumericalAggregateOpts::skip_nans());
    array.aggregations().compute_result(&count, &mut ctx)?;
    array
        .statistics()
        .compute_uncompressed_size_in_bytes(&mut ctx);
    array.aggregations().compute_result(&aggregate, &mut ctx)?;

    // SAFETY: the original slots are unchanged, so validation and logical values are preserved.
    let rewritten = unsafe {
        array
            .clone()
            .with_slots(array.slots().iter().cloned().collect())
    }?;
    assert_ne!(array.addr(), rewritten.addr());
    assert_eq!(
        rewritten.statistics().get(Stat::Sum),
        array.statistics().get(Stat::Sum)
    );
    assert_eq!(
        rewritten.aggregations().get_result(&count),
        array.aggregations().get_result(&count)
    );
    assert_eq!(
        rewritten.aggregations().get_result(&aggregate),
        Precision::Absent
    );
    assert_eq!(
        rewritten.statistics().get(Stat::UncompressedSizeInBytes),
        Precision::Absent
    );
    rewritten.aggregations().clear();
    assert!(array.aggregations().get_result(&aggregate).is_exact());

    let nullable = PrimitiveArray::from_option_iter([Some(1u32), Some(2)]).into_array();
    nullable.statistics().inherit_from(array.statistics());
    assert_eq!(nullable.statistics().get(Stat::Sum), Precision::Absent);
    let shorter = buffer![1u32].into_array();
    shorter.statistics().inherit_from(array.statistics());
    assert_eq!(shorter.statistics().get(Stat::Sum), Precision::Absent);
    Ok(())
}

#[rstest]
#[case(false)]
#[case(true)]
fn nan_options_keep_distinct_results(#[case] include_first: bool) -> VortexResult<()> {
    let array = buffer![1.0f64, f64::NAN, 3.0].into_array();
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let skip = NumericalAggregateOpts::skip_nans();
    let include = NumericalAggregateOpts::include_nans();
    let options = if include_first {
        [include, skip]
    } else {
        [skip, include]
    };
    for options in options {
        array
            .aggregations()
            .compute_result(&Sum.bind(options), &mut ctx)?;
        min_max(&array, &mut ctx, options)?;
    }
    assert_eq!(f64::try_from(&sum(&array, &mut ctx)?)?, 4.0);
    assert!(
        array
            .aggregations()
            .compute_as::<f64>(&Sum.bind(include), &mut ctx)?
            .is_nan()
    );
    assert_eq!(
        array
            .aggregations()
            .compute_as::<f64>(&Min.bind(skip), &mut ctx)?,
        1.0
    );
    assert_eq!(
        array
            .aggregations()
            .compute_as::<f64>(&Max.bind(skip), &mut ctx)?,
        3.0
    );
    assert!(
        array
            .aggregations()
            .compute_as::<f64>(&Min.bind(include), &mut ctx)?
            .is_nan()
    );
    assert!(
        array
            .aggregations()
            .compute_as::<f64>(&Max.bind(include), &mut ctx)?
            .is_nan()
    );
    // SAFETY: unchanged slots preserve the logical input and its validity and order.
    let rewritten = unsafe {
        array
            .clone()
            .with_slots(array.slots().iter().cloned().collect())
    }?;
    for options in [skip, include] {
        for aggregate in [Sum.bind(options), Min.bind(options), Max.bind(options)] {
            assert!(rewritten.aggregations().get_result(&aggregate).is_exact());
        }
    }
    Ok(())
}

#[test]
fn sorted_final_flags_do_not_hide_stream_boundaries() -> VortexResult<()> {
    let first = buffer![1i32, 2].into_array();
    let second = buffer![0i32, 3].into_array();
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    assert!(is_sorted(&first, &mut ctx)?);
    assert!(is_sorted(&second, &mut ctx)?);
    let mut accumulator = Accumulator::try_new(
        IsSorted,
        IsSortedOptions { strict: false },
        first.dtype().clone(),
    )?;
    accumulator.accumulate(&first, &mut ctx)?;
    accumulator.accumulate(&second, &mut ctx)?;
    assert!(!bool::try_from(&accumulator.finish()?)?);
    Ok(())
}

#[test]
fn fixed_setters_replace_and_clear_while_publication_preserves_exact() {
    let array = buffer![1i32, 2].into_array();
    let key = Stat::Min.finalized_aggregate_fn();
    let lower = Scalar::primitive(0i32, Nullability::Nullable);
    let tighter = Scalar::primitive(1i32, Nullability::Nullable);
    array
        .aggregations()
        .insert_result(key.clone(), Precision::Inexact(lower.clone()));
    array
        .aggregations()
        .insert_result(key.clone(), Precision::Inexact(tighter.clone()));
    assert_eq!(
        array.aggregations().get_result(key),
        Precision::Inexact(lower)
    );
    array
        .aggregations()
        .insert_result(key.clone(), Precision::Exact(tighter.clone()));
    array
        .aggregations()
        .insert_result(key.clone(), Precision::Inexact(tighter));
    assert!(array.aggregations().get_result(key).is_exact());

    array.statistics().set(Stat::Min, Precision::inexact(0i32));
    assert_eq!(
        array.statistics().get(Stat::Min),
        Precision::Inexact(Scalar::from(0i32))
    );
    array.statistics().set(Stat::Min, Precision::Absent);
    assert_eq!(array.aggregations().get_result(key), Precision::Absent);
}

#[derive(Debug)]
struct RejectSumKernel;

impl DynAggregateKernel for RejectSumKernel {
    fn aggregate(
        &self,
        _aggregate: &AggregateFnRef,
        _batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        vortex_bail!("the cached result should precede this kernel")
    }
}

#[test]
fn helper_result_reconstructs_typed_state_before_kernel_dispatch() -> VortexResult<()> {
    let array = buffer![1u32, 2, 3].into_array();
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    assert_eq!(u64::try_from(&sum(&array, &mut ctx)?)?, 6);
    static KERNEL: RejectSumKernel = RejectSumKernel;
    session
        .get::<AggregateFnSession>()
        .register_aggregate_kernel(Primitive.id(), Some(Sum.id()), &KERNEL);

    let mut accumulator = Accumulator::try_new(
        Sum,
        NumericalAggregateOpts::skip_nans(),
        array.dtype().clone(),
    )?;
    accumulator.accumulate(&array, &mut ctx)?;
    assert_eq!(u64::try_from(&accumulator.finish()?)?, 6);
    array.aggregations().clear();
    assert!(accumulator.accumulate(&array, &mut ctx).is_err());
    Ok(())
}

#[rstest]
#[case::clear_all(false)]
#[case::clear_one(true)]
fn cache_removal_drops_custom_functions_after_unlock(#[case] clear_one: bool) {
    let owner = Arc::new(buffer![1u32, 2].into_array());
    let unlocked = Arc::new(AtomicBool::new(false));
    let aggregate = TwiceRows {
        calls: Arc::new(AtomicUsize::new(0)),
        _drop_probe: Some(Arc::new(CacheDropProbe {
            owner: Arc::downgrade(&owner),
            unlocked: Arc::clone(&unlocked),
        })),
    }
    .bind(EmptyOptions);
    owner
        .aggregations()
        .insert_result(aggregate, Precision::Exact(4u64.into()));

    if clear_one {
        // An equal request can have a separate vtable allocation from the retained cache key.
        let request = TwiceRows {
            calls: Arc::new(AtomicUsize::new(0)),
            _drop_probe: None,
        }
        .bind(EmptyOptions);
        owner.aggregations().clear_result(&request);
    } else {
        owner.aggregations().clear();
    }
    assert!(unlocked.load(Ordering::Relaxed));
    assert!(
        owner
            .aggregations()
            .snapshot_results()
            .iter()
            .next()
            .is_none()
    );
}

#[rstest]
#[case::single_unique(false, false)]
#[case::single_held(false, true)]
#[case::iterative_unique(true, false)]
#[case::iterative_held(true, true)]
fn execution_snapshots_preserve_unique_input_ownership(
    #[case] iterative: bool,
    #[case] hold_input: bool,
) -> VortexResult<()> {
    let values = Buffer::copy_from([1i32, 2, 3]);
    let original_ptr = values.as_ptr();
    let clones = Arc::new(AtomicUsize::new(0));
    let array = Array::try_from_parts(ArrayParts::new(
        OwnershipProbe,
        DType::Primitive(PType::I32, Nullability::NonNullable),
        values.len(),
        OwnershipProbeData {
            values,
            clones: Arc::clone(&clones),
        },
    ))?
    .into_array();
    let sum = Stat::Sum.finalized_aggregate_fn();
    let result = Precision::Exact(Scalar::primitive(6i64, Nullability::Nullable));
    array
        .aggregations()
        .insert_result(sum.clone(), result.clone());
    array.statistics().set(
        Stat::UncompressedSizeInBytes,
        Precision::Exact(12u64.into()),
    );
    let held = hold_input.then(|| array.clone());
    let mut ctx = array_session().create_execution_ctx();

    let output = if iterative {
        array.execute_until::<Primitive>(&mut ctx)?
    } else {
        array.execute::<ArrayRef>(&mut ctx)?
    };

    assert_eq!(clones.load(Ordering::Relaxed), usize::from(hold_input));
    let output_ptr = output.as_::<Primitive>().as_slice::<i32>().as_ptr();
    assert_eq!(output_ptr == original_ptr, !hold_input);
    assert_eq!(output.aggregations().get_result(sum), result);
    assert_eq!(
        output.statistics().get(Stat::UncompressedSizeInBytes),
        Precision::Absent
    );
    assert_arrays_eq!(output, buffer![1i32, 2, 3].into_array(), &mut ctx);
    drop(held);
    Ok(())
}

#[derive(Clone, Debug)]
struct OwnershipProbe;

#[derive(Debug)]
struct OwnershipProbeData {
    values: Buffer<i32>,
    clones: Arc<AtomicUsize>,
}

impl Clone for OwnershipProbeData {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::Relaxed);
        Self {
            values: self.values.clone(),
            clones: Arc::clone(&self.clones),
        }
    }
}

impl Display for OwnershipProbeData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("ownership-probe")
    }
}

impl ArrayHash for OwnershipProbeData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _eq_mode: EqMode) {
        self.values.hash(state);
    }
}

impl ArrayEq for OwnershipProbeData {
    fn array_eq(&self, other: &Self, _eq_mode: EqMode) -> bool {
        self.values == other.values
    }
}

impl ValidityVTable<OwnershipProbe> for OwnershipProbe {
    fn validity(_array: ArrayView<'_, OwnershipProbe>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl VTable for OwnershipProbe {
    type TypedArrayData = OwnershipProbeData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.ownership-probe");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype == &DType::Primitive(PType::I32, Nullability::NonNullable),
            "OwnershipProbe requires non-nullable I32, got {dtype}"
        );
        vortex_ensure!(
            len == data.values.len(),
            "OwnershipProbe requires length {}, got {len}",
            data.values.len()
        );
        vortex_ensure!(
            slots.is_empty(),
            "OwnershipProbe requires no slots, got {}",
            slots.len()
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        assert_eq!(idx, 0);
        BufferHandle::new_host(array.data().values.clone().into_byte_buffer())
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        (idx == 0).then(|| "values".to_string())
    }

    fn with_buffers(
        &self,
        _array: ArrayView<'_, Self>,
        _buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("OwnershipProbe cannot replace buffers")
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("OwnershipProbe cannot be deserialized")
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("OwnershipProbe slot index {idx} out of bounds")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let data = match array.try_into_parts() {
            Ok(parts) => parts.data,
            Err(array) => array.data().clone(),
        };
        // A retained parent forces both the data clone and a copy when converting to a mutable buffer.
        let values = data.values.into_mut();
        Ok(ExecutionResult::done(PrimitiveArray::new(
            values.freeze(),
            Validity::NonNullable,
        )))
    }
}

#[test]
fn detached_transfer_preserves_keys_bounds_and_portability() -> VortexResult<()> {
    let array = buffer![1.0f64, f64::NAN, 3.0].into_array();
    let mut ctx = array_session().create_execution_ctx();
    let skip_sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let include_sum = Sum.bind(NumericalAggregateOpts::include_nans());
    array.aggregations().compute_result(&skip_sum, &mut ctx)?;
    array
        .aggregations()
        .compute_result(&include_sum, &mut ctx)?;
    let bound = Max.bind(NumericalAggregateOpts::skip_nans());
    array.aggregations().insert_result(
        bound.clone(),
        Precision::Inexact(Scalar::primitive(5.0f64, Nullability::Nullable)),
    );
    let custom = TwiceRows {
        calls: Arc::new(AtomicUsize::new(0)),
        _drop_probe: None,
    }
    .bind(EmptyOptions);
    array.aggregations().compute_result(&custom, &mut ctx)?;
    array
        .statistics()
        .compute_uncompressed_size_in_bytes(&mut ctx);
    let results = array.aggregations().snapshot_results();
    let dtype = array.dtype().clone();
    let len = array.len();
    drop(array);

    let output = buffer![1.0f64, f64::NAN, 3.0].into_array();
    output
        .aggregations()
        .inherit_from_snapshot(&results, &dtype, len);

    for key in [&skip_sum, &include_sum, &bound] {
        assert_eq!(
            output.aggregations().get_result(key),
            results.get_result(key)
        );
    }
    assert_eq!(output.aggregations().get_result(&custom), Precision::Absent);
    assert_eq!(
        output.statistics().get(Stat::UncompressedSizeInBytes),
        Precision::Absent
    );
    Ok(())
}

#[test]
fn detached_transfer_preserves_exact_null() -> VortexResult<()> {
    let array = buffer![u64::MAX, 1].into_array();
    let mut ctx = array_session().create_execution_ctx();
    let result = sum(&array, &mut ctx)?;
    assert!(result.is_null());
    let results = array.aggregations().snapshot_results();
    let dtype = array.dtype().clone();
    let len = array.len();
    drop(array);

    let output = buffer![u64::MAX, 1].into_array();
    output
        .aggregations()
        .inherit_from_snapshot(&results, &dtype, len);

    assert_eq!(
        output
            .aggregations()
            .get_result(Stat::Sum.finalized_aggregate_fn()),
        Precision::Exact(result)
    );
    Ok(())
}

#[rstest]
#[case::matching(DType::Primitive(PType::U32, Nullability::NonNullable), 2, true)]
#[case::different_dtype(DType::Primitive(PType::U32, Nullability::Nullable), 2, false)]
#[case::different_length(DType::Primitive(PType::U32, Nullability::NonNullable), 1, false)]
fn detached_transfer_checks_source_dtype_and_length(
    #[case] source_dtype: DType,
    #[case] source_len: usize,
    #[case] transfer: bool,
) -> VortexResult<()> {
    let array = buffer![1u32, 2].into_array();
    let mut ctx = array_session().create_execution_ctx();
    sum(&array, &mut ctx)?;
    let results = array.aggregations().snapshot_results();
    let output = buffer![1u32, 2].into_array();

    output
        .aggregations()
        .inherit_from_snapshot(&results, &source_dtype, source_len);

    assert_eq!(
        output
            .aggregations()
            .get_result(Stat::Sum.finalized_aggregate_fn())
            .is_exact(),
        transfer
    );
    Ok(())
}

#[rstest]
#[case::increasing(vec![1i32, 2, 3], false, true)]
#[case::constant(vec![1i32, 1, 1], true, false)]
fn slices_keep_only_true_flags(
    #[case] values: Vec<i32>,
    #[case] constant: bool,
    #[case] strict: bool,
) -> VortexResult<()> {
    let array = PrimitiveArray::from_iter(values).into_array();
    let mut ctx = array_session().create_execution_ctx();
    for aggregate in [&*IS_CONSTANT, &*IS_SORTED, &*IS_STRICT_SORTED] {
        array.aggregations().compute_result(aggregate, &mut ctx)?;
    }
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    array.aggregations().compute_result(&sum, &mut ctx)?;

    let sliced = array.slice(1..3)?;
    for (aggregate, expected) in [
        (&*IS_CONSTANT, constant),
        (&*IS_SORTED, true),
        (&*IS_STRICT_SORTED, strict),
    ] {
        assert_eq!(
            sliced.aggregations().get_result_as::<bool>(aggregate)?,
            if expected {
                Precision::Exact(true)
            } else {
                Precision::Absent
            },
        );
    }
    assert_eq!(sliced.aggregations().get_result(&sum), Precision::Absent);

    let empty = array.slice(0..0)?;
    assert_eq!(
        empty.aggregations().get_result(&IS_CONSTANT),
        Precision::Absent
    );
    assert!(!is_constant(&empty, &mut ctx)?);
    Ok(())
}

#[test]
fn gathered_bounds_yield_to_exact_results() -> VortexResult<()> {
    let source = buffer![10i32, 20, 30].into_array();
    let indices = buffer![1u32].into_array();
    let target = buffer![20i32].into_array();
    let mut ctx = array_session().create_execution_ctx();
    min_max(&source, &mut ctx, NumericalAggregateOpts::skip_nans())?;

    propagate_take_results(&source, &target, &indices)?;
    for (aggregate, bound) in [(&*MIN_SKIP_NANS, 10), (&*MAX_SKIP_NANS, 30)] {
        assert_eq!(
            target.aggregations().get_result_as::<i32>(aggregate)?,
            Precision::Inexact(bound)
        );
    }

    min_max(&target, &mut ctx, NumericalAggregateOpts::skip_nans())?;
    propagate_take_results(&source, &target, &indices)?;
    for aggregate in [&*MIN_SKIP_NANS, &*MAX_SKIP_NANS] {
        assert_eq!(
            target.aggregations().get_result_as::<i32>(aggregate)?,
            Precision::Exact(20)
        );
    }
    Ok(())
}

#[rstest]
#[case::nonnullable(vec![Some(0u32)], vec![Some(42i32)], true)]
#[case::nullable(vec![None, Some(0u32)], vec![None, Some(42i32)], false)]
#[case::empty(vec![], vec![], false)]
fn gathered_constantness_requires_values_and_nonnull_indices(
    #[case] indices: Vec<Option<u32>>,
    #[case] values: Vec<Option<i32>>,
    #[case] expected: bool,
) -> VortexResult<()> {
    let source = buffer![42i32, 42].into_array();
    let indices = if indices.iter().all(Option::is_some) {
        PrimitiveArray::from_iter(indices.into_iter().flatten()).into_array()
    } else {
        PrimitiveArray::from_option_iter(indices).into_array()
    };
    let target = PrimitiveArray::from_option_iter(values).into_array();
    let mut ctx = array_session().create_execution_ctx();
    source
        .aggregations()
        .compute_result(&IS_CONSTANT, &mut ctx)?;

    propagate_take_results(&source, &target, &indices)?;
    assert_eq!(
        target.aggregations().get_result_as::<bool>(&IS_CONSTANT)?,
        if expected {
            Precision::Exact(true)
        } else {
            Precision::Absent
        },
    );
    Ok(())
}
