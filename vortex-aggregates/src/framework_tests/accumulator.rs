// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::min::Min;
use crate::sum::Sum;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// A sentinel sum of `42.0`, distinguishable from `dict_of_seven()`'s natural sum of `7.0`.
#[derive(Debug)]
struct SentinelSumPartialKernel;
impl DynAggregateKernel for SentinelSumPartialKernel {
    fn aggregate(
        &self,
        _aggregate_fn: &AggregateFnRef,
        _batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        Ok(Some(Scalar::primitive(42.0f64, Nullability::Nullable)))
    }
}

fn fresh_session() -> VortexSession {
    SESSION.clone()
}

fn dict_of_seven() -> ArrayRef {
    DictArray::try_new(buffer![0u32].into_array(), buffer![7.0f64].into_array())
        .expect("valid dictionary")
        .into_array()
}

#[test]
fn cached_sum_precedes_encoding_kernel() -> VortexResult<()> {
    static KERNEL: SentinelSumPartialKernel = SentinelSumPartialKernel;
    let session = fresh_session();
    session
        .get::<AggregateFnSession>()
        .register_aggregate_kernel(Dict.id(), Some(Sum.id()), &KERNEL);
    let mut ctx = session.create_execution_ctx();

    let batch = dict_of_seven();
    batch
        .statistics()
        .set(Stat::Sum, Precision::Exact(ScalarValue::from(7.0f64)));

    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;
    acc.accumulate(&batch, &mut ctx)?;

    assert_eq!(acc.finish()?.as_primitive().as_::<f64>(), Some(7.0));
    Ok(())
}

/// Each batch caches its own result, not the running result and not on the canonical array
/// it executes into.
#[test]
fn caches_result_on_requested_batch() -> VortexResult<()> {
    let mut ctx = fresh_session().create_execution_ctx();
    let dict = |codes: ArrayRef| -> VortexResult<ArrayRef> {
        Ok(DictArray::try_new(codes, buffer![5i32, 1, 9].into_array())?.into_array())
    };
    let batch1 = dict(buffer![0u32, 1, 2].into_array())?;
    let batch2 = dict(buffer![2u32, 0].into_array())?;

    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(Min, NumericalAggregateOpts::default(), dtype)?;
    acc.accumulate(&batch1, &mut ctx)?;
    acc.accumulate(&batch2, &mut ctx)?;

    assert_eq!(
        acc.finish()?,
        Scalar::primitive(1i32, Nullability::Nullable)
    );
    assert_eq!(
        batch1.statistics().get_as::<i32>(Stat::Min),
        Precision::exact(1i32)
    );
    assert_eq!(
        batch2.statistics().get_as::<i32>(Stat::Min),
        Precision::exact(5i32)
    );
    Ok(())
}

#[test]
fn overflowed_sum_is_not_cached() -> VortexResult<()> {
    let mut ctx = fresh_session().create_execution_ctx();
    let batch = buffer![i64::MAX, i64::MAX].into_array();

    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;
    acc.accumulate(&batch, &mut ctx)?;

    assert!(acc.finish()?.is_null());
    assert!(batch.statistics().get(Stat::Sum).as_exact().is_none());
    Ok(())
}

fn sum_i32_accumulator(options: NumericalAggregateOpts) -> VortexResult<AccumulatorRef> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    Ok(Box::new(Accumulator::try_new(Sum, options, dtype)?))
}

#[test]
fn merge_from_drains_other_accumulator() -> VortexResult<()> {
    let mut ctx = fresh_session().create_execution_ctx();
    let mut global = sum_i32_accumulator(NumericalAggregateOpts::default())?;
    let mut local = sum_i32_accumulator(NumericalAggregateOpts::default())?;

    global.accumulate(&buffer![10i32, 20].into_array(), &mut ctx)?;
    local.accumulate(&buffer![5i32].into_array(), &mut ctx)?;
    global.merge_from(local.as_mut())?;

    assert_eq!(
        global.finish()?,
        Scalar::primitive(35i64, Nullability::Nullable)
    );
    // The merged-from accumulator is reset back to the empty state.
    assert_eq!(
        local.finish()?,
        Scalar::primitive(0i64, Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn merge_from_rejects_a_different_aggregate() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut sum = sum_i32_accumulator(NumericalAggregateOpts::default())?;
    let mut min: AccumulatorRef = Box::new(Accumulator::try_new(
        Min,
        NumericalAggregateOpts::default(),
        dtype,
    )?);

    assert!(sum.merge_from(min.as_mut()).is_err());
    Ok(())
}

#[test]
fn merge_from_rejects_mismatched_options() -> VortexResult<()> {
    let mut skipping = sum_i32_accumulator(NumericalAggregateOpts::skip_nans())?;
    let mut including = sum_i32_accumulator(NumericalAggregateOpts::include_nans())?;

    assert!(skipping.merge_from(including.as_mut()).is_err());
    Ok(())
}
