// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Accumulator dispatch tests that exercise [`Combined`](crate::combined::Combined) through
//! [`Mean`](crate::mean::Mean).

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
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::combined::Combined;
use crate::combined::PairOptions;
use crate::mean::Mean;

fn fresh_session() -> VortexSession {
    crate::test_session()
}

fn dict_of_seven() -> ArrayRef {
    DictArray::try_new(buffer![0u32].into_array(), buffer![7.0f64].into_array())
        .expect("valid dictionary")
        .into_array()
}

/// Mean partial sentinel `{sum: 42.0, count: 1}` — distinguishable from the
/// natural fan-out result `{sum: 7.0, count: 1}` that `Combined::try_accumulate`
/// would produce for `dict_of_seven()`.
#[derive(Debug)]
struct SentinelMeanPartialKernel;
impl DynAggregateKernel for SentinelMeanPartialKernel {
    fn aggregate(
        &self,
        _aggregate_fn: &AggregateFnRef,
        _batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        Ok(Some(sentinel_partial()))
    }
}

/// Returns `Ok(None)` => kernel declined, dispatch falls through.
#[derive(Debug)]
struct DeclineKernel;
impl DynAggregateKernel for DeclineKernel {
    fn aggregate(
        &self,
        _aggregate_fn: &AggregateFnRef,
        _batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        Ok(None)
    }
}

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

fn mean_f64_accumulator() -> VortexResult<Accumulator<Combined<Mean>>> {
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    Accumulator::try_new(
        Mean::combined(),
        PairOptions(
            NumericalAggregateOpts::default(),
            NumericalAggregateOpts::default(),
        ),
        dtype,
    )
}

fn sentinel_partial() -> Scalar {
    let acc = mean_f64_accumulator().expect("build accumulator");
    let sum = Scalar::primitive(42.0f64, Nullability::Nullable);
    let count = Scalar::primitive(1u64, Nullability::NonNullable);
    Scalar::struct_(acc.dtypes().partial_dtype.clone(), vec![sum, count])
}

/// Kernel registered for `(Dict, Combined<Mean>)` fires in preference to
/// `Combined::try_accumulate`'s fan-out path — proves the dispatch reorder.
#[test]
fn combined_kernel_fires() -> VortexResult<()> {
    static KERNEL: SentinelMeanPartialKernel = SentinelMeanPartialKernel;
    let session = fresh_session();
    session
        .get::<AggregateFnSession>()
        .register_aggregate_kernel(Dict.id(), Some(Mean::combined().id()), &KERNEL);
    let mut ctx = session.create_execution_ctx();

    let mut acc = mean_f64_accumulator()?;
    acc.accumulate(&dict_of_seven(), &mut ctx)?;
    let partial = acc.flush()?;

    let s = partial.as_struct();
    assert_eq!(
        s.field("sum").unwrap().as_primitive().as_::<f64>(),
        Some(42.0)
    );
    assert_eq!(
        s.field("count").unwrap().as_primitive().as_::<u64>(),
        Some(1)
    );
    Ok(())
}

/// An execution context snapshots the aggregate kernels on first use, so a kernel registered
/// afterwards only fires through a new context.
#[test]
fn execution_ctx_snapshots_aggregate_kernels_on_first_use() -> VortexResult<()> {
    static KERNEL: SentinelMeanPartialKernel = SentinelMeanPartialKernel;
    let session = fresh_session();
    let mut ctx = session.create_execution_ctx();

    let mut acc = mean_f64_accumulator()?;
    acc.accumulate(&dict_of_seven(), &mut ctx)?;
    assert_eq!(
        acc.flush()?
            .as_struct()
            .field("sum")
            .unwrap()
            .as_primitive()
            .as_::<f64>(),
        Some(7.0)
    );

    session
        .get::<AggregateFnSession>()
        .register_aggregate_kernel(Dict.id(), Some(Mean::combined().id()), &KERNEL);

    acc.accumulate(&dict_of_seven(), &mut ctx)?;
    assert_eq!(
        acc.flush()?
            .as_struct()
            .field("sum")
            .unwrap()
            .as_primitive()
            .as_::<f64>(),
        Some(7.0)
    );

    acc.accumulate(&dict_of_seven(), &mut session.create_execution_ctx())?;
    assert_eq!(acc.flush()?, sentinel_partial());
    Ok(())
}

/// Kernel returns `Ok(None)` => dispatch falls through to `Combined::try_accumulate`'s
/// natural fan-out. The natural partial is `{sum: 7.0, count: 1}`.
#[test]
fn fallback_when_kernel_declines() -> VortexResult<()> {
    static KERNEL: DeclineKernel = DeclineKernel;
    let session = fresh_session();
    session
        .get::<AggregateFnSession>()
        .register_aggregate_kernel(Dict.id(), Some(Mean::combined().id()), &KERNEL);
    let mut ctx = session.create_execution_ctx();

    let mut acc = mean_f64_accumulator()?;
    acc.accumulate(&dict_of_seven(), &mut ctx)?;
    let partial = acc.flush()?;

    let s = partial.as_struct();
    assert_eq!(
        s.field("sum").unwrap().as_primitive().as_::<f64>(),
        Some(7.0)
    );
    assert_eq!(
        s.field("count").unwrap().as_primitive().as_::<u64>(),
        Some(1)
    );
    Ok(())
}

/// A kernel registered for the inner `(Dict, Sum)` child fires when accumulating a
/// Dict batch through `Combined<Mean>`. This is the reusable-primitive case the
/// refactor enables: no `(Dict, Combined<Mean>)` kernel is needed.
#[test]
fn child_kernel_fires_through_combined() -> VortexResult<()> {
    static KERNEL: SentinelSumPartialKernel = SentinelSumPartialKernel;
    let session = fresh_session();
    session
        .get::<AggregateFnSession>()
        .register_aggregate_kernel(Dict.id(), Some(Sum.id()), &KERNEL);
    let mut ctx = session.create_execution_ctx();

    let mut acc = mean_f64_accumulator()?;
    acc.accumulate(&dict_of_seven(), &mut ctx)?;
    let partial = acc.flush()?;

    let s = partial.as_struct();
    // `Sum` child returned the sentinel 42.0 — proves the (Dict, Sum) kernel fired
    // via `Combined<Mean>`'s fan-out. `Count`'s native `try_accumulate` reads the
    // batch's valid_count, so count is the real 1.
    assert_eq!(
        s.field("sum").unwrap().as_primitive().as_::<f64>(),
        Some(42.0)
    );
    assert_eq!(
        s.field("count").unwrap().as_primitive().as_::<u64>(),
        Some(1)
    );
    Ok(())
}

#[test]
fn merge_from_combines_child_accumulators() -> VortexResult<()> {
    let mut ctx = fresh_session().create_execution_ctx();
    let mut global: AccumulatorRef = Box::new(mean_f64_accumulator()?);
    let mut local: AccumulatorRef = Box::new(mean_f64_accumulator()?);

    global.accumulate(&buffer![1.0f64, 2.0].into_array(), &mut ctx)?;
    local.accumulate(&buffer![6.0f64].into_array(), &mut ctx)?;
    global.merge_from(local.as_mut())?;

    assert_eq!(global.finish()?.as_primitive().as_::<f64>(), Some(3.0));
    Ok(())
}
