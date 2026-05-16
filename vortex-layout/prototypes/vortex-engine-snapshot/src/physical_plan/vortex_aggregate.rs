//! `VortexAggregate`: a TransformNode that wraps a Vortex
//! `Accumulator` and emits the final scalar at end-of-input.
//!
//! Generic over the aggregate function. The transform accumulates
//! every input batch's array directly via Vortex's compute layer
//! (no canonicalization to Vec<i64> — encoded arrays stay encoded
//! when the aggregate kernel supports it).
//!
//! Output: a one-row Batch wrapping the final scalar as a constant
//! array.

use std::sync::Arc;
use std::task::Poll;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_session::VortexSession;

use crate::Domain;
use crate::DomainSpan;
use crate::EngineError;
use crate::EngineResult;
use crate::OutputContract;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, TransformCtx, TransformNode, TransformOutput,
};
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::lowering::{LoweringCtx, PipelineTail};
use crate::physical_plan::plan::Operator;

/// Plan-time `VortexAggregate` operator. Carries the typed
/// aggregate function and options; constructs the accumulator at
/// lane init.
pub struct VortexAggregate<F: AggregateFnVTable> {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    output_domain: Domain,
    output_contract: OutputContract,
    aggregate: F,
    options: F::Options,
    session: VortexSession,
    input: Box<dyn Operator>,
    /// Optional pre-aggregation cast. If set, each incoming batch's
    /// array is cast to this dtype before being passed to
    /// `Accumulator::accumulate`. Useful e.g. to upcast i64 → f64
    /// so `Sum` doesn't saturate on overflow.
    accumulate_dtype: Option<DType>,
}

impl<F> VortexAggregate<F>
where
    F: AggregateFnVTable + 'static,
    F::Options: Clone + Send + Sync + 'static,
{
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        input_contract: OutputContract,
        output_domain: Domain,
        output_contract: OutputContract,
        aggregate: F,
        options: F::Options,
        session: VortexSession,
        input: Box<dyn Operator>,
    ) -> Self {
        Self {
            label: label.into(),
            input_domain,
            input_contract,
            output_domain,
            output_contract,
            aggregate,
            options,
            session,
            input,
            accumulate_dtype: None,
        }
    }

    /// Set an explicit dtype for the accumulator (and an implicit
    /// per-batch cast). Use when the aggregate kernel needs a
    /// different dtype than the input — e.g. f64 accumulation over
    /// i64 input to avoid i64-sum overflow.
    pub fn with_accumulate_dtype(mut self, dtype: DType) -> Self {
        self.accumulate_dtype = Some(dtype);
        self
    }
}

pub struct VortexAggregateNode<F: AggregateFnVTable> {
    label: String,
    aggregate: F,
    options: F::Options,
    input_dtype: DType,
    accumulate_dtype: DType,
    session: VortexSession,
}

pub struct VortexAggregateLocal<F: AggregateFnVTable> {
    accumulator: Accumulator<F>,
    session: VortexSession,
    input_done: bool,
    emitted: bool,
}

impl<F> TransformNode for VortexAggregateNode<F>
where
    F: AggregateFnVTable + Clone + 'static,
    F::Options: Clone + Send + Sync + 'static,
    F::Partial: Send + 'static,
{
    type LocalState = VortexAggregateLocal<F>;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        let accumulator = Accumulator::try_new(
            self.aggregate.clone(),
            self.options.clone(),
            self.accumulate_dtype.clone(),
        )
        .map_err(|e| EngineError::message(format!("accumulator: {e}")))?;
        Ok(VortexAggregateLocal {
            accumulator,
            session: self.session.clone(),
            input_done: false,
            emitted: false,
        })
    }

    fn can_accept_input(&self, local: &Self::LocalState) -> bool {
        !local.input_done
    }

    fn push_input(
        &self,
        local: &mut Self::LocalState,
        batch: Batch,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        let mut array = batch.array().clone();
        // Cast to the accumulator's expected dtype if needed.
        if array.dtype() != &self.accumulate_dtype {
            array = array
                .cast(self.accumulate_dtype.clone())
                .map_err(|e| EngineError::message(format!("cast for accumulate: {e}")))?;
        }
        let mut exec_ctx = local.session.create_execution_ctx();
        local
            .accumulator
            .accumulate(&array, &mut exec_ctx)
            .map_err(|e| EngineError::message(format!("accumulate: {e}")))?;
        Ok(())
    }

    fn finish_input(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        local.input_done = true;
        Ok(())
    }

    fn poll_next_output(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput> {
        if local.emitted {
            return Poll::Ready(Ok(TransformOutput::Finished));
        }
        if !local.input_done {
            return Poll::Ready(Ok(TransformOutput::NeedInput));
        }
        let scalar = match local.accumulator.finish() {
            Ok(s) => s,
            Err(e) => {
                return Poll::Ready(Err(EngineError::message(format!("finish: {e}"))));
            }
        };
        let array = ConstantArray::new(scalar, 1).into_array();
        local.emitted = true;
        let span = DomainSpan::new(0, 1);
        Poll::Ready(Ok(TransformOutput::Batch(Batch::new(array, span))))
    }
}

impl<F> Operator for VortexAggregate<F>
where
    F: AggregateFnVTable + Clone + Send + Sync + 'static,
    F::Options: Clone + Send + Sync + 'static,
    F::Partial: Send + 'static,
{
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        ctx.register_domain(self.output_domain.clone())?;
        let accumulate_dtype = self
            .accumulate_dtype
            .clone()
            .unwrap_or_else(|| self.input_contract.dtype().clone());
        let tail = tail.prepend_transform(
            self.input_domain.clone(),
            self.input_contract.clone(),
            VortexAggregateNode {
                label: self.label.clone(),
                aggregate: self.aggregate.clone(),
                options: self.options.clone(),
                input_dtype: self.input_contract.dtype().clone(),
                accumulate_dtype,
                session: self.session.clone(),
            },
        );
        self.input.lower(ctx, tail)
    }
}

// Force-use Arc to satisfy any future Send bounds — not strictly
// required today since accumulators are constructed per-lane in
// init_local.
#[allow(dead_code)]
fn _assert_traits<F: AggregateFnVTable + Send + Sync + 'static>() {
    fn assert_send<T: Send>() {}
    let _: fn() = assert_send::<VortexAggregateLocal<F>>;
    drop(Arc::new(0)); // keep Arc in scope so the import isn't pruned
}
