//! `ParentChildMin`: for each parent, emit `min(child_value)` over
//! the parent's contiguous child range.
//!
//! Inputs:
//! - port 0 (`offsets`): an `i64` stream of length `P + 1` over
//!   `D_parent`. Element `i` is the start index in the child
//!   domain where parent `i` begins; element `i+1` is its
//!   (exclusive) end.
//! - port 1 (`values`): an `i64` stream over `D_child` carrying
//!   one value per child row.
//!
//! Output: an `i64` stream of length `P` over `D_output` carrying
//! `min(values[offsets[i] .. offsets[i+1]])` for each `i`.
//!
//! v0 implementation: buffer both inputs fully, compute the min
//! pass once both build barriers have fired. Multi-input lowering
//! follows the same pattern as `SortedMergeJoin`.

use std::sync::Arc;
use std::sync::Mutex;
use std::task::Poll;

use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;

use crate::Domain;
use crate::DomainSpan;
use crate::EngineResult;
use crate::OutputContract;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, Parallelism, PendingSend, SinkCtx, SinkNode, SourceCtx,
    SourceNode,
};
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::lowering::{LoweringCtx, LoweringCtxExt, PipelineTail};
use crate::physical_plan::plan::Operator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputSide {
    Offsets,
    Values,
}

#[derive(Default)]
struct ParentChildMinInner {
    offsets: Vec<i64>,
    values: Vec<i64>,
    offsets_done: bool,
    values_done: bool,
}

#[derive(Clone, Default)]
pub struct ParentChildMinState {
    inner: Arc<Mutex<ParentChildMinInner>>,
}

impl ParentChildMinState {
    pub fn push(&self, side: InputSide, values: Vec<i64>) {
        let mut inner = self.inner.lock().unwrap();
        match side {
            InputSide::Offsets => inner.offsets.extend(values),
            InputSide::Values => inner.values.extend(values),
        }
    }

    pub fn close(&self, side: InputSide) {
        let mut inner = self.inner.lock().unwrap();
        match side {
            InputSide::Offsets => inner.offsets_done = true,
            InputSide::Values => inner.values_done = true,
        }
    }

    fn compute(&self) -> Vec<i64> {
        let inner = self.inner.lock().unwrap();
        debug_assert!(inner.offsets_done && inner.values_done);
        let p = inner.offsets.len().saturating_sub(1);
        let mut out = Vec::with_capacity(p);
        for i in 0..p {
            let lo = inner.offsets[i] as usize;
            let hi = inner.offsets[i + 1] as usize;
            let slice = &inner.values[lo..hi];
            let m = slice.iter().copied().min().expect("empty child range");
            out.push(m);
        }
        out
    }
}

// ---- Sink: writes one input side into ParentChildMinState --------

pub struct ParentChildMinSink {
    label: String,
    state: ParentChildMinState,
    side: InputSide,
}

#[derive(Default)]
pub struct ParentChildMinSinkLocal;

impl SinkNode for ParentChildMinSink {
    type LocalState = ParentChildMinSinkLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(ParentChildMinSinkLocal)
    }

    fn poll_send(
        &self,
        _local: &mut Self::LocalState,
        _ctx: &mut SinkCtx<'_, '_>,
        send: &mut PendingSend,
    ) -> OperatorPoll<()> {
        if let Some(batch) = send.take() {
            self.state.push(self.side, batch.values());
        }
        Poll::Ready(Ok(()))
    }

    fn poll_finish(
        &self,
        _local: &mut Self::LocalState,
        _ctx: &mut SinkCtx<'_, '_>,
    ) -> OperatorPoll<()> {
        self.state.close(self.side);
        Poll::Ready(Ok(()))
    }
}

// ---- Source: reads min-per-parent from ParentChildMinState -------

pub struct ParentChildMinSource {
    label: String,
    state: ParentChildMinState,
    batch_rows: usize,
}

pub struct ParentChildMinSourceLocal {
    mins: Option<Vec<i64>>,
    cursor: usize,
}

impl SourceNode for ParentChildMinSource {
    type LocalState = ParentChildMinSourceLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(ParentChildMinSourceLocal {
            mins: None,
            cursor: 0,
        })
    }

    fn poll_next(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>> {
        if local.mins.is_none() {
            local.mins = Some(self.state.compute());
            local.cursor = 0;
        }
        let mins = local.mins.as_ref().unwrap();
        if local.cursor >= mins.len() {
            return Poll::Ready(Ok(None));
        }
        let end = (local.cursor + self.batch_rows).min(mins.len());
        let slice: Vec<i64> = mins[local.cursor..end].to_vec();
        let span = DomainSpan::new(local.cursor as u64, (end - local.cursor) as u64);
        let array = PrimitiveArray::from_iter(slice).into_array();
        local.cursor = end;
        Poll::Ready(Ok(Some(Batch::new(array, span))))
    }
}

// ---- Plan-time Operator -----------------------------------------

pub struct ParentChildMin {
    label: String,
    offsets_domain: Domain,
    offsets_contract: OutputContract,
    offsets: Box<dyn Operator>,
    values_domain: Domain,
    values_contract: OutputContract,
    values: Box<dyn Operator>,
    output_domain: Domain,
    output_contract: OutputContract,
    batch_rows: usize,
}

impl ParentChildMin {
    pub fn new(
        label: impl Into<String>,
        offsets_domain: Domain,
        offsets_contract: OutputContract,
        offsets: Box<dyn Operator>,
        values_domain: Domain,
        values_contract: OutputContract,
        values: Box<dyn Operator>,
        output_domain: Domain,
        output_contract: OutputContract,
    ) -> Self {
        Self {
            label: label.into(),
            offsets_domain,
            offsets_contract,
            offsets,
            values_domain,
            values_contract,
            values,
            output_domain,
            output_contract,
            batch_rows: 64,
        }
    }

    pub fn with_batch_rows(mut self, batch_rows: usize) -> Self {
        assert!(batch_rows > 0, "batch_rows must be positive");
        self.batch_rows = batch_rows;
        self
    }
}

impl Operator for ParentChildMin {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let state = ParentChildMinState::default();
        let offsets_done = ctx.new_pipeline_barrier();
        let values_done = ctx.new_pipeline_barrier();

        ctx.register_domain(self.offsets_domain.clone())?;
        ctx.register_domain(self.values_domain.clone())?;
        ctx.register_domain(self.output_domain.clone())?;

        let offsets_tail = PipelineTail::new(
            self.offsets_domain.clone(),
            self.offsets_contract.clone(),
            ParentChildMinSink {
                label: format!("{}:offsets_sink", self.label),
                state: state.clone(),
                side: InputSide::Offsets,
            },
        )
        .publishes(offsets_done);
        self.offsets.lower(ctx, offsets_tail)?;

        let values_tail = PipelineTail::new(
            self.values_domain.clone(),
            self.values_contract.clone(),
            ParentChildMinSink {
                label: format!("{}:values_sink", self.label),
                state: state.clone(),
                side: InputSide::Values,
            },
        )
        .publishes(values_done);
        self.values.lower(ctx, values_tail)?;

        let out_tail = tail.depends_on(offsets_done).depends_on(values_done);
        ctx.emit_pipeline(
            out_tail,
            self.output_domain.clone(),
            self.output_contract.clone(),
            ParentChildMinSource {
                label: format!("{}:source", self.label),
                state,
                batch_rows: self.batch_rows,
            },
        )?;
        Ok(())
    }
}
