//! Worked v2 operators. These are intentionally small — enough to
//! validate the lowering API and runtime end-to-end.
//!
//! Each operator implements [`Operator`] (plan-time) and either
//! provides a [`SourceNode`] / [`SinkNode`] (terminal in a pipeline)
//! or a [`TransformNode`] (transparent, prepended onto a tail) for
//! the runtime. All three roles are poll-based; async / CPU
//! offloads go through `ctx.spawn*`.

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
    SourceNode, TransformCtx, TransformNode, TransformOutput,
};
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::lowering::{LoweringCtx, LoweringCtxExt, PipelineTail};
use crate::physical_plan::plan::Operator;

// ---- IntSource ---------------------------------------------------

/// Emits all rows from a fixed `Vec<i64>` in batches of
/// `batch_rows`. Useful for tests and demos.
pub struct IntSource {
    label: String,
    output_domain: Domain,
    output_contract: OutputContract,
    values: Arc<Vec<i64>>,
    batch_rows: usize,
}

impl IntSource {
    pub fn new(
        label: impl Into<String>,
        output_domain: Domain,
        output_contract: OutputContract,
        values: Vec<i64>,
    ) -> Self {
        Self {
            label: label.into(),
            output_domain,
            output_contract,
            values: Arc::new(values),
            batch_rows: 64,
        }
    }

    pub fn with_batch_rows(mut self, batch_rows: usize) -> Self {
        assert!(batch_rows > 0, "batch_rows must be positive");
        self.batch_rows = batch_rows;
        self
    }
}

pub struct IntSourceLocal {
    values: Arc<Vec<i64>>,
    cursor: usize,
    batch_rows: usize,
}

impl SourceNode for IntSource {
    type LocalState = IntSourceLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(IntSourceLocal {
            values: Arc::clone(&self.values),
            cursor: 0,
            batch_rows: self.batch_rows,
        })
    }

    fn poll_next(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>> {
        if local.cursor >= local.values.len() {
            return Poll::Ready(Ok(None));
        }
        let end = (local.cursor + local.batch_rows).min(local.values.len());
        let slice: Vec<i64> = local.values[local.cursor..end].to_vec();
        let span = DomainSpan::new(local.cursor as u64, (end - local.cursor) as u64);
        let array = PrimitiveArray::from_iter(slice).into_array();
        local.cursor = end;
        Poll::Ready(Ok(Some(Batch::new(array, span))))
    }
}

impl Operator for IntSource {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        ctx.register_domain(self.output_domain.clone())?;
        let source = IntSource {
            label: self.label.clone(),
            output_domain: self.output_domain.clone(),
            output_contract: self.output_contract.clone(),
            values: Arc::clone(&self.values),
            batch_rows: self.batch_rows,
        };
        ctx.emit_pipeline(
            tail,
            self.output_domain.clone(),
            self.output_contract.clone(),
            source,
        )?;
        Ok(())
    }
}

// ---- ProjectOne (transform: multiply by a constant) --------------

/// Multiplies every i64 value in the batch by `factor`.
pub struct ProjectOne {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    factor: i64,
    input: Box<dyn Operator>,
}

impl ProjectOne {
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        input_contract: OutputContract,
        factor: i64,
        input: Box<dyn Operator>,
    ) -> Self {
        Self {
            label: label.into(),
            input_domain,
            input_contract,
            factor,
            input,
        }
    }
}

pub struct ProjectOneNode {
    label: String,
    factor: i64,
}

#[derive(Default)]
pub struct ProjectOneLocal {
    pending: Option<Batch>,
    finished: bool,
}

impl TransformNode for ProjectOneNode {
    type LocalState = ProjectOneLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(ProjectOneLocal::default())
    }

    fn can_accept_input(&self, local: &Self::LocalState) -> bool {
        local.pending.is_none() && !local.finished
    }

    fn push_input(
        &self,
        local: &mut Self::LocalState,
        batch: Batch,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        let values: Vec<i64> = batch.values().into_iter().map(|v| v * self.factor).collect();
        let span = batch.span();
        let array = PrimitiveArray::from_iter(values).into_array();
        local.pending = Some(Batch::new(array, span));
        Ok(())
    }

    fn finish_input(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        local.finished = true;
        Ok(())
    }

    fn poll_next_output(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput> {
        if let Some(batch) = local.pending.take() {
            Poll::Ready(Ok(TransformOutput::Batch(batch)))
        } else if local.finished {
            Poll::Ready(Ok(TransformOutput::Finished))
        } else {
            Poll::Ready(Ok(TransformOutput::NeedInput))
        }
    }
}

impl Operator for ProjectOne {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let tail = tail.prepend_transform(
            self.input_domain.clone(),
            self.input_contract.clone(),
            ProjectOneNode {
                label: self.label.clone(),
                factor: self.factor,
            },
        );
        self.input.lower(ctx, tail)
    }
}

// ---- CollectSink -------------------------------------------------

/// Collects all received rows into a shared `Vec<i64>`. Used by
/// tests and demos as the sink.
pub struct CollectSink {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    rows: Arc<Mutex<Vec<i64>>>,
    input: Box<dyn Operator>,
}

impl CollectSink {
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        input_contract: OutputContract,
        rows: Arc<Mutex<Vec<i64>>>,
        input: Box<dyn Operator>,
    ) -> Self {
        Self {
            label: label.into(),
            input_domain,
            input_contract,
            rows,
            input,
        }
    }
}

pub struct CollectSinkNode {
    label: String,
    rows: Arc<Mutex<Vec<i64>>>,
}

#[derive(Default)]
pub struct CollectSinkLocal;

impl SinkNode for CollectSinkNode {
    type LocalState = CollectSinkLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(CollectSinkLocal)
    }

    fn poll_send(
        &self,
        _local: &mut Self::LocalState,
        _ctx: &mut SinkCtx<'_, '_>,
        send: &mut PendingSend,
    ) -> OperatorPoll<()> {
        if let Some(batch) = send.take() {
            let values = batch.values();
            let mut sink = self.rows.lock().unwrap();
            sink.extend(values);
        }
        Poll::Ready(Ok(()))
    }

    fn poll_finish(
        &self,
        _local: &mut Self::LocalState,
        _ctx: &mut SinkCtx<'_, '_>,
    ) -> OperatorPoll<()> {
        Poll::Ready(Ok(()))
    }
}

impl Operator for CollectSink {
    fn lower(&self, _ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        drop(tail);
        Err(crate::physical_plan::error::BuildError::message(
            "CollectSink::lower should not be called directly; use lower_as_root",
        ))
    }
}

impl CollectSink {
    pub fn lower_as_root(&self, ctx: &mut dyn LoweringCtx) -> BuildResult<()> {
        ctx.register_domain(self.input_domain.clone())?;
        let tail = PipelineTail::new(
            self.input_domain.clone(),
            self.input_contract.clone(),
            CollectSinkNode {
                label: self.label.clone(),
                rows: Arc::clone(&self.rows),
            },
        );
        self.input.lower(ctx, tail)
    }
}
