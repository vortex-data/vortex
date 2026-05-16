//! `Limit`: pass through the first N rows of input, then terminate
//! the stream.
//!
//! v2 has no row demand, so this operator does **not** propagate a
//! limit upstream. Once it has emitted N rows it returns
//! `TransformOutput::Finished` and the runtime stops pulling new
//! batches from the source. The source still produces work that
//! the downstream doesn't consume — exactly the v2 trade we
//! explicitly accepted (no back-prop).

use std::task::Poll;

use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;

use crate::Domain;
use crate::DomainSpan;
use crate::EngineResult;
use crate::OutputContract;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, TransformCtx, TransformNode, TransformOutput,
};
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::lowering::{LoweringCtx, PipelineTail};
use crate::physical_plan::plan::Operator;

pub struct Limit {
    label: String,
    input_domain: Domain,
    input_contract: OutputContract,
    limit: u64,
    input: Box<dyn Operator>,
}

impl Limit {
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        input_contract: OutputContract,
        limit: u64,
        input: Box<dyn Operator>,
    ) -> Self {
        Self {
            label: label.into(),
            input_domain,
            input_contract,
            limit,
            input,
        }
    }
}

pub struct LimitNode {
    label: String,
    limit: u64,
}

#[derive(Default)]
pub struct LimitLocal {
    emitted: u64,
    pending: Option<Batch>,
    finished: bool,
}

impl TransformNode for LimitNode {
    type LocalState = LimitLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(LimitLocal::default())
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
        if local.finished {
            return Ok(());
        }
        let remaining = self.limit.saturating_sub(local.emitted);
        if remaining == 0 {
            local.finished = true;
            return Ok(());
        }
        let in_rows = batch.rows() as u64;
        if in_rows <= remaining {
            local.emitted += in_rows;
            if local.emitted == self.limit {
                local.finished = true;
            }
            local.pending = Some(batch);
        } else {
            let take = remaining as usize;
            let mut values = batch.values();
            values.truncate(take);
            let span = DomainSpan::new(batch.span().start(), remaining);
            let array = PrimitiveArray::from_iter(values).into_array();
            local.emitted = self.limit;
            local.finished = true;
            local.pending = Some(Batch::new(array, span));
        }
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

impl Operator for Limit {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let tail = tail.prepend_transform(
            self.input_domain.clone(),
            self.input_contract.clone(),
            LimitNode {
                label: self.label.clone(),
                limit: self.limit,
            },
        );
        self.input.lower(ctx, tail)
    }
}
