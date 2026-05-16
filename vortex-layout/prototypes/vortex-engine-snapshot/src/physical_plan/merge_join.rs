//! Sorted merge-join operator (v2).
//!
//! Demonstrates the multi-input lowering pattern from
//! `passive-sync-execution.md`:
//!
//! - `Operator::lower` emits three pipelines: left build, right
//!   build, output.
//! - Both build pipelines terminate at a `MergeJoinSink` writing
//!   into shared `MergeJoinState` (an `Arc<Mutex<…>>`).
//! - The output pipeline's `MergeJoinSource` reads from the same
//!   shared state once both build barriers fire.
//!
//! v0 implementation: each side buffers fully before merge.

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
pub enum JoinSide {
    Left,
    Right,
}

#[derive(Default)]
struct MergeJoinInner {
    left: Vec<i64>,
    right: Vec<i64>,
    left_done: bool,
    right_done: bool,
}

#[derive(Clone, Default)]
pub struct MergeJoinState {
    inner: Arc<Mutex<MergeJoinInner>>,
}

impl MergeJoinState {
    pub fn push(&self, side: JoinSide, values: Vec<i64>) {
        let mut inner = self.inner.lock().unwrap();
        match side {
            JoinSide::Left => inner.left.extend(values),
            JoinSide::Right => inner.right.extend(values),
        }
    }

    pub fn close(&self, side: JoinSide) {
        let mut inner = self.inner.lock().unwrap();
        match side {
            JoinSide::Left => inner.left_done = true,
            JoinSide::Right => inner.right_done = true,
        }
    }

    pub fn ready(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.left_done && inner.right_done
    }

    pub fn merged(&self) -> Vec<i64> {
        let inner = self.inner.lock().unwrap();
        merge_sorted(&inner.left, &inner.right)
    }
}

fn merge_sorted(left: &[i64], right: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let li = i;
                while i < left.len() && left[i] == left[li] {
                    i += 1;
                }
                let rj = j;
                while j < right.len() && right[j] == right[rj] {
                    j += 1;
                }
                for ll in li..i {
                    for _rr in rj..j {
                        out.push(left[ll]);
                    }
                }
            }
        }
    }
    out
}

// ---- Sink node: writes one side into MergeJoinState --------------

pub struct MergeJoinSink {
    label: String,
    state: MergeJoinState,
    side: JoinSide,
}

#[derive(Default)]
pub struct MergeJoinSinkLocal;

impl SinkNode for MergeJoinSink {
    type LocalState = MergeJoinSinkLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(MergeJoinSinkLocal)
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

// ---- Source node: reads merged result from MergeJoinState --------

pub struct MergeJoinSource {
    label: String,
    state: MergeJoinState,
}

pub struct MergeJoinSourceLocal {
    merged: Option<Vec<i64>>,
    cursor: usize,
}

impl SourceNode for MergeJoinSource {
    type LocalState = MergeJoinSourceLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        Ok(MergeJoinSourceLocal {
            merged: None,
            cursor: 0,
        })
    }

    fn poll_next(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>> {
        if local.merged.is_none() {
            assert!(
                self.state.ready(),
                "MergeJoinSource polled before both sides finished"
            );
            local.merged = Some(self.state.merged());
            local.cursor = 0;
        }
        let merged = local.merged.as_ref().unwrap();
        if local.cursor >= merged.len() {
            return Poll::Ready(Ok(None));
        }
        let span = DomainSpan::new(local.cursor as u64, (merged.len() - local.cursor) as u64);
        let slice: Vec<i64> = merged[local.cursor..].to_vec();
        let array = PrimitiveArray::from_iter(slice).into_array();
        local.cursor = merged.len();
        Poll::Ready(Ok(Some(Batch::new(array, span))))
    }
}

// ---- Plan-time Operator -----------------------------------------

pub struct SortedMergeJoin {
    label: String,
    left_domain: Domain,
    left_contract: OutputContract,
    left: Box<dyn Operator>,
    right_domain: Domain,
    right_contract: OutputContract,
    right: Box<dyn Operator>,
    output_domain: Domain,
    output_contract: OutputContract,
}

impl SortedMergeJoin {
    pub fn new(
        label: impl Into<String>,
        left_domain: Domain,
        left_contract: OutputContract,
        left: Box<dyn Operator>,
        right_domain: Domain,
        right_contract: OutputContract,
        right: Box<dyn Operator>,
        output_domain: Domain,
        output_contract: OutputContract,
    ) -> Self {
        Self {
            label: label.into(),
            left_domain,
            left_contract,
            left,
            right_domain,
            right_contract,
            right,
            output_domain,
            output_contract,
        }
    }
}

impl Operator for SortedMergeJoin {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        use crate::physical_plan::merge_join_resource::MergeJoinResource;

        ctx.register_domain(self.left_domain.clone())?;
        ctx.register_domain(self.right_domain.clone())?;
        ctx.register_domain(self.output_domain.clone())?;

        // Asymmetric streaming merge-join (Velox-shape):
        //   - right_sink: side pipeline ending at a channel sink.
        //   - merge_transform: inserted inline in the left pipeline
        //     between the left source and the rest of the
        //     downstream chain. Reads from the channel as needed.
        //
        // Channel capacity is 1, matching the left side's single
        // held batch in the transform. The channel exists to
        // decouple the two pipeline drivers, not to prefetch:
        // I/O read-ahead is the source's job, so the right scan's
        // own queue absorbs latency. A deeper channel here would
        // make the right pipeline race ahead with no symmetric
        // mechanism on the left, which is a buffering asymmetry
        // without a corresponding benefit.
        let (right_sink, merge_transform) =
            MergeJoinResource::new(self.label.clone(), /*capacity=*/ 1);

        // Right build pipeline: child → ... → right_sink.
        let right_tail = PipelineTail::new(
            self.right_domain.clone(),
            self.right_contract.clone(),
            right_sink,
        );
        self.right.lower(ctx, right_tail)?;

        // Left/output pipeline: child → ... → merge_transform → tail.sink.
        // The merge transform's input is the left domain; its output
        // is the merge join's output domain (already set on `tail`).
        let left_tail = tail.prepend_transform(
            self.left_domain.clone(),
            self.left_contract.clone(),
            merge_transform,
        );
        self.left.lower(ctx, left_tail)?;
        Ok(())
    }
}
