//! `Gather<N>`: dynamic-expansion fan-in operator.
//!
//! Takes N child operator subtrees. At plan-build time it emits **one**
//! pipeline — the combine pipeline — whose source is a `GatherSource`.
//! The child operators are stashed on the source; at run time the
//! source uses a `PipelineSubmitter` to lower and schedule them in
//! batches of `target_concurrency` (typically `2 × lane_count`).
//!
//! Each submitted child pipeline terminates at a `GatherSink` whose
//! `poll_send` pushes its batch into a shared mpsc channel; the
//! combine pipeline drains that channel. When `pending_children` is
//! empty and all in-flight children's `GatherSink`s have closed
//! their tx clones, the receiver returns `None` and the combine
//! pipeline finalises.
//!
//! Policy is currently `AnyReady` — the source pulls whichever batch
//! arrives first, no cross-child ordering. (An `InOrder` variant with
//! per-channel buffers + a cursor is the natural follow-up.)
//!
//! Compared to a static N-pipeline plan, this gives:
//! - Bounded `init_local` activations (≤ `target_concurrency` files
//!   open / streams running concurrently).
//! - No per-pipeline barrier — admission is just "have we kicked off
//!   the next child yet."
//! - Errors propagate: a child pipeline's `EngineResult::Err` is
//!   surfaced from the `SubmissionFuture` we keep in
//!   `FuturesUnordered`, and the source returns it to the runtime.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::Poll;

use futures::Stream;
use futures::channel::mpsc;
use futures::stream::FuturesUnordered;

use crate::Domain;
use crate::EngineError;
use crate::EngineResult;
use crate::OutputContract;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, Parallelism, PendingSend, SinkCtx, SinkNode, SourceCtx,
    SourceNode,
};
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::lowering::{LoweringCtx, LoweringCtxExt, PipelineTail};
use crate::physical_plan::plan::Operator;
use crate::physical_plan::submitter::{PipelineSubmitter, SubmissionFuture};

/// One child of a `Gather`. The operator's output `(domain, contract)`
/// is carried alongside so the per-child PipelineTail can be built at
/// submission time without re-deriving it.
pub struct GatherInput {
    pub op: Box<dyn Operator>,
    pub output_domain: Domain,
    pub output_contract: OutputContract,
}

impl GatherInput {
    pub fn new(
        op: Box<dyn Operator>,
        output_domain: Domain,
        output_contract: OutputContract,
    ) -> Self {
        Self {
            op,
            output_domain,
            output_contract,
        }
    }
}

/// Plan-time operator. Set `target_concurrency` to roughly
/// `2 × num_workers`; for v0 this is a fixed config field.
pub struct Gather {
    label: String,
    children: Mutex<Option<VecDeque<GatherInput>>>,
    /// Output of the Gather itself (passed to the tail). Children
    /// must all produce data with the same dtype as this contract;
    /// each child's domain is allowed to differ (per-shard).
    output_domain: Domain,
    output_contract: OutputContract,
    target_concurrency: usize,
}

impl Gather {
    pub fn new(
        label: impl Into<String>,
        children: Vec<GatherInput>,
        output_domain: Domain,
        output_contract: OutputContract,
        target_concurrency: usize,
    ) -> Self {
        Self {
            label: label.into(),
            children: Mutex::new(Some(children.into_iter().collect())),
            output_domain,
            output_contract,
            target_concurrency: target_concurrency.max(1),
        }
    }
}

impl Operator for Gather {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        ctx.register_domain(self.output_domain.clone())?;

        // Take ownership of the children — `Operator::lower` is `&self`,
        // so we go through the Mutex<Option<…>> dance. Lowering happens
        // exactly once per Gather instance.
        let children = self
            .children
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                crate::physical_plan::error::BuildError::message(
                    "Gather::lower called more than once",
                )
            })?;

        // Register each child's output domain too so the per-child
        // PipelineTail can use it without re-emitting validation.
        for input in &children {
            ctx.register_domain(input.output_domain.clone())?;
        }

        // Channel sized at target_concurrency: enough buffering to
        // absorb a tick of producers but bounded so producers
        // backpressure when the consumer can't keep up.
        let (tx, rx) = mpsc::channel::<EngineResult<Batch>>(self.target_concurrency);

        let source = GatherSource {
            label: self.label.clone(),
            output_domain: self.output_domain.clone(),
            output_contract: self.output_contract.clone(),
            inner: Mutex::new(Some(GatherSourceInner {
                rx,
                master_tx: Some(tx),
                pending: children,
                target_concurrency: self.target_concurrency,
            })),
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

// ---- Source side: drains the shared mpsc, runs the submission loop --------

pub struct GatherSource {
    label: String,
    output_domain: Domain,
    output_contract: OutputContract,
    inner: Mutex<Option<GatherSourceInner>>,
}

/// Plan-time state moved into the lane on `init_local`.
struct GatherSourceInner {
    rx: mpsc::Receiver<EngineResult<Batch>>,
    /// Master sender we clone into each `GatherSink`. Dropped once
    /// all children have been submitted, so the rx can hit `None`
    /// after the last child's sink also drops its clone.
    master_tx: Option<mpsc::Sender<EngineResult<Batch>>>,
    pending: VecDeque<GatherInput>,
    target_concurrency: usize,
}

pub struct GatherLocal {
    rx: mpsc::Receiver<EngineResult<Batch>>,
    master_tx: Option<mpsc::Sender<EngineResult<Batch>>>,
    pending: VecDeque<GatherInput>,
    inflight: FuturesUnordered<SubmissionFuture>,
    submitter: PipelineSubmitter,
    target_concurrency: usize,
}

impl SourceNode for GatherSource {
    type LocalState = GatherLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        let inner = self
            .inner
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| EngineError::message("Gather::init_local: inner already taken"))?;
        let submitter = runtime
            .submitter()
            .cloned()
            .ok_or_else(|| {
                EngineError::message(
                    "Gather requires a PipelineSubmitter in LocalInitRuntime — \
                     run the plan via `run_plan_blocking[_with_io]` so the runtime can wire one through",
                )
            })?;
        Ok(GatherLocal {
            rx: inner.rx,
            master_tx: inner.master_tx,
            pending: inner.pending,
            inflight: FuturesUnordered::new(),
            submitter,
            target_concurrency: inner.target_concurrency,
        })
    }

    fn poll_next(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>> {
        // 1. Drain any completed in-flight submissions. Surfaces errors
        //    and frees up slots. We loop so we drain everything ready
        //    in one tick.
        loop {
            match Pin::new(&mut local.inflight).poll_next(ctx.cx()) {
                Poll::Pending => break,
                Poll::Ready(None) => break, // empty FuturesUnordered
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                Poll::Ready(Some(Ok(()))) => {}
            }
        }

        // 2. Top up the in-flight set to target_concurrency.
        while local.inflight.len() < local.target_concurrency {
            let Some(input) = local.pending.pop_front() else {
                break;
            };
            let Some(master_tx) = local.master_tx.as_ref() else {
                break;
            };
            let tx = master_tx.clone();
            let sink = GatherSink {
                label: format!("{}:sink", self.label),
                sender: Mutex::new(Some(tx)),
            };
            // Each child's pipeline ends at this GatherSink. The
            // tail's input contract = the child's declared output
            // (per-shard domain + the shared dtype).
            let tail = PipelineTail::new(
                input.output_domain.clone(),
                input.output_contract.clone(),
                sink,
            );
            let fut = local
                .submitter
                .submit(input.op, tail)
                .map_err(|e| {
                    EngineError::message(format!(
                        "Gather: submit child failed: {e}"
                    ))
                })?;
            local.inflight.push(fut);
        }

        // 3. If we've submitted everything we have, drop the master
        //    tx so the receiver can hit `None` once child sinks
        //    finish. Idempotent.
        if local.pending.is_empty() && local.master_tx.is_some() {
            local.master_tx = None;
        }

        // 4. Pull a batch from the shared channel.
        match Pin::new(&mut local.rx).poll_next(ctx.cx()) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // Receiver closed: all senders dropped (= all children
                // finished AND master_tx dropped). Make sure inflight
                // has fully drained before we report EOF; if there are
                // still pending in-flight submissions, we wait for them
                // (their errors might still propagate).
                if local.inflight.is_empty() {
                    Poll::Ready(Ok(None))
                } else {
                    Poll::Pending
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Err(e)),
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Ok(Some(batch))),
        }
    }
}

// ---- Sink side: each submitted child's pipeline tail. ---------------------

/// Sink placed at the tail of each child pipeline. Forwards batches
/// into the Gather's shared mpsc. Holds its own Tx clone; dropping
/// the sink releases the clone (signalling "this child done" to the
/// shared receiver once all such clones are gone).
pub struct GatherSink {
    label: String,
    sender: Mutex<Option<mpsc::Sender<EngineResult<Batch>>>>,
}

pub struct GatherSinkLocal {
    sender: Option<mpsc::Sender<EngineResult<Batch>>>,
}

impl SinkNode for GatherSink {
    type LocalState = GatherSinkLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        let sender = self
            .sender
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| EngineError::message("GatherSink: sender already taken"))?;
        Ok(GatherSinkLocal {
            sender: Some(sender),
        })
    }

    fn poll_send(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut SinkCtx<'_, '_>,
        send: &mut PendingSend,
    ) -> OperatorPoll<()> {
        let Some(sender) = local.sender.as_mut() else {
            return Poll::Ready(Err(EngineError::message(
                "GatherSink: send after finish",
            )));
        };
        use futures::Sink as FutSink;
        match FutSink::<EngineResult<Batch>>::poll_ready(Pin::new(sender), ctx.cx()) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(_)) => Poll::Ready(Err(EngineError::message(
                "GatherSink: receiver dropped",
            ))),
            Poll::Ready(Ok(())) => {
                let Some(batch) = send.take() else {
                    return Poll::Ready(Ok(()));
                };
                match Pin::new(sender).start_send(Ok(batch)) {
                    Ok(()) => Poll::Ready(Ok(())),
                    Err(_) => Poll::Ready(Err(EngineError::message(
                        "GatherSink: receiver dropped at start_send",
                    ))),
                }
            }
        }
    }

    fn poll_finish(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut SinkCtx<'_, '_>,
    ) -> OperatorPoll<()> {
        // Drop our sender clone. Once all child sinks AND the
        // Gather's master_tx have all dropped their senders, the
        // receiver returns `None`.
        local.sender.take();
        Poll::Ready(Ok(()))
    }
}
