//! Streaming merge-join resource (Velox-shape: asymmetric).
//!
//! `MergeJoinResource::new(label, capacity)` returns:
//!
//! ```text
//!   (right_sink, merge_transform)
//! ```
//!
//! - `right_sink` is a `SinkNode` placed at the tail of the
//!   right-side build pipeline. It forwards each right-side batch
//!   into a bounded channel.
//! - `merge_transform` is a `TransformNode` inserted inline in the
//!   left-side pipeline. It receives left batches via the standard
//!   `push_input` flow, holds them, and reads right batches from
//!   the channel as the merge state machine demands.
//!
//! Two pipelines, one channel, fully streaming. The merge state
//! machine interleaves both sides as cursors advance; no
//! buffer-the-whole-side step.
//!
//! No `select!` is needed. When the transform's `poll_next_output`
//! is called but no right batch is ready, it returns `Poll::Pending`
//! after registering the waker via `ctx.cx()`. The right pipeline's
//! sink, when it next succeeds in `start_send`, fires the waker via
//! the channel's internal queue → lane re-polls → transform now
//! sees the new right batch. The runtime's existing waker plumbing
//! is the select.

use std::sync::Mutex;
use std::task::Poll;

use futures::Sink as FutSink;
use futures::Stream;
use futures::channel::mpsc;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;

use crate::DomainSpan;
use crate::EngineError;
use crate::EngineResult;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, Parallelism, PendingSend, SinkCtx, SinkNode,
    TransformCtx, TransformNode, TransformOutput,
};

/// Factory namespace for constructing the merge-join sink + transform pair.
pub struct MergeJoinResource;

impl MergeJoinResource {
    /// Construct a streaming merge-join resource. The two returned
    /// nodes share a bounded channel internally. Place
    /// `right_sink` at the tail of the right build pipeline;
    /// prepend `merge_transform` onto the tail of the left
    /// pipeline.
    pub fn new(
        label: impl Into<String>,
        capacity: usize,
    ) -> (MergeJoinRightSink, MergeJoinTransform) {
        let label = label.into();
        let (tx, rx) = mpsc::channel::<Batch>(capacity);
        (
            MergeJoinRightSink {
                label: format!("{label}:right_sink"),
                sender: Mutex::new(Some(tx)),
            },
            MergeJoinTransform {
                label: format!("{label}:transform"),
                receiver: Mutex::new(Some(rx)),
            },
        )
    }
}

// ---- Right-side sink: forwards into the channel -----------------

pub struct MergeJoinRightSink {
    label: String,
    sender: Mutex<Option<mpsc::Sender<Batch>>>,
}

pub struct MergeJoinRightSinkLocal {
    sender: Option<mpsc::Sender<Batch>>,
}

impl SinkNode for MergeJoinRightSink {
    type LocalState = MergeJoinRightSinkLocal;

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
            .ok_or_else(|| EngineError::message("MergeJoinRightSink: sender already taken"))?;
        Ok(MergeJoinRightSinkLocal {
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
                "MergeJoinRightSink: send after finish",
            )));
        };
        match FutSink::<Batch>::poll_ready(std::pin::Pin::new(sender), ctx.cx()) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(_)) => Poll::Ready(Err(EngineError::message(
                "merge-join right channel closed (receiver dropped)",
            ))),
            Poll::Ready(Ok(())) => {
                let Some(batch) = send.take() else {
                    return Poll::Ready(Ok(()));
                };
                match std::pin::Pin::new(sender).start_send(batch) {
                    Ok(()) => Poll::Ready(Ok(())),
                    Err(_) => Poll::Ready(Err(EngineError::message(
                        "merge-join right channel closed at start_send",
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
        // Dropping the sender closes the channel — signals end of
        // right input to the transform.
        local.sender.take();
        Poll::Ready(Ok(()))
    }
}

// ---- Merge transform: drives the join state machine -------------

pub struct MergeJoinTransform {
    label: String,
    receiver: Mutex<Option<mpsc::Receiver<Batch>>>,
}

pub struct MergeJoinTransformLocal {
    rx: mpsc::Receiver<Batch>,

    left_held: Option<Batch>,
    left_values: Vec<i64>,
    left_cursor: usize,
    left_done: bool,

    right_held: Option<Batch>,
    right_values: Vec<i64>,
    right_cursor: usize,
    right_done: bool,

    /// While true, we are accumulating an equal-key run on both
    /// sides into `run_left` and `run_right`. The run-emit step
    /// produces a Cartesian product, clears the buffers, and
    /// returns to the comparing state.
    in_run: bool,
    run_key: i64,
    run_left: Vec<i64>,
    run_right: Vec<i64>,
    left_run_done: bool,
    right_run_done: bool,

    /// Output batch ready to be emitted.
    pending_output: Option<Batch>,
}

impl TransformNode for MergeJoinTransform {
    type LocalState = MergeJoinTransformLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn init_local(&self, _runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        let rx = self
            .receiver
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| EngineError::message("MergeJoinTransform: receiver already taken"))?;
        Ok(MergeJoinTransformLocal {
            rx,
            left_held: None,
            left_values: Vec::new(),
            left_cursor: 0,
            left_done: false,
            right_held: None,
            right_values: Vec::new(),
            right_cursor: 0,
            right_done: false,
            in_run: false,
            run_key: 0,
            run_left: Vec::new(),
            run_right: Vec::new(),
            left_run_done: false,
            right_run_done: false,
            pending_output: None,
        })
    }

    fn can_accept_input(&self, local: &Self::LocalState) -> bool {
        // Accept the next left batch only when we've consumed the
        // current one and have nothing pending to emit.
        local.pending_output.is_none()
            && local.left_held.is_none()
            && !local.left_done
    }

    fn push_input(
        &self,
        local: &mut Self::LocalState,
        batch: Batch,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        debug_assert!(local.left_held.is_none());
        local.left_values = batch.values();
        local.left_cursor = 0;
        local.left_held = Some(batch);
        Ok(())
    }

    fn finish_input(
        &self,
        local: &mut Self::LocalState,
        _ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        local.left_done = true;
        Ok(())
    }

    fn poll_next_output(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput> {
        // 1. Drain any output staged on the previous tick.
        if let Some(batch) = local.pending_output.take() {
            return Poll::Ready(Ok(TransformOutput::Batch(batch)));
        }

        loop {
            // --- In-run accumulation -------------------------------
            // While `in_run`, we are collecting all rows on each
            // side matching `run_key` into `run_left`/`run_right`.
            if local.in_run {
                // Extend the left run.
                while !local.left_run_done {
                    if local.left_held.is_some() && local.left_cursor >= local.left_values.len()
                    {
                        local.left_held = None;
                        local.left_values.clear();
                        local.left_cursor = 0;
                    }
                    if local.left_held.is_none() {
                        if local.left_done {
                            local.left_run_done = true;
                            break;
                        }
                        return Poll::Ready(Ok(TransformOutput::NeedInput));
                    }
                    let k = local.left_values[local.left_cursor];
                    if k == local.run_key {
                        local.run_left.push(k);
                        local.left_cursor += 1;
                    } else {
                        local.left_run_done = true;
                        break;
                    }
                }
                // Extend the right run.
                while !local.right_run_done {
                    if local.right_held.is_some()
                        && local.right_cursor >= local.right_values.len()
                    {
                        local.right_held = None;
                        local.right_values.clear();
                        local.right_cursor = 0;
                    }
                    if local.right_held.is_none() {
                        if local.right_done {
                            local.right_run_done = true;
                            break;
                        }
                        // Pull more from the right channel.
                        match std::pin::Pin::new(&mut local.rx).poll_next(ctx.cx()) {
                            Poll::Pending => return Poll::Pending,
                            Poll::Ready(None) => {
                                local.right_done = true;
                                local.right_run_done = true;
                                break;
                            }
                            Poll::Ready(Some(batch)) => {
                                local.right_values = batch.values();
                                local.right_cursor = 0;
                                local.right_held = Some(batch);
                                if local.right_values.is_empty() {
                                    local.right_held = None;
                                    continue;
                                }
                            }
                        }
                    }
                    let k = local.right_values[local.right_cursor];
                    if k == local.run_key {
                        local.run_right.push(k);
                        local.right_cursor += 1;
                    } else {
                        local.right_run_done = true;
                        break;
                    }
                }
                // Both runs captured — emit Cartesian product.
                let mut out_values =
                    Vec::with_capacity(local.run_left.len() * local.run_right.len());
                for &lv in &local.run_left {
                    for _ in 0..local.run_right.len() {
                        out_values.push(lv);
                    }
                }
                local.in_run = false;
                local.left_run_done = false;
                local.right_run_done = false;
                local.run_left.clear();
                local.run_right.clear();
                if !out_values.is_empty() {
                    return Poll::Ready(Ok(TransformOutput::Batch(make_batch(out_values))));
                }
                // Empty Cartesian (one side was empty) — fall
                // through to the comparing phase.
                continue;
            }

            // --- Comparing phase -----------------------------------
            // Ensure both sides have heads (or are exhausted).
            if local.left_held.is_some() && local.left_cursor >= local.left_values.len() {
                local.left_held = None;
                local.left_values.clear();
                local.left_cursor = 0;
            }
            if local.left_held.is_none() {
                if local.left_done {
                    return Poll::Ready(Ok(TransformOutput::Finished));
                }
                return Poll::Ready(Ok(TransformOutput::NeedInput));
            }
            if local.right_held.is_some() && local.right_cursor >= local.right_values.len() {
                local.right_held = None;
                local.right_values.clear();
                local.right_cursor = 0;
            }
            if local.right_held.is_none() && !local.right_done {
                match std::pin::Pin::new(&mut local.rx).poll_next(ctx.cx()) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(None) => local.right_done = true,
                    Poll::Ready(Some(batch)) => {
                        local.right_values = batch.values();
                        local.right_cursor = 0;
                        local.right_held = Some(batch);
                        if local.right_values.is_empty() {
                            local.right_held = None;
                            continue;
                        }
                    }
                }
            }
            // Inner equi-join: if either side is permanently
            // exhausted, no more matches possible.
            if local.right_done && local.right_held.is_none() {
                // Drain remaining left input but produce no more
                // matches; transition to Finished after upstream
                // closes.
                local.left_held = None;
                local.left_values.clear();
                local.left_cursor = 0;
                if local.left_done {
                    return Poll::Ready(Ok(TransformOutput::Finished));
                }
                return Poll::Ready(Ok(TransformOutput::NeedInput));
            }
            // Compare current heads.
            let lk = local.left_values[local.left_cursor];
            let rk = local.right_values[local.right_cursor];
            match lk.cmp(&rk) {
                std::cmp::Ordering::Less => local.left_cursor += 1,
                std::cmp::Ordering::Greater => local.right_cursor += 1,
                std::cmp::Ordering::Equal => {
                    // Enter the in-run accumulation phase.
                    local.in_run = true;
                    local.run_key = lk;
                    local.left_run_done = false;
                    local.right_run_done = false;
                    // Loop continues; the in_run branch above runs next.
                }
            }
        }
    }
}

fn make_batch(out_values: Vec<i64>) -> Batch {
    let span = DomainSpan::from_len(out_values.len() as u64);
    let array = PrimitiveArray::from_iter(out_values).into_array();
    Batch::new(array, span)
}
