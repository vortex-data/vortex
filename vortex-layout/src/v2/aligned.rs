// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`AlignedArrayStream`] — a k-way zip of source streams that emits
//! row-aligned slices.
//!
//! ## Pipelined model
//!
//! Each child source runs in its own producer task that pushes
//! chunks into a bounded kanal channel. The zip's `poll_next` reads
//! from those channels, never directly polls the source streams.
//! This means children's I/O runs in parallel — a slow child no
//! longer halts the others. Bounded channels (~4 chunks) provide
//! backpressure.
//!
//! Why this matters: the alignment naturally serialises emit
//! decisions (every step needs the smallest currently-available
//! length across all children), but there's no reason the children
//! themselves have to be polled sequentially. Each child can run
//! its own pipeline up to the channel-buffer depth ahead of where
//! the zip is consuming.
//!
//! ## Producer task lifecycle
//!
//! Tasks are held by `_producers: Vec<Task<()>>` and aborted on Drop.
//! When the AlignedArrayStream is dropped, its receivers are dropped,
//! producer sends fail with `SendError::ReceiveClosed`, the producer
//! task exits cleanly. The abort on Task drop is a backstop in case
//! the producer task is mid-poll on the source.
//!
//! ## Alignment
//!
//! Each step picks the smallest currently-available row count across
//! children, slices each to that length, emits them. Larger remaining
//! slices stay buffered. Optionally, callers can request a minimum
//! batch size via `AlignedArrayStream::with_min_rows` — the stream concats
//! per-child until each has at least that many rows queued (or hits
//! EOF), trading a copy for fewer/larger emitted batches.

#![allow(clippy::cognitive_complexity)]

use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::time::Instant;

use futures::Stream;
use futures::StreamExt as _;
use kanal::AsyncReceiver;
use vortex_array::ArrayRef;
use vortex_array::IntoArray as _;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;

use crate::v2::experiment::trace_flow;

/// How many chunks each child's producer may run ahead of the zip.
/// Bounded by the channel capacity — backpressure kicks in when the
/// zip is slower than the child can produce.
const CHILD_BUFFER_DEPTH: usize = 4;

/// K-way row-aligned zip. See module docs.
pub struct AlignedArrayStream {
    id: u64,
    label: &'static str,
    children: Vec<ChildState>,
    /// Optional minimum batch size hint — see [`Self::with_min_rows`].
    min_rows: Option<usize>,
    /// Producer tasks, one per child. Held to keep them alive; Drop
    /// aborts them. The producers also exit naturally when the zip
    /// drops the receivers (channel closes, send fails).
    _producers: Vec<Task<()>>,
}

struct ChildState {
    /// Wrapped receiver — implements `Stream<Item = VortexResult<ArrayRef>>`
    /// by pulling from the kanal channel.
    receiver_stream: SendableArrayStream,
    /// Output dtype (used when concatenating multiple buffered batches).
    dtype: DType,
    /// Currently-buffered head array.
    head: Option<ArrayRef>,
    /// True once the receiver_stream has yielded `Poll::Ready(None)` —
    /// further polls would re-enter `unfold` and panic.
    eof: bool,
}

impl AlignedArrayStream {
    /// Construct a pipelined k-way zip. Each child stream is moved
    /// into a producer task spawned via `handle`; the resulting
    /// receivers feed the zip.
    pub fn new(children: Vec<SendableArrayStream>, handle: Handle) -> Self {
        Self::new_labeled(children, handle, "aligned")
    }

    /// Construct a pipelined k-way zip with a trace label identifying
    /// the plan node that owns the alignment.
    pub fn new_labeled(
        children: Vec<SendableArrayStream>,
        handle: Handle,
        label: &'static str,
    ) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut child_states = Vec::with_capacity(children.len());
        let mut producers = Vec::with_capacity(children.len());
        let buffer_depth = CHILD_BUFFER_DEPTH;
        let child_count = children.len();
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                aligned_id = id,
                aligned_label = label,
                child_count,
                buffer_depth,
                "aligned new"
            );
        }
        for (child_idx, child) in children.into_iter().enumerate() {
            let dtype = child.dtype().clone();
            let (sender, receiver) = kanal::bounded_async(buffer_depth);
            let task = handle.spawn(producer_task(id, label, child_idx, child, sender));
            let receiver_stream: SendableArrayStream = Box::pin(ArrayStreamAdapter::new(
                dtype.clone(),
                futures::stream::unfold(receiver, recv_one),
            ));
            child_states.push(ChildState {
                receiver_stream,
                dtype,
                head: None,
                eof: false,
            });
            producers.push(task);
        }
        Self {
            id,
            label,
            children: child_states,
            min_rows: None,
            _producers: producers,
        }
    }

    /// Try to emit at least `min_rows` rows per step (concatenating
    /// per-child as needed). Short final batches are still emitted
    /// when one or more children hit EOF.
    pub fn with_min_rows(mut self, min_rows: usize) -> Self {
        self.min_rows = Some(min_rows.max(1));
        self
    }

    /// Number of children being zipped.
    pub fn arity(&self) -> usize {
        self.children.len()
    }
}

/// Drain `source` into `sender`. Exits cleanly when source EOFs or
/// when the receiver closes (consumer dropped the channel).
async fn producer_task(
    aligned_id: u64,
    aligned_label: &'static str,
    child_idx: usize,
    mut source: SendableArrayStream,
    sender: kanal::AsyncSender<VortexResult<ArrayRef>>,
) {
    let trace = trace_flow();
    while let Some(item) = source.next().await {
        let rows = item.as_ref().map_or(0, |array| array.len());
        let send_start = Instant::now();
        if sender.send(item).await.is_err() {
            if trace {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    aligned_id,
                    aligned_label,
                    child_idx,
                    rows,
                    "aligned producer receiver closed"
                );
            }
            return;
        }
        let send_elapsed = send_start.elapsed();
        if trace {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                aligned_id,
                aligned_label,
                child_idx,
                rows,
                send_elapsed_ms = send_elapsed.as_secs_f64() * 1000.0,
                "aligned producer sent"
            );
        }
    }
    if trace {
        tracing::debug!(
            target: "vortex_layout::v2::flow",
            aligned_id,
            aligned_label,
            child_idx,
            "aligned producer eof"
        );
    }
}

/// Stream-unfold step for a kanal receiver.
async fn recv_one(
    recv: AsyncReceiver<VortexResult<ArrayRef>>,
) -> Option<(
    VortexResult<ArrayRef>,
    AsyncReceiver<VortexResult<ArrayRef>>,
)> {
    match recv.recv().await {
        Ok(item) => Some((item, recv)),
        Err(_) => None,
    }
}

impl Stream for AlignedArrayStream {
    type Item = VortexResult<Vec<ArrayRef>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let target = this.min_rows.unwrap_or(1);
        let trace = trace_flow();

        // Step 1: refill each child's head until it has at least
        // `target` rows OR the receiver-stream EOFs. Polling each
        // receiver-stream is cheap (channel pop); the actual source
        // I/O happened concurrently in the producer task.
        for (child_idx, child) in this.children.iter_mut().enumerate() {
            loop {
                let buffered = child.head.as_ref().map_or(0, |a| a.len());
                if buffered >= target || child.eof {
                    break;
                }
                match child.receiver_stream.poll_next_unpin(cx) {
                    Poll::Pending => {
                        if trace {
                            tracing::debug!(
                                target: "vortex_layout::v2::flow",
                                aligned_id = this.id,
                                aligned_label = this.label,
                                child_idx,
                                buffered,
                                target,
                                "aligned pending child"
                            );
                        }
                        return Poll::Pending;
                    }
                    Poll::Ready(None) => {
                        child.eof = true;
                        break;
                    }
                    Poll::Ready(Some(Err(err))) => {
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Ready(Some(Ok(arr))) => {
                        if arr.is_empty() {
                            continue;
                        }
                        child.head = Some(match child.head.take() {
                            None => arr,
                            Some(prev) => {
                                let chunked = match ChunkedArray::try_new(
                                    vec![prev, arr],
                                    child.dtype.clone(),
                                ) {
                                    Ok(c) => c,
                                    Err(err) => {
                                        return Poll::Ready(Some(Err(err)));
                                    }
                                };
                                chunked.into_array()
                            }
                        });
                    }
                }
            }
        }

        // Step 2: decide emit length = min head len across children.
        // A child whose head is `None` after refill has no more rows;
        // the aligned stream is done.
        let mut emit_len = usize::MAX;
        for child in &this.children {
            match child.head.as_ref() {
                Some(arr) => emit_len = emit_len.min(arr.len()),
                None => return Poll::Ready(None),
            }
        }
        if emit_len == 0 {
            return Poll::Ready(None);
        }

        // Step 3: take `emit_len` rows from each child's head.
        let mut output = Vec::with_capacity(this.children.len());
        for child in this.children.iter_mut() {
            let head = child
                .head
                .take()
                .ok_or_else(|| vortex_err!("AlignedArrayStream: head missing after refill"));
            let head = match head {
                Ok(h) => h,
                Err(e) => return Poll::Ready(Some(Err(e))),
            };
            if head.len() == emit_len {
                output.push(head);
            } else {
                let front = match head.slice(0..emit_len) {
                    Ok(a) => a,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                };
                let rest = match head.slice(emit_len..head.len()) {
                    Ok(a) => a,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                };
                child.head = Some(rest);
                output.push(front);
            }
        }

        if trace {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                aligned_id = this.id,
                aligned_label = this.label,
                emit_len,
                child_count = this.children.len(),
                target,
                "aligned emit"
            );
        }

        Poll::Ready(Some(Ok(output)))
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;

    use super::AlignedArrayStream;

    fn primitive_dtype() -> DType {
        DType::Primitive(PType::I32, NonNullable)
    }

    fn stream_of(parts: Vec<Vec<i32>>) -> SendableArrayStream {
        let dtype = primitive_dtype();
        let arrays: Vec<_> = parts
            .into_iter()
            .map(|part| Ok(PrimitiveArray::from_iter(part).into_array()))
            .collect();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream::iter(arrays)))
    }

    #[test]
    fn aligns_misaligned_children() -> VortexResult<()> {
        block_on(|handle| async move {
            let left = stream_of(vec![vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
            let right = stream_of(vec![vec![11, 12, 13, 14], vec![15, 16, 17, 18, 19, 20]]);

            let mut s = AlignedArrayStream::new(vec![left, right], handle);
            let mut steps: Vec<(usize, usize)> = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            assert_eq!(steps, vec![(4, 4), (6, 6)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn ends_when_any_child_eof_with_no_remainder() -> VortexResult<()> {
        block_on(|handle| async move {
            let left = stream_of(vec![vec![1, 2, 3]]);
            let right = stream_of(vec![vec![10, 11, 12, 13, 14]]);
            let mut s = AlignedArrayStream::new(vec![left, right], handle);
            let mut steps = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            assert_eq!(steps, vec![(3, 3)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn min_rows_concats_until_threshold() -> VortexResult<()> {
        block_on(|handle| async move {
            let left = stream_of(vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
                vec![7, 8, 9],
                vec![10, 11, 12],
            ]);
            let right = stream_of(vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
                vec![7, 8, 9],
                vec![10, 11, 12],
            ]);

            let mut s = AlignedArrayStream::new(vec![left, right], handle).with_min_rows(8);
            let mut steps = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            assert_eq!(steps, vec![(9, 9), (3, 3)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn skips_empty_inner_arrays() -> VortexResult<()> {
        block_on(|handle| async move {
            let left = stream_of(vec![vec![], vec![1, 2, 3], vec![]]);
            let right = stream_of(vec![vec![10, 20, 30]]);
            let mut s = AlignedArrayStream::new(vec![left, right], handle);
            let mut steps = Vec::new();
            while let Some(item) = s.next().await {
                let pair = item?;
                steps.push((pair[0].len(), pair[1].len()));
            }
            assert_eq!(steps, vec![(3, 3)]);
            Ok::<_, vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn arity_matches_children() {
        block_on(|handle| async move {
            let s = AlignedArrayStream::new(
                vec![
                    stream_of(vec![vec![1]]),
                    stream_of(vec![vec![1]]),
                    stream_of(vec![vec![1]]),
                ],
                handle,
            );
            assert_eq!(s.arity(), 3);
        })
    }
}
