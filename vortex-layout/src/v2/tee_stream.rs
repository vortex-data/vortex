// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`TeeStream`] — multiplex one source array stream into N
//! row-range-keyed subscriber streams.
//!
//! ## Topology
//!
//! ```text
//! source ─────┐
//!             │ (one task)
//!         producer ──┬── kanal channel ──→ subscriber 0 (row range A)
//!                    ├── kanal channel ──→ subscriber 1 (row range B)
//!                    └── kanal channel ──→ subscriber N (row range C)
//! ```
//!
//! Each subscriber registers a `Range<u64>` over the source's row
//! space. The producer task pulls source chunks in order, computes
//! each chunk's cumulative row range, and forwards an Arc-cloned
//! (or sliced) copy to every subscriber whose range intersects.
//! Subscribers receive only chunks intersecting their range, already
//! sliced. Bounded per-subscriber channels provide backpressure.
//!
//! ## Lifecycle
//!
//! 1. [`TeeStream::new`] — wraps a source, holds it pending start.
//! 2. [`TeeStream::subscribe`] — register subscribers (call N times).
//!    All subscribers must register before [`TeeStream::start`].
//! 3. [`TeeStream::start`] — spawns the producer task. Idempotent.
//!
//! For [`crate::v2::let_use::LetPlan`], the wrapping is:
//! `LetPlan::execute` calls `publish_stream` (creating the tee),
//! then `body.execute` (which synchronously calls
//! `UsePlan::execute` → `tee.subscribe` for every consumer in the
//! body subtree), then `tee.start()`. The producer runs from then
//! until source EOF or all subscribers drop.
//!
//! ## Use case
//!
//! Backs streaming [`crate::v2::let_use::LetPlan`] /
//! [`crate::v2::let_use::UsePlan`]: one source plan, many consumers,
//! each interested in a specific row range. Common case after CSE:
//! 16 fields × N chunks all sharing one filter mask, 16N
//! subscribers each interested in one chunk's slice — source still
//! produces N chunks once, fans out to 16 subscribers per chunk.

use std::ops::Range;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use kanal::AsyncReceiver;
use kanal::AsyncSender;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;

// We use unbounded channels because consumers in this codebase
// (notably `AlignedArrayStream`) poll subscriber streams
// sequentially in one task — a bounded channel deadlocks when the
// producer fills the channel of a not-yet-polled subscriber and
// blocks. Memory is bounded by the source's total size; for the
// typical case (filter masks) that's small.

/// Multiplexes one source [`SendableArrayStream`] into N
/// row-range-keyed subscribers via per-subscriber kanal channels.
pub struct TeeStream {
    inner: Arc<Mutex<TeeInner>>,
    dtype: DType,
    handle: Handle,
}

struct TeeInner {
    /// Source stream — moved into the producer task at `start()`.
    source: Option<SendableArrayStream>,
    /// One sender per subscriber.
    senders: Vec<SubscriberSender>,
    /// Producer task handle. `None` until `start()` is called.
    producer: Option<Task<()>>,
}

struct SubscriberSender {
    row_range: Range<u64>,
    sender: AsyncSender<VortexResult<ArrayRef>>,
}

impl TeeStream {
    /// Wrap `source` in a tee. Spawns nothing yet — call
    /// [`Self::subscribe`] for each consumer, then [`Self::start`].
    pub fn new(source: SendableArrayStream, handle: Handle) -> Self {
        let dtype = source.dtype().clone();
        Self {
            inner: Arc::new(Mutex::new(TeeInner {
                source: Some(source),
                senders: Vec::new(),
                producer: None,
            })),
            dtype,
            handle,
        }
    }

    /// The source's [`DType`]. Subscriber streams report the same dtype.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Register a subscriber for the given `row_range`. Returns a
    /// stream that will yield chunks intersecting `row_range`,
    /// already sliced, in source order. Must be called before
    /// [`Self::start`].
    pub fn subscribe(&self, row_range: Range<u64>) -> SendableArrayStream {
        let (sender, receiver) = kanal::unbounded_async();
        self.inner
            .lock()
            .senders
            .push(SubscriberSender { row_range, sender });
        let dtype = self.dtype.clone();
        Box::pin(ArrayStreamAdapter::new(
            dtype,
            stream::unfold(receiver, recv_one),
        ))
    }

    /// Spawn the producer task. Call after all subscribers have
    /// registered. Idempotent — subsequent calls are no-ops.
    pub fn start(&self) {
        let mut inner = self.inner.lock();
        if inner.producer.is_some() {
            return;
        }
        let Some(source) = inner.source.take() else {
            return;
        };
        let inner_arc = Arc::clone(&self.inner);
        let task = self.handle.spawn(producer_task(source, inner_arc));
        inner.producer = Some(task);
    }
}

/// Receive one item from a kanal receiver. Returns `None` when the
/// channel is closed.
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

/// The producer task: pulls source chunks, fans each out to every
/// subscriber whose row range intersects the chunk. Slices on the
/// way out so each subscriber's channel only carries data it asked
/// for.
async fn producer_task(mut source: SendableArrayStream, inner: Arc<Mutex<TeeInner>>) {
    let mut rows_produced: u64 = 0;
    while let Some(item) = source.next().await {
        match item {
            Ok(chunk) => {
                let chunk_len = chunk.len();
                let chunk_len_u64 = chunk_len as u64;
                let chunk_range = rows_produced..(rows_produced + chunk_len_u64);
                rows_produced += chunk_len_u64;

                // Snapshot senders (cheap: clone the AsyncSender +
                // copy the row_range). We hold the lock briefly so
                // subscribers added concurrently aren't seen by this
                // chunk — but for our usage pattern all subscribers
                // are registered before `start()`, so no concurrent
                // adds happen during produce.
                let targets: Vec<_> = {
                    let inner = inner.lock();
                    inner
                        .senders
                        .iter()
                        .filter(|s| {
                            // Overlap with chunk_range.
                            s.row_range.end > chunk_range.start
                                && s.row_range.start < chunk_range.end
                        })
                        .map(|s| (s.row_range.clone(), s.sender.clone()))
                        .collect()
                };

                for (range, sender) in targets {
                    let intersect_start = chunk_range.start.max(range.start);
                    let intersect_end = chunk_range.end.min(range.end);
                    let local_start = usize::try_from(intersect_start - chunk_range.start)
                        .unwrap_or_else(|_| unreachable!("chunk slice offset must fit in usize"));
                    let local_end = usize::try_from(intersect_end - chunk_range.start)
                        .unwrap_or_else(|_| unreachable!("chunk slice offset must fit in usize"));
                    let sliced = if local_start == 0 && local_end == chunk_len {
                        chunk.clone()
                    } else {
                        match chunk.slice(local_start..local_end) {
                            Ok(a) => a,
                            Err(e) => {
                                // Surface the slice error to this
                                // subscriber and move on.
                                let _ = sender.send(Err(e)).await;
                                continue;
                            }
                        }
                    };
                    // If the receiver dropped, send returns an error;
                    // just drop the chunk for that subscriber.
                    let _ = sender.send(Ok(sliced)).await;
                }
            }
            Err(e) => {
                // Forward the error to every subscriber, then drop
                // the senders to close the channels so consumers
                // observe `None` after the error.
                let err = Arc::new(e);
                let senders: Vec<_> = inner
                    .lock()
                    .senders
                    .iter()
                    .map(|s| s.sender.clone())
                    .collect();
                for sender in senders {
                    let _ = sender.send(Err(VortexError::from(Arc::clone(&err)))).await;
                }
                inner.lock().senders.clear();
                return;
            }
        }
    }
    // EOF: dropping the senders closes the channels; consumers see
    // None and terminate.
    inner.lock().senders.clear();
}

#[cfg(test)]
#[allow(deprecated, reason = "tests use to_primitive() to inspect values")]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::StreamExt;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexError;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;

    use super::TeeStream;

    fn dtype() -> DType {
        DType::Primitive(PType::I32, NonNullable)
    }

    fn source_from(chunks: Vec<Vec<i32>>) -> SendableArrayStream {
        let arrays: Vec<_> = chunks
            .into_iter()
            .map(|c| Ok(PrimitiveArray::from_iter(c).into_array()))
            .collect();
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype(), stream::iter(arrays)))
    }

    async fn collect_i32(mut s: SendableArrayStream) -> VortexResult<Vec<i32>> {
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            let arr = item?;
            let buf = arr.to_primitive().into_buffer::<i32>();
            out.extend(buf.iter().copied());
        }
        Ok(out)
    }

    #[test]
    fn full_range_subscribers_see_every_row() -> VortexResult<()> {
        block_on(|handle| async move {
            let source = source_from(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
            let tee = TeeStream::new(source, handle);
            let s1 = tee.subscribe(0..9);
            let s2 = tee.subscribe(0..9);
            tee.start();
            let (r1, r2) = futures::future::join(collect_i32(s1), collect_i32(s2)).await;
            assert_eq!(r1?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
            assert_eq!(r2?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn subscribers_see_only_their_row_range() -> VortexResult<()> {
        block_on(|handle| async move {
            let source = source_from(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
            let tee = TeeStream::new(source, handle);
            let s1 = tee.subscribe(0..3);
            let s2 = tee.subscribe(3..6);
            let s3 = tee.subscribe(6..9);
            tee.start();
            let (r1, r2, r3) =
                futures::future::join3(collect_i32(s1), collect_i32(s2), collect_i32(s3)).await;
            assert_eq!(r1?, vec![1, 2, 3]);
            assert_eq!(r2?, vec![4, 5, 6]);
            assert_eq!(r3?, vec![7, 8, 9]);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn subscribers_get_sliced_chunks_when_range_straddles_boundary() -> VortexResult<()> {
        block_on(|handle| async move {
            let source = source_from(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
            let tee = TeeStream::new(source, handle);
            let s = tee.subscribe(1..7);
            tee.start();
            let mut chunks: Vec<Vec<i32>> = Vec::new();
            let mut s = s;
            while let Some(item) = s.next().await {
                let arr = item?;
                let buf = arr.to_primitive().into_buffer::<i32>();
                chunks.push(buf.iter().copied().collect());
            }
            assert_eq!(chunks, vec![vec![2, 3], vec![4, 5, 6], vec![7]]);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn source_polled_once_per_chunk() -> VortexResult<()> {
        block_on(|handle| async move {
            let polls = Arc::new(AtomicUsize::new(0));
            let polls_for_stream = Arc::clone(&polls);
            let arrays: Vec<_> = vec![vec![1], vec![2], vec![3]]
                .into_iter()
                .map(|c| Ok(PrimitiveArray::from_iter(c).into_array()))
                .collect();
            let counted = stream::iter(arrays).inspect(move |_| {
                polls_for_stream.fetch_add(1, Ordering::SeqCst);
            });
            let source: SendableArrayStream =
                ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype(), counted));

            let tee = TeeStream::new(source, handle);
            let s1 = tee.subscribe(0..3);
            let s2 = tee.subscribe(0..3);
            tee.start();
            let (..) = futures::future::join(collect_i32(s1), collect_i32(s2)).await;
            assert_eq!(polls.load(Ordering::SeqCst), 3);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn errors_propagate_to_subscribers_in_range() -> VortexResult<()> {
        block_on(|handle| async move {
            let chunks: Vec<VortexResult<_>> = vec![
                Ok(PrimitiveArray::from_iter([1i32]).into_array()),
                Err(vortex_error::vortex_err!("boom")),
            ];
            let source: SendableArrayStream =
                ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype(), stream::iter(chunks)));
            let tee = TeeStream::new(source, handle);
            let mut s1 = tee.subscribe(0..2);
            let mut s2 = tee.subscribe(0..2);
            tee.start();
            assert!(s1.next().await.unwrap().is_ok());
            assert!(s2.next().await.unwrap().is_ok());
            assert!(s1.next().await.unwrap().is_err());
            assert!(s2.next().await.unwrap().is_err());
            assert!(s1.next().await.is_none());
            assert!(s2.next().await.is_none());
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }
}
