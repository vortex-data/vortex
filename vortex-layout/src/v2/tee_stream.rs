// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`TeeStream`] — multiplex one source array stream into N independent
//! subscriber streams.
//!
//! Each subscriber sees every chunk in source order, at its own pace.
//! The source is polled at most once per chunk, and chunks are
//! `Arc`-shared (`ArrayRef` is `Arc<dyn Array>`) so the fan-out cost
//! is one Arc-clone per subscriber per chunk.
//!
//! ## Backpressure
//!
//! The producer pauses when ahead of the slowest subscriber by
//! [`Self::DEFAULT_LOOKAHEAD`] chunks. This bounds memory at roughly
//! `lookahead * chunk_size` — a fast subscriber doesn't run away
//! from a slow one and exhaust memory.
//!
//! ## Lazy init
//!
//! The source isn't polled until at least one subscriber is created
//! and polled. A [`TeeStream`] with no subscribers does nothing.
//!
//! ## Use case
//!
//! Backs streaming [`crate::v2::let_use::LetPlan`] / [`crate::v2::let_use::UsePlan`]:
//! one source plan, multiple `Use` consumers, each pulling chunks at
//! its own rate without re-executing the source.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::Stream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexError;
use vortex_error::VortexResult;

/// Multiplexes one source [`SendableArrayStream`] into N subscribers.
///
/// Construct with [`Self::new`], then call [`Self::subscribe`] once
/// per consumer. Each `subscribe` returns a fresh
/// [`SendableArrayStream`] that emits the same sequence of chunks as
/// the source.
pub struct TeeStream {
    state: Arc<Mutex<TeeState>>,
    dtype: DType,
}

impl TeeStream {
    /// How many chunks the producer may run ahead of the slowest
    /// subscriber. Bounds in-flight memory.
    pub const DEFAULT_LOOKAHEAD: usize = 8;

    /// Wrap `source` in a tee.
    pub fn new(source: SendableArrayStream) -> Self {
        Self::with_lookahead(source, Self::DEFAULT_LOOKAHEAD)
    }

    /// Wrap `source` in a tee with a custom lookahead bound. Lookahead
    /// must be at least 1 (otherwise the producer can never run).
    pub fn with_lookahead(source: SendableArrayStream, lookahead: usize) -> Self {
        assert!(lookahead >= 1, "TeeStream lookahead must be ≥ 1");
        let dtype = source.dtype().clone();
        Self {
            state: Arc::new(Mutex::new(TeeState {
                source: Some(source),
                buffer: VecDeque::new(),
                base_position: 0,
                subscribers: Vec::new(),
                source_eof: false,
                source_error: None,
                terminator_position: None,
                lookahead,
            })),
            dtype,
        }
    }

    /// The source's [`DType`]. Subscriber streams report the same dtype.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Create a fresh subscriber stream. Each subscriber sees every
    /// chunk the source produces, in source order, starting from the
    /// next chunk the source emits *or* from the first chunk if no
    /// subscriber has polled yet.
    ///
    /// Subscribing late (after some chunks have already been buffered
    /// and consumed by other subscribers) is **not** supported — new
    /// subscribers always pick up at the current head, missing past
    /// chunks. In practice every subscriber is registered before the
    /// stream is first polled, so this isn't a constraint in real use.
    pub fn subscribe(&self) -> SendableArrayStream {
        let mut state = self.state.lock();
        let id = state.subscribers.len();
        // New subscriber starts at the current head — i.e. the next
        // chunk produced. Catching up to past chunks isn't supported.
        let start_position = state.base_position + state.buffer.len();
        state.subscribers.push(SubscriberState {
            position: start_position,
            waker: None,
            cancelled: false,
        });
        Box::pin(TeeSubscriber {
            id,
            state: Arc::clone(&self.state),
            dtype: self.dtype.clone(),
        })
    }
}

/// Per-subscriber state held inside [`TeeState`].
struct SubscriberState {
    /// Absolute index of the next chunk this subscriber wants.
    position: usize,
    /// Waker registered when this subscriber was last `Pending`.
    /// Woken when new chunks become available or EOF is reached.
    waker: Option<Waker>,
    /// True once the subscriber stream is dropped — the producer
    /// then ignores this subscriber for backpressure / GC purposes.
    cancelled: bool,
}

/// Shared state owned by the [`TeeStream`] and all its subscribers.
struct TeeState {
    /// The source stream. `None` once EOF or error has been observed.
    source: Option<SendableArrayStream>,
    /// Buffered chunks. Index `i` here corresponds to absolute
    /// position `base_position + i`.
    buffer: VecDeque<ArrayRef>,
    /// Absolute index of `buffer[0]`.
    base_position: usize,
    subscribers: Vec<SubscriberState>,
    /// True once the source has yielded `Poll::Ready(None)`. Set
    /// concurrently with `terminator_position` being set.
    source_eof: bool,
    /// First fatal error observed pulling from source. Each
    /// subscriber yields a clone exactly once when it reaches
    /// `terminator_position`.
    source_error: Option<Arc<VortexError>>,
    /// Absolute index at which the source ended (EOF or error).
    /// `None` while the source is still active.
    /// Subscribers whose position equals this value yield the error
    /// (if any) once and advance; subscribers whose position is
    /// strictly greater yield `None`.
    terminator_position: Option<usize>,
    lookahead: usize,
}

impl TeeState {
    /// Drop chunks at the head of the buffer that every live
    /// subscriber has already advanced past. Adjusts `base_position`.
    fn gc(&mut self) {
        let min = self
            .subscribers
            .iter()
            .filter(|s| !s.cancelled)
            .map(|s| s.position)
            .min();
        let Some(min) = min else {
            // No live subscribers — drop everything we've buffered
            // (still cheap; just helps if subscribers are added back).
            self.base_position += self.buffer.len();
            self.buffer.clear();
            return;
        };
        while self.base_position < min && !self.buffer.is_empty() {
            self.buffer.pop_front();
            self.base_position += 1;
        }
    }

    /// True iff the source is at least `lookahead` chunks ahead of
    /// the slowest live subscriber. The producer pauses in that case.
    fn at_lookahead_limit(&self) -> bool {
        let head = self.base_position + self.buffer.len();
        let min_live = self
            .subscribers
            .iter()
            .filter(|s| !s.cancelled)
            .map(|s| s.position)
            .min();
        match min_live {
            Some(min) => head.saturating_sub(min) >= self.lookahead,
            // No live subscribers — don't poll source at all.
            None => true,
        }
    }

    fn wake_all(&mut self) {
        for sub in &mut self.subscribers {
            if let Some(w) = sub.waker.take() {
                w.wake();
            }
        }
    }
}

/// One subscriber's stream view of a [`TeeStream`].
struct TeeSubscriber {
    id: usize,
    state: Arc<Mutex<TeeState>>,
    dtype: DType,
}

impl Stream for TeeSubscriber {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut state = self.state.lock();
        loop {
            // 1. If the chunk we want is already buffered, return it.
            let my_position = state.subscribers[self.id].position;
            if my_position >= state.base_position
                && my_position < state.base_position + state.buffer.len()
            {
                let chunk = state.buffer[my_position - state.base_position].clone();
                state.subscribers[self.id].position = my_position + 1;
                state.gc();
                return Poll::Ready(Some(Ok(chunk)));
            }

            // 2. If the source has terminated, surface the
            //    terminator (error or EOF) exactly once per
            //    subscriber, then yield `None` thereafter.
            if let Some(term_pos) = state.terminator_position {
                if my_position == term_pos {
                    // First time we hit the terminator — advance our
                    // position past it and yield either the error
                    // (cloned) or `None` (EOF).
                    state.subscribers[self.id].position = my_position + 1;
                    return match state.source_error.clone() {
                        Some(err) => Poll::Ready(Some(Err(VortexError::from(err)))),
                        None => Poll::Ready(None),
                    };
                }
                // Past the terminator — clean EOF for this subscriber.
                debug_assert!(my_position > term_pos);
                return Poll::Ready(None);
            }

            // 3. Need to advance the source. If we'd run past the
            //    lookahead limit, register a waker and yield.
            if state.at_lookahead_limit() {
                state.subscribers[self.id].waker = Some(cx.waker().clone());
                return Poll::Pending;
            }

            // 4. Poll the source. We hold the state lock across the
            //    poll — fine because the source is not allowed to
            //    re-enter the tee, and the source's own poll uses
            //    its own waker registration.
            let Some(source) = state.source.as_mut() else {
                // Source already moved out (terminator should be
                // set, but we haven't seen it yet on this iter — try
                // again).
                continue;
            };
            match source.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    state.buffer.push_back(chunk);
                    state.wake_all();
                    continue;
                }
                Poll::Ready(Some(Err(e))) => {
                    state.source_error = Some(Arc::new(e));
                    state.terminator_position = Some(state.base_position + state.buffer.len());
                    state.source = None;
                    state.wake_all();
                    continue;
                }
                Poll::Ready(None) => {
                    state.source_eof = true;
                    state.terminator_position = Some(state.base_position + state.buffer.len());
                    state.source = None;
                    state.wake_all();
                    continue;
                }
                Poll::Pending => {
                    state.subscribers[self.id].waker = Some(cx.waker().clone());
                    return Poll::Pending;
                }
            }
        }
    }
}

impl ArrayStream for TeeSubscriber {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Drop for TeeSubscriber {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.subscribers[self.id].cancelled = true;
        // GC chunks the surviving subscribers no longer need; wake
        // any subscriber that was previously blocked on the dropped
        // one's lookahead.
        state.gc();
        state.wake_all();
    }
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

    /// Helper: collect a stream's chunks as Vec<Vec<i32>>.
    async fn collect_i32(mut s: SendableArrayStream) -> VortexResult<Vec<Vec<i32>>> {
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            let arr = item?;
            let buf = arr.to_primitive().into_buffer::<i32>();
            out.push(buf.iter().copied().collect());
        }
        Ok(out)
    }

    #[test]
    fn two_subscribers_see_same_chunks() -> VortexResult<()> {
        block_on(|_| async move {
            let source = source_from(vec![vec![1, 2], vec![3, 4], vec![5]]);
            let tee = TeeStream::new(source);
            let s1 = tee.subscribe();
            let s2 = tee.subscribe();

            let (r1, r2) = futures::future::join(collect_i32(s1), collect_i32(s2)).await;
            assert_eq!(r1?, vec![vec![1, 2], vec![3, 4], vec![5]]);
            assert_eq!(r2?, vec![vec![1, 2], vec![3, 4], vec![5]]);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn source_polled_once_per_chunk() -> VortexResult<()> {
        block_on(|_| async move {
            let polls = Arc::new(AtomicUsize::new(0));
            let polls_for_stream = Arc::clone(&polls);
            let chunks = vec![vec![1], vec![2], vec![3]];
            let arrays: Vec<_> = chunks
                .into_iter()
                .map(|c| Ok(PrimitiveArray::from_iter(c).into_array()))
                .collect();
            let counted = stream::iter(arrays).inspect(move |_| {
                polls_for_stream.fetch_add(1, Ordering::SeqCst);
            });
            let source: SendableArrayStream =
                ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype(), counted));

            let tee = TeeStream::new(source);
            let s1 = tee.subscribe();
            let s2 = tee.subscribe();
            let (..) = futures::future::join(collect_i32(s1), collect_i32(s2)).await;
            assert_eq!(polls.load(Ordering::SeqCst), 3);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn drop_one_subscriber_unblocks_other() -> VortexResult<()> {
        block_on(|_| async move {
            let source = source_from(vec![vec![1], vec![2], vec![3], vec![4]]);
            let tee = TeeStream::with_lookahead(source, 2);
            let s1 = tee.subscribe();
            let s2 = tee.subscribe();
            // Drop s2 immediately; s1 should still get every chunk.
            drop(s2);
            let r1 = collect_i32(s1).await?;
            assert_eq!(r1, vec![vec![1], vec![2], vec![3], vec![4]]);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn no_subscribers_means_no_source_polling() -> VortexResult<()> {
        // Source that would panic if polled.
        let polled = Arc::new(AtomicUsize::new(0));
        let polled_for_stream = Arc::clone(&polled);
        let chunks: Vec<VortexResult<_>> = vec![Ok(PrimitiveArray::from_iter([1i32]).into_array())];
        let counted = stream::iter(chunks).inspect(move |_| {
            polled_for_stream.fetch_add(1, Ordering::SeqCst);
        });
        let source: SendableArrayStream =
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype(), counted));
        let _tee = TeeStream::new(source);
        // No subscribers, no polling.
        assert_eq!(polled.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn errors_propagate_to_all_subscribers() -> VortexResult<()> {
        block_on(|_| async move {
            let chunks: Vec<VortexResult<_>> = vec![
                Ok(PrimitiveArray::from_iter([1i32]).into_array()),
                Err(vortex_error::vortex_err!("boom")),
            ];
            let source: SendableArrayStream =
                ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype(), stream::iter(chunks)));
            let tee = TeeStream::new(source);
            let mut s1 = tee.subscribe();
            let mut s2 = tee.subscribe();
            // First chunk OK.
            assert!(s1.next().await.unwrap().is_ok());
            assert!(s2.next().await.unwrap().is_ok());
            // Second yields the error to both.
            assert!(s1.next().await.unwrap().is_err());
            assert!(s2.next().await.unwrap().is_err());
            // After the error, both terminate.
            assert!(s1.next().await.is_none());
            assert!(s2.next().await.is_none());
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }
}
