// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-memory segment-demand recording and deterministic replay for executor experiments.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use std::time::Duration;
use std::time::Instant;

use futures::FutureExt;
use futures::future;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::segments::ReadAtNowait;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;

/// One segment request at the instant the executor submitted it.
#[derive(Clone, Copy, Debug)]
pub struct IoDemand {
    /// Zero-based request position.
    pub ordinal: usize,
    /// Requested segment.
    pub segment: SegmentId,
    /// Time since recording began.
    pub needed_at: Duration,
}

/// A source wrapper that records exact demand order while serving in-memory bytes immediately.
pub struct RecordingSegmentSource {
    inner: Arc<dyn SegmentSource>,
    started: Instant,
    demands: Mutex<Vec<IoDemand>>,
}

impl RecordingSegmentSource {
    /// Wrap an in-memory source and start a fresh demand trace.
    pub fn new(inner: Arc<dyn SegmentSource>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            started: Instant::now(),
            demands: Mutex::new(Vec::new()),
        })
    }

    /// Snapshot the recorded demand sequence.
    pub fn demands(&self) -> Vec<IoDemand> {
        self.demands.lock().clone()
    }
}

impl SegmentSource for RecordingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let mut demands = self.demands.lock();
        let ordinal = demands.len();
        demands.push(IoDemand {
            ordinal,
            segment: id,
            needed_at: self.started.elapsed(),
        });
        drop(demands);
        self.inner.request(id)
    }

    fn request_nowait(&self, _id: SegmentId) -> VortexResult<ReadAtNowait> {
        // Force every logical need through `request`, where it receives one ordered timestamp.
        Ok(ReadAtNowait::Unsupported)
    }
}

/// An immediate in-memory source that verifies and serves a previously recorded demand order.
pub struct ReplaySegmentSource {
    inner: Arc<dyn SegmentSource>,
    expected: Arc<[SegmentId]>,
    state: Arc<ReplayState>,
}

impl ReplaySegmentSource {
    /// Create a fresh replay cursor over an exact recorded request sequence.
    pub fn new(inner: Arc<dyn SegmentSource>, demands: &[IoDemand]) -> Arc<Self> {
        Arc::new(Self {
            inner,
            expected: demands.iter().map(|demand| demand.segment).collect(),
            state: Arc::new(ReplayState {
                next: AtomicUsize::new(0),
                waiters: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Number of requests consumed from the replay trace.
    pub fn consumed(&self) -> usize {
        self.state.next.load(Ordering::Acquire)
    }
}

impl SegmentSource for ReplaySegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let Some(ordinal) = self.expected.iter().position(|expected| *expected == id) else {
            return future::ready(Err(vortex_err!(
                "I/O replay received unrecorded segment {id:?}"
            )))
            .boxed();
        };
        ReplayFuture {
            ordinal,
            id,
            inner: self.inner.request(id),
            replay: Arc::clone(&self.state),
        }
        .boxed()
    }

    fn request_nowait(&self, _id: SegmentId) -> VortexResult<ReadAtNowait> {
        Ok(ReadAtNowait::Unsupported)
    }
}

struct ReplayState {
    next: AtomicUsize,
    waiters: Mutex<Vec<Waker>>,
}

struct ReplayFuture {
    ordinal: usize,
    id: SegmentId,
    inner: SegmentFuture,
    replay: Arc<ReplayState>,
}

impl Future for ReplayFuture {
    type Output = VortexResult<vortex_array::buffer::BufferHandle>;

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let next = self.replay.next.load(Ordering::Acquire);
        if self.ordinal < next {
            return Poll::Ready(Err(vortex_err!(
                "I/O replay requested segment {:?} more than once",
                self.id
            )));
        }
        if self.ordinal != next {
            let mut waiters = self.replay.waiters.lock();
            if !waiters.iter().any(|waiter| waiter.will_wake(cx.waker())) {
                waiters.push(cx.waker().clone());
            }
            return Poll::Pending;
        }

        match self.inner.poll_unpin(cx) {
            Poll::Ready(result) => {
                self.replay.next.fetch_add(1, Ordering::AcqRel);
                for waiter in std::mem::take(&mut *self.replay.waiters.lock()) {
                    waiter.wake();
                }
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
