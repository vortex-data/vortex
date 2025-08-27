// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use handle::*;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};

mod coalesce;
mod handle;
pub mod multithread;
pub mod singlethread;
pub mod tokio;
pub mod worker;

use crate::runtime::coalesce::CoalescedRequest;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::FutureExt;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

/// A Vortex runtime provides an abstract way of scheduling mixed I/O and CPU workloads onto the
/// various threading models supported by Vortex.
///
/// The models we currently support are:
/// * Single-threaded: all work is driven on the current thread.
/// * Multi-threaded: work is driven on a pool of threads managed by Vortex.
/// * Worker Pool: work is driven on a pool of threads provided by the caller.
/// * Tokio: work is driven on a Tokio runtime provided by the caller.
///
/// Note: users will interact with the [`Handle`] API rather than the [`Runtime`] trait.
///
/// FIXME(ngates): these should really have handles that get dropped and cancel?
pub(crate) trait Runtime: Send + Sync {
    /// Spawns a future to be executed on the runtime's scheduling context.
    ///
    /// The future will continue to be executed in the background and should pass results out via
    /// a one-shot channel if necessary.
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>);

    /// Spawns a CPU-bound task for execution on the runtime.
    fn spawn_cpu(&self, task: CpuTask);

    /// Passes a stream of I/O tasks to be executed on the runtime.
    fn spawn_io(&self, stream: BoxStream<'static, IoTask>, concurrency: usize);
}

pub(crate) struct CpuTask {
    runnable: Box<dyn FnOnce() + Send + 'static>,
    // TODO(ngates): we may want worker affinity and other metadata in here?
    //  We may also just want to use an async task Runnable and accept that it's blocking?
}

impl CpuTask {
    pub(crate) fn run(self) {
        (self.runnable)()
    }
}

// NOTE(ngates): we may well want to make this an enum so we have better control over the common
//  cases of files and object store, we can always have a fallback to arbitrary futures.
pub struct IoTask {
    source: Arc<dyn IoDriver>,
    request: IoReq,
}

impl IoTask {
    pub fn new_request(source: Arc<dyn IoDriver>, request: IoRequest) -> Self {
        IoTask {
            source,
            request: IoReq::Request(request),
        }
    }

    pub fn new_coalesced(source: Arc<dyn IoDriver>, request: CoalescedRequest) -> Self {
        IoTask {
            source,
            request: IoReq::Coalesced(request),
        }
    }
}

enum IoReq {
    Request(IoRequest),
    Coalesced(CoalescedRequest),
}

impl IoTask {
    /// Run this task as a `!Send` future.
    ///
    /// In some cases, this is more optimized than the `Send` version if the calling runtime
    /// supports it.
    pub async fn run_local(self) {
        match self.request {
            IoReq::Request(req) => req.callback.complete(
                self.source
                    .read_local(req.offset, req.length, req.alignment)
                    .await,
            ),
            IoReq::Coalesced(req) => {
                let result = self
                    .source
                    .read_local(
                        req.range.start,
                        usize::try_from(req.range.end - req.range.start)
                            .vortex_expect("too big for usize"),
                        req.alignment,
                    )
                    .await;
                req.resolve(result)
            }
        }
    }

    /// Run this task as a `Send` future.
    pub async fn run_send(self) {
        match self.request {
            IoReq::Request(req) => req.callback.complete(
                self.source
                    .read_send(req.offset, req.length, req.alignment)
                    .await,
            ),
            IoReq::Coalesced(req) => {
                let result = self
                    .source
                    .read_send(
                        req.range.start,
                        usize::try_from(req.range.end - req.range.start)
                            .vortex_expect("too big for usize"),
                        req.alignment,
                    )
                    .await;
                req.resolve(result)
            }
        }
    }
}

pub struct Read(pub(super) ReadState);

impl Read {
    pub fn ready(result: VortexResult<ByteBuffer>) -> Self {
        Read(ReadState::Ready(Some(result)))
    }

    pub fn future() -> (Self, ReadCompletion) {
        let (send, recv) = oneshot::channel();
        (Read(ReadState::Future(recv)), ReadCompletion(send))
    }
}

pub enum ReadState {
    Ready(Option<VortexResult<ByteBuffer>>),
    Future(oneshot::Receiver<VortexResult<ByteBuffer>>),
}

pub struct ReadCompletion(oneshot::Sender<VortexResult<ByteBuffer>>);

impl ReadCompletion {
    pub fn complete(self, result: VortexResult<ByteBuffer>) {
        self.0
            .send(result)
            .map_err(|e| vortex_err!("Sender dropped: {e}"))
            .vortex_expect("Failed to send read completion");
    }
}

impl Future for Read {
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut self.0 {
            ReadState::Ready(maybe_result) => Poll::Ready(
                maybe_result
                    .take()
                    .vortex_expect("Read future polled after completion"),
            ),
            ReadState::Future(fut) => match ready!(fut.poll_unpin(cx)) {
                Ok(result) => Poll::Ready(result),
                Err(e) => Poll::Ready(Err(vortex_err!(
                    "Failed to read from file, IoTask dropped by runtime: {e}"
                ))),
            },
        }
    }
}
