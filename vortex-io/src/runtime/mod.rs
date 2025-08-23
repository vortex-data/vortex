// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use handle::*;

mod handle;
pub mod multithread;
pub mod singlethread;
pub mod tokio;
pub mod worker;

use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use std::fs::File;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use vortex_buffer::{Alignment, ByteBuffer};
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
/// ## Implementation
///
/// The runtime abstraction is largely just a collection of injection queues used to submit the
/// three types of work: I/O, CPU, and scheduling.
///
/// Each threading model has some associated `drive_on_*` methods that take the receiver side of
/// these queues and performs the actual work of driving them to completion.
///
/// The submission end of these queues is accessible via a [`Handle`], which should be cloned and
/// passed around when constructing async futures in Vortex.
pub(crate) trait Runtime: Send + Sync {
    /// Spawns a future to be executed on the runtime's scheduling context.
    ///
    /// The future will continue to be executed in the background and should pass results out via
    /// a one-shot channel if necessary.
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>);

    /// Spawns a CPU-bound task for execution on the runtime.
    fn spawn_cpu(&self, task: CpuTask);

    /// Submits an I/O read request to be executed on the runtime.
    fn spawn_io(&self, request: FileIoRequest);

    // Drive an I/O future to completion on the runtime. Note that I/O futures are `!Send`.
    // fn drive_io(&self, future: LocalBoxFuture<'static, ()>);
}

pub trait VortexRead: 'static + Send + Sync {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read;

    // FIXME(ngates): remove this.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;
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

#[derive(Debug)]
pub(crate) struct FileIoRequest {
    file: Arc<File>,
    offset: u64,
    length: usize,
    alignment: Alignment,
    send: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl FileIoRequest {
    pub(crate) fn resolve(self, result: VortexResult<ByteBuffer>) {
        if let Err(e) = self.send.send(result) {
            log::trace!("Receiver dropped {e}");
        }
    }
}

pub struct Read(ReadState);

impl Read {
    pub fn ready(result: VortexResult<ByteBuffer>) -> Self {
        Read(ReadState::Ready(Some(result)))
    }

    pub fn future() -> (Self, ReadCompletion) {
        let (send, recv) = oneshot::channel();
        (Read(ReadState::Future(recv)), ReadCompletion(send))
    }
}

enum ReadState {
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
                Err(e) => Poll::Ready(Err(vortex_err!("Read failed: {e}"))),
            },
        }
    }
}
