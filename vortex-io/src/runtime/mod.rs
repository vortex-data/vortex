// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use handle::*;
use std::fs::File;
use std::future::ready;
use std::os::unix::fs::FileExt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};

mod handle;
pub mod multithread;
pub mod singlethread;
pub mod tokio;
pub mod worker;

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::FutureExt;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};

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

    /// Submits a stream of I/O tasks to be executed on the runtime.
    /// Note that any concurrency has already been implemented as part of the stream, so the caller
    /// should simply drive the stream and call [`IoTask::run`] on each item.
    fn spawn_io(&self, stream: BoxStream<'static, IoTask>);
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

pub(crate) struct IoTask {
    source: IoSource,
    offset: u64,
    length: usize,
    alignment: Alignment,
    callback: ReadCompletion,
}

#[derive(Clone)]
pub enum IoSource {
    Memory(ByteBuffer),
    File(Arc<File>),
    #[cfg(feature = "object_store")]
    Object {
        store: Arc<dyn object_store::ObjectStore>,
        path: Arc<object_store::path::Path>,
    },
}

impl IoTask {
    // TODO(ngates): ideally this future would be !Send.
    // FIXME(ngates): the future should not be boxed.
    pub(crate) fn run(self) -> impl Future<Output = ()> {
        match self.source {
            IoSource::Memory(buffer) => {
                let offset =
                    usize::try_from(self.offset).vortex_expect("Offset out of bounds for usize");
                let slice = buffer
                    .slice_unaligned(offset..offset + self.length)
                    .aligned(self.alignment);
                self.callback.complete(Ok(slice));
                ready(()).boxed()
            }
            IoSource::File(file) => {
                // TODO(ngates): should we spawn this onto a blocking pool? Possibly.
                let mut buffer = ByteBufferMut::with_capacity_aligned(self.length, self.alignment);
                unsafe { buffer.set_len(self.length) };
                match file.read_exact_at(&mut buffer, self.offset) {
                    Ok(()) => self.callback.complete(Ok(buffer.freeze())),
                    Err(e) => self.callback.complete(Err(VortexError::from(e))),
                }
                ready(()).boxed()
            }
            #[cfg(feature = "object_store")]
            IoSource::Object { store, path } => {
                use futures::TryFutureExt;

                async move {
                    // FIXME(ngates): use get_opts and copy directly into aligned buffer to avoid double copy.
                    let result = store
                        .get_range(&path, self.offset..self.offset + self.length as u64)
                        .map_ok(|data| {
                            let mut buffer =
                                ByteBufferMut::with_capacity_aligned(data.len(), self.alignment);
                            unsafe { buffer.set_len(data.len()) };
                            buffer.as_mut_slice().copy_from_slice(&data);
                            buffer.freeze()
                        })
                        .map_err(VortexError::from)
                        .await;
                    self.callback.complete(result)
                }
                .boxed()
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
