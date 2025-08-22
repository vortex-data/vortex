// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use handle::*;

mod handle;
mod multithread;
mod singlethread;
mod tokio;
pub mod worker;

use async_task::Runnable;
use flume::Receiver;
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
pub struct Runtime {
    /// Queue of scheduling futures.
    sched_recv: Receiver<Runnable>,
    /// Queue of exclusively CPU-bound tasks.
    cpu_recv: Receiver<CpuTask>,
    /// I/O queue for reading data from files.
    io_recv: Receiver<FileIoRequest>,

    /// Queue of pending scheduling tasks.
    sched_pending: Receiver<Runnable>,

    // A handle holds the "submission" side of the runtime.
    handle: Handle,
}

impl Default for Runtime {
    fn default() -> Self {
        let scheduler =
        let (sched_send, sched_recv) = flume::unbounded();
        let (cpu_send, cpu_recv) = flume::unbounded();
        let (io_send, io_recv) = flume::unbounded();

        Self {
            sched_recv,
            cpu_recv,
            io_recv,
            handle: Handle(Arc::new(Inner {
                sched_send,
                cpu_send,
                io_send,
            })),
        }
    }
}

impl Runtime {
    /// Returns a [`Handle`] for spawning work onto this [`Runtime`].
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}

pub trait VortexRead: 'static + Send + Sync {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read;

    // FIXME(ngates): remove this.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;
}

pub(crate) struct CpuTask {
    runnable: Option<Box<dyn FnOnce() + Send + 'static>>,
    // TODO(ngates): we may want worker affinity and other metadata in here?
    //  We may also just want to use an async task Runnable and accept that it's blocking?
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
