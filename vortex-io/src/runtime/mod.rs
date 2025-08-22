// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use handle::*;

mod handle;
mod multithread;
mod singlethread;
mod tokio;
pub mod worker;

use flume::{Receiver, Sender};
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use smol::Executor;
use std::fs::File;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use vortex_buffer::{Alignment, ByteBuffer, Iter};
use vortex_error::{vortex_err, VortexExpect, VortexResult};

pub struct Runtime {
    // The main executor driving our spawned futures.
    executor: Arc<Executor<'static>>,

    // I/O queues for reading data.
    file_io_send: Sender<FileIoRequest>,
    file_io_recv: Receiver<FileIoRequest>,
    file_io_exec: Executor<'static>,
}

impl Default for Runtime {
    fn default() -> Self {
        let (file_io_send, file_io_recv) = flume::unbounded();
        Self {
            executor: Default::default(),
            file_io_send,
            file_io_recv,
            file_io_exec: Default::default(),
        }
    }
}

impl Runtime {
    /// Create a new [`Handle`] for spawning work onto this [`Runtime`].
    // FIXME(ngates): we could hold a Handle on self, and return a cloneable reference?
    pub fn new_handle(&self) -> Handle {
        Handle {
            executor: self.executor.clone(),
            file_io_send: self.file_io_send.clone(),
        }
    }
}

pub trait VortexRead: 'static + Send + Sync {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read;

    // FIXME(ngates): remove this.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;
}

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
