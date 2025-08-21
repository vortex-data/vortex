// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod handle;
mod multithread;
mod tokio;
mod worker;

use crate::runtime::handle::Handle;
use flume::{Receiver, Sender};
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use smol::Executor;
use std::fs::File;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

pub struct Runtime {
    executor: Executor<'static>,

    // I/O queues for reading data.
    file_io_send: Sender<FileIoRequest>,
    file_io_recv: Receiver<FileIoRequest>,
    file_io_exec: Executor<'static>,
}

impl Runtime {
    /// Create a new [`Handle`] for spawning work onto this [`Runtime`].
    pub fn new_handle(&self) -> Handle {
        Handle {
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

pub struct WorkerPool<T> {
    _phantom: PhantomData<T>,
}

impl<T> WorkerPool<T> {
    pub fn new_worker(&self) -> Worker<T> {
        todo!()
    }
}

pub struct Worker<T> {
    _phantom: PhantomData<T>,
}

#[cfg(test)]
mod tests {
    use crate::runtime::handle::Handle;
    use std::fs::File;
    use std::sync::Arc;
    use vortex_buffer::Alignment;

    #[test]
    fn test_spawn() {
        // A runtime does nothing unless we drive it. We want to support three threading models
        // to drive a runtime:
        //
        //  * Drive a stream from multiple worker threads, emitting results out of order.
        //  * Drive a stream from a single worker thread, using background execution threads.
        //  * Drive a stream on a Tokio runtime.

        // Once we create a runtime (possibly pre-configuring the threading model..)
        let runtime = Handle::new();

        // We can then create I/O futures?
        let read = runtime.open_file(Arc::new(File::open("/dev/zero").unwrap()));

        // Now we need to drive the future to completion.
        // Does this use the runtime's configured threading model?
        let result = runtime
            .block_on(read.read(0, 100, Alignment::none()))
            .unwrap();

        // Or, we can drive a stream of futures?
        // let result = runtime.block_on_stream();
    }
}
