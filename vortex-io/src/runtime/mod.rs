// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod tokio;

use flume::{Receiver, Sender};
use futures::Stream;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use smol::Executor;
use std::fs::File;
use std::marker::PhantomData;
use std::os::unix::fs::MetadataExt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

pub trait Runtime {
    fn open_file(&self, file: Arc<File>) -> Arc<dyn VortexRead>;

    #[cfg(feature = "object_store")]
    fn open_object_store(
        &self,
        object_store: Arc<dyn object_store::ObjectStore>,
        path: &object_store::path::Path,
    ) -> Arc<dyn VortexRead>;
}

/// A runtime that drives futures from a pool of worker threads.
pub struct WorkerRuntime {
    // The main executor for driving "scheduling" futures on the runtime.
    executor: Executor<'static>,

    // I/O queues for reading data.
    file_io_send: Sender<FileIoRequest>,
    file_io_recv: Receiver<FileIoRequest>,
    file_io_exec: Executor<'static>,
}

impl WorkerRuntime {
    /// Create a worker pool to drive the given stream of futures to completion.
    pub fn spawn_worker_stream<T>(
        stream: impl Stream<Item = impl Future<Output = T>>,
    ) -> WorkerPool<T> {
        todo!()
    }
}

pub trait VortexRead: 'static + Send + Sync {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read;

    // FIXME(ngates): remove this.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;
}

struct FileRead {
    file: Arc<File>,
    send: Sender<FileIoRequest>,
}

impl VortexRead for FileRead {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let (send, recv) = oneshot::channel();
        self.send
            .send(FileIoRequest {
                file: self.file.clone(),
                offset,
                length,
                alignment,
                send,
            })
            .map_err(|e| vortex_err!("Sender dropped: {e}"))
            .vortex_expect("Failed to send read request");
        Read(ReadState::Future(recv))
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        async move { Ok(file.metadata()?.size()) }.boxed()
    }
}

struct FileIoRequest {
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
    use crate::runtime::Runtime;
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
        let runtime = Runtime::new();

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
