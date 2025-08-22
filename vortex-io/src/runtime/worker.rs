// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{FileIoRequest, Runtime};
use flume::Receiver;
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use smol::Executor;
use std::sync::Arc;

impl Runtime {
    /// Returns a worker pool that can be used to drive the Runtime and in the process emit
    /// items from the stream.
    pub fn drive_stream_on_pool<T>(
        self,
        _stream: impl Stream<Item = impl Future<Output = T>>,
    ) -> WorkerPool<T> {
        todo!()
    }
}

pub struct WorkerPool<T> {
    // The primary executor.
    executor: Arc<Executor<'static>>,

    // The stream that this worker pool was created to drive.
    // Note that this stream is likely to spawn new scheduling futures onto the runtime.
    stream: BoxStream<'static, BoxFuture<'static, T>>,

    // The I/O request queue.
    file_io_recv: Receiver<FileIoRequest>,
}

pub struct Worker<T> {
    pool: Arc<WorkerPool<T>>,
}

/// Implementation of an iterator that actually drives the underlying runtime.
impl<T> Iterator for Worker<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
