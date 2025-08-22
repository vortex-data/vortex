// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{FileIoRequest, Runtime};
use flume::Receiver;
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use smol::lock::Mutex;
use smol::Executor;
use std::sync::Arc;

impl Runtime {
    /// Returns a worker pool that can be used to drive the Runtime and in the process emit
    /// items from the stream.
    pub fn drive_stream_on_pool<T: 'static + Send + Sync>(
        self,
        _stream: impl Stream<Item = T>,
    ) -> WorkerPool<T> {
        todo!()
    }
}

pub struct WorkerPool<T: 'static + Send + Sync> {
    shared: Arc<Shared<T>>,
}

struct Shared<T: 'static + Send + Sync> {
    // The primary executor.
    executor: Arc<Executor<'static>>,

    // The I/O request queue.
    file_io_recv: Receiver<FileIoRequest>,

    // The stream that this worker pool was created to drive.
    // Note that this stream is likely to spawn new scheduling futures onto the runtime.
    stream: Mutex<BoxStream<'static, BoxFuture<'static, T>>>,
}

impl<T: 'static + Send + Sync> WorkerPool<T> {
    pub fn new_worker(&self) -> Worker<T> {
        todo!()
    }
}

pub struct Worker<T: 'static + Send + Sync> {
    pool: Arc<WorkerPool<T>>,
}

/// Implementation of an iterator that actually drives the underlying runtime.
impl<T: 'static + Send + Sync> Iterator for Worker<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
