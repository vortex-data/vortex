// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, FileIoRequest, Read, ReadState, VortexRead};
use async_task::{Runnable, Schedule, WithInfo};
use flume::Sender;
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use std::fs::File;
use std::marker::PhantomData;
use std::os::unix::fs::MetadataExt;
use std::sync::Arc;
use vortex_buffer::Alignment;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

/// Represents a handle to a Vortex runtime that can be used to enqueue CPU- or I/O-bound tasks.
///
/// Handles can be thought of like the "send" end of a channel, where the runtime is the "receive"
/// end that is actually driven.
#[derive(Clone)]
pub struct Handle(pub(super) Arc<Inner>);

pub(super) struct Inner {
    pub(super) sched_send: Sender<Runnable>,
    pub(super) sched_schedule: Arc<dyn Schedule + Send + Sync>,
    pub(super) cpu_send: Sender<CpuTask>,
    pub(super) io_send: Sender<FileIoRequest>,
}

impl Handle {
    /// Spawn a new future onto the runtime.
    ///
    /// If the returned future is dropped, the work is cancelled.
    pub fn spawn<F, R>(&self, f: F) -> impl Future<Output = R> + use<F, R>
    where
        F: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        // TODO(ngates): we may want to avoid scheduling back onto the main runtime? But we cannot
        //  push tasks into a queue unless we type erase them...
        let schedule = self.0.sched_schedule.clone();
        let (runnable, task) = async_task::spawn(
            f,
            WithInfo(move |runnable, info| schedule.schedule(runnable, info)),
        );
        self.0
            .sched_send
            .send(runnable)
            .map_err(|e| vortex_err!("Runtime dropped"))
            .vortex_expect("Runtime dropped");
        task
    }

    /// Spawn a CPU-bound task for execution on the runtime.
    pub fn spawn_cpu<F, R>(&self, _f: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        todo!()
    }

    /// Opens a file whose following read requests will occur on the underlying runtime.
    /// TODO(ngates): this API isn't quite right. We want something that takes an IoDriver and
    ///  wraps up requests with some Arc<dyn Any> data that get pushed onto the I/O queue?
    ///  Or maybe, we spawn multiple I/O queues that get driven on the same smol executor?
    pub(crate) fn open_file(&self, file: Arc<File>) -> Arc<dyn VortexRead> {
        Arc::new(FileRead {
            file,
            send: self.0.io_send.clone(),
        })
    }

    #[cfg(feature = "object_store")]
    pub(crate) fn open_object_store(
        &self,
        object_store: Arc<dyn object_store::ObjectStore>,
        path: &object_store::path::Path,
    ) -> Arc<dyn VortexRead> {
        todo!()
    }
}

/// A handle to the result of a spawned CPU task.
///
/// If the handle is dropped prior to the task being executed, it _may_ be skipped.
pub struct TaskHandle<T> {
    _phantom: PhantomData<T>,
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
