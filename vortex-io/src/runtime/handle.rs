// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, FileIoRequest, Read, ReadState, Runtime, VortexRead};
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use std::fs::File;
use std::marker::PhantomData;
use std::os::unix::fs::MetadataExt;
use std::sync::Arc;
use vortex_buffer::Alignment;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

/// Represents a handle to a Vortex runtime that can be used to enqueue CPU- or I/O-bound tasks.
#[derive(Clone)]
pub struct Handle(pub(crate) Arc<dyn Runtime>);

impl Handle {
    /// Spawn a new scheduling future onto the runtime.
    ///
    // TODO(ngates): we should pass a new handle into a function here, then we should use handles
    //  to carry both affinity and priority information back to the runtime.
    //  For example, we can spawn each split of a scan operation. Each spawn on the same handle
    //  creates a sibling task, which have sequential priority. All CPU tasks spawned from the same
    //  handle can have the same affinity? Something like that?
    pub fn spawn<F, R>(&self, f: F) -> impl Future<Output = R> + use<F, R>
    where
        F: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let (send, recv) = oneshot::channel();
        self.0.spawn_scheduling(
            async move {
                if let Err(e) = send.send(f.await) {
                    log::trace!("Failed to send task result: {e}");
                }
            }
            .boxed(),
        );
        async move {
            recv.await
                .map_err(|e| vortex_err!("Failed to await result, runtime dropped: {e}"))
                .vortex_expect("Failed to await result")
        }
    }

    /// Spawn a CPU-bound task for execution on the runtime.
    pub fn spawn_cpu<F, R>(&self, f: F) -> impl Future<Output = R> + Send + 'static
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // TODO(ngates): we want a droppable handle for this.
        let (send, recv) = oneshot::channel();
        self.0.spawn_cpu(CpuTask {
            runnable: Box::new(move || {
                let _ = send.send(f());
            }),
        });
        async move {
            recv.await
                .map_err(|e| vortex_err!("Task cancelled: {e}"))
                .vortex_expect("Task cancelled")
        }
    }

    /// Opens a file whose following read requests will occur on the underlying runtime.
    // TODO(ngates): this API isn't quite right. We want something that takes an IoDriver and
    //  wraps up requests with some Arc<dyn Any> data that get pushed onto the I/O queue?
    //  Or maybe, we spawn multiple I/O queues that get driven on the same smol executor?
    pub(crate) fn open_file(&self, file: Arc<File>) -> Arc<dyn VortexRead> {
        Arc::new(FileRead {
            file,
            runtime: self.0.clone(),
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
    // FIXME(ngates): I kind of want to get rid of the Runtime Sync bound? Do we?
    runtime: Arc<dyn Runtime>,
}

impl VortexRead for FileRead {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let (send, recv) = oneshot::channel();
        self.runtime.spawn_io(FileIoRequest {
            file: self.file.clone(),
            offset,
            length,
            alignment,
            send,
        });
        Read(ReadState::Future(recv))
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        async move { Ok(file.metadata()?.size()) }.boxed()
    }
}
