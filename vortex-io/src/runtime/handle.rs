// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, IoSource, IoTask, Read, Runtime};
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use std::fs::File;
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

    pub fn open(&self, source: IoSource) -> FileIo {
        let (send, recv) = flume::unbounded();
        self.0.spawn_io(recv.into_stream().boxed());
        FileIo { source, send }
    }

    /// Opens a file whose following read requests will occur on the underlying runtime.
    // TODO(ngates): this API isn't quite right. We want something that takes an IoDriver and
    //  wraps up requests with some Arc<dyn Any> data that get pushed onto the I/O queue?
    //  Or maybe, we spawn multiple I/O queues that get driven on the same smol executor?
    //
    // FIXME(ngates): this API can create a channel that is used for the entire lifetime of the
    //  file. We can then pass the other end of the channel to the runtime.
    pub fn open_file(&self, read: Arc<File>) -> FileIo {
        self.open(IoSource::File(read))
    }

    // FIXME(ngates): similar to open-file, this creates a channel. We can decide whether or not
    //  the channel is one per file, or one per object store? Probably the latter. In which case,
    //  we kind of need a custom VortexObjectStore struct to pass back. Alternatively, we create
    //  our own ObjectStore impl that holds a handle, but it's easy to misuse since its the same
    //  type.
    //
    // The problem is, the scope of something like this within e.g. DataFusion is difficult to
    // manage. We kind of want to scope the S3 object store to the DataFusion session. But we
    // currently launch a new Vortex handle for each scan. If we use handles to manage scoping
    // and priority of tasks, then we need the object store queue to be separated from that.
    #[cfg(feature = "object_store")]
    pub fn open_object_store(
        &self,
        store: Arc<dyn object_store::ObjectStore>,
        path: Arc<object_store::path::Path>,
    ) -> FileIo {
        self.open(IoSource::Object { store, path })
    }
}

/// A file that can be read from using a Vortex runtime.
///
/// This essentially provides a wrapper to bind a handle to a read interface. It is optional, but
/// should be used carefully because the subsequent read operations must be driven on the same
/// runtime.
#[derive(Clone)]
pub struct FileIo {
    source: IoSource,
    send: flume::Sender<IoTask>,
}

impl FileIo {
    pub fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let (read, callback) = Read::future();
        // FIXME(ngates): are these queues bounded? If so we should await here.
        self.send
            .send(IoTask {
                source: self.source.clone(),
                offset,
                length,
                alignment,
                callback,
            })
            .map_err(|e| vortex_err!("Runtime dropped {e}"))
            .vortex_expect("File read");
        read
    }

    // FIXME(ngates): non-static future
    pub fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        match &self.source {
            IoSource::Memory(buffer) => {
                let size = buffer.len() as u64;
                async move { Ok(size) }.boxed()
            }
            IoSource::File(file) => {
                let file = file.clone();
                async move { Ok(file.metadata()?.len()) }.boxed()
            }
            #[cfg(feature = "object_store")]
            IoSource::Object { store, path } => {
                let store = store.clone();
                let path = path.clone();
                async move { Ok(store.head(&path).await?.size) }.boxed()
            }
        }
    }
}
