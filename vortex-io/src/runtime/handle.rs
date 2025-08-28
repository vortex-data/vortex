// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::coalesce::{CoalescedRequest, CoalescedStreamExt};
use crate::runtime::{CpuTask, IoTask, Read, ReadCompletion, Runtime};
use futures::future::{BoxFuture, LocalBoxFuture, Shared};
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryFutureExt};
use std::fs::File;
use std::marker::PhantomData;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{
    vortex_err, vortex_panic, SharedVortexResult, VortexError, VortexExpect, VortexResult,
};

/// Represents a handle to a Vortex runtime that can be used to enqueue CPU- or I/O-bound tasks.
#[derive(Clone)]
pub struct Handle<'rt>(pub(crate) Arc<dyn Runtime<'rt> + 'rt>);

impl Handle<'static> {
    // FIXME(ngates): remove this!
    pub fn no_op() -> Self {
        struct NoOp;

        impl Runtime<'static> for NoOp {
            fn spawn_scheduling(&self, _fut: BoxFuture<'static, ()>) {
                vortex_panic!("NoOp runtime cannot spawn tasks");
            }

            fn spawn_cpu(&self, _task: CpuTask) {
                vortex_panic!("NoOp runtime cannot spawn CPU tasks");
            }

            fn spawn_io(&self, _stream: BoxStream<'static, IoTask>, _concurrency: usize) {
                vortex_panic!("NoOp runtime cannot spawn I/O tasks");
            }
        }

        Self(Arc::new(NoOp))
    }
}

impl<'rt> Handle<'rt> {
    /// Spawn a new scheduling future onto the runtime.
    ///
    // TODO(ngates): we should pass a new handle into a function here, then we should use handles
    //  to carry both affinity and priority information back to the runtime.
    //  For example, we can spawn each split of a scan operation. Each spawn on the same handle
    //  creates a sibling task, which have sequential priority. All CPU tasks spawned from the same
    //  handle can have the same affinity? Something like that?
    pub fn spawn<F, R>(&self, f: F) -> impl Future<Output = R> + use<'rt, F, R>
    where
        F: Future<Output = R> + Send + 'rt,
        R: Send + 'rt,
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
    pub fn spawn_cpu<F, R>(&self, f: F) -> impl Future<Output = R> + Send + 'rt
    where
        // Unlike scheduling futures, the CPU task should have a static lifetime because it
        // doesn't need to access to handle to spawn more work.
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
    //
    // FIXME(ngates): this API can create a channel that is used for the entire lifetime of the
    //  file. We can then pass the other end of the channel to the runtime.
    pub fn open_file(&self, read: Arc<File>) -> FileIo<'rt> {
        self.open(read)
    }

    pub fn open(&self, driver: Arc<dyn IoSource>) -> FileIo<'rt> {
        let (send, recv) = flume::unbounded();

        // Construct the size future in case we need it.
        let size = driver.size();

        let concurrency = driver.concurrency();
        let name = driver.name();

        let stream = recv.into_stream();
        let stream = match driver.coalescing_window() {
            None => stream
                .map(move |req: IoRequest| IoTask::new_request(driver.clone(), req))
                .boxed(),
            Some(window) => stream
                .coalesce(window)
                .map(move |req: CoalescedRequest| IoTask::new_coalesced(driver.clone(), req))
                .boxed(),
        };

        self.0.spawn_io(stream, concurrency);

        FileIo {
            name,
            size,
            send,
            _phantom: Default::default(),
        }
    }
}

pub trait IoSource: Send + Sync + 'static {
    fn name(&self) -> String;

    fn coalescing_window(&self) -> Option<u64>;

    fn concurrency(&self) -> usize;

    /// Returns a shared future that resolves to the byte size of the underlying data source.
    fn size(&self) -> Shared<BoxFuture<'static, SharedVortexResult<u64>>>;

    /// Perform a single read operation.
    ///
    /// The returned future must be `Send`, and should not require a specific runtime to drive it.
    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>>;

    fn read_local(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> LocalBoxFuture<'static, VortexResult<ByteBuffer>> {
        self.read_send(offset, length, alignment).boxed_local()
    }
}

impl IoSource for ByteBuffer {
    fn name(&self) -> String {
        format!("ByteBuffer({})", self.len())
    }

    fn coalescing_window(&self) -> Option<u64> {
        None
    }

    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> Shared<BoxFuture<'static, SharedVortexResult<u64>>> {
        let len = self.len() as u64;
        async move { Ok(len) }.boxed().shared()
    }

    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let buffer = self.clone();
        async move {
            if offset + length as u64 > buffer.len() as u64 {
                return Err(vortex_err!("Read out of bounds"));
            }
            let mut slice = ByteBufferMut::with_capacity_aligned(length, alignment);
            unsafe { slice.set_len(length) };
            slice
                .as_mut_slice()
                .copy_from_slice(&buffer.as_slice()[offset as usize..offset as usize + length]);
            Ok(slice.freeze())
        }
        .boxed()
    }
}

impl IoSource for File {
    fn name(&self) -> String {
        "file".to_string()
    }

    fn coalescing_window(&self) -> Option<u64> {
        Some(4096) // 4 KB
    }

    fn concurrency(&self) -> usize {
        16
    }

    fn size(&self) -> Shared<BoxFuture<'static, SharedVortexResult<u64>>> {
        let file = self
            .try_clone()
            .vortex_expect("Failed to clone file handle");
        async move {
            let metadata = file.metadata().map_err(VortexError::from)?;
            Ok(metadata.len())
        }
        .boxed()
        .shared()
    }

    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let file = self
            .try_clone()
            .vortex_expect("Failed to clone file handle");
        async move {
            let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
            unsafe { buffer.set_len(length) };
            match file.read_exact_at(&mut buffer, offset) {
                Ok(()) => Ok(buffer.freeze()),
                Err(e) => Err(VortexError::from(e)),
            }
        }
        .boxed()
    }
}

#[cfg(feature = "object_store")]
pub struct ObjectStoreIo {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
    concurrency: usize,
    coalesce_window: u64, // In bytes
}

#[cfg(feature = "object_store")]
impl ObjectStoreIo {
    pub fn new(store: Arc<dyn object_store::ObjectStore>, path: object_store::path::Path) -> Self {
        Self {
            store,
            path,
            concurrency: 128,
            coalesce_window: 1 * 1024 * 1024, // 1 MB
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    pub fn with_coalesce_window(mut self, window: u64) -> Self {
        // Currently a no-op, as coalescing is always enabled with a fixed window.
        self.coalesce_window = window;
        self
    }
}

#[cfg(feature = "object_store")]
impl IoSource for ObjectStoreIo {
    fn name(&self) -> String {
        self.path.to_string()
    }

    fn coalescing_window(&self) -> Option<u64> {
        Some(1024 * 1024) // 1 MB
    }

    fn concurrency(&self) -> usize {
        64
    }

    fn size(&self) -> Shared<BoxFuture<'static, SharedVortexResult<u64>>> {
        let store = self.store.clone();
        let path = self.path.clone();
        async move {
            Ok(store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)?)
        }
        .boxed()
        .shared()
    }

    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let store = self.store.clone();
        let path = self.path.clone();

        async move {
            let range = offset..offset + length as u64;
            // FIXME(ngates): see object_store.rs
            let bytes = store.get_range(&path, range).await?;
            let buffer = ByteBuffer::from(bytes).aligned(alignment);
            Ok(buffer)
        }
        .boxed()
    }
}

/// A file that can be read from using a Vortex runtime.
///
/// This essentially provides a wrapper to bind a handle to a read interface. It is optional, but
/// should be used carefully because the subsequent read operations must be driven on the same
/// runtime.
#[derive(Clone)]
pub struct FileIo<'rt> {
    name: String,
    size: Shared<BoxFuture<'static, SharedVortexResult<u64>>>,
    send: flume::Sender<IoRequest>,
    _phantom: PhantomData<&'rt ()>,
}

pub struct IoRequest {
    pub offset: u64,
    pub length: usize,
    pub alignment: Alignment,
    pub callback: ReadCompletion,
}

impl FileIo<'_> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let (read, callback) = Read::future();
        if let Err(e) = self.send.send(IoRequest {
            offset,
            length,
            alignment,
            callback,
        }) {
            vortex_panic!("Failed to send I/O task, runtime terminated: {e}");
        }
        read
    }

    pub fn size(&self) -> impl Future<Output = VortexResult<u64>> {
        self.size.clone().map_err(VortexError::from)
    }
}
