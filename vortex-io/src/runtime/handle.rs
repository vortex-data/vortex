// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::coalesce::{CoalescedRequest, CoalescedStreamExt};
use crate::runtime::{CpuTask, IoTask, Read, ReadCompletion, Runtime};
use futures::future::{BoxFuture, LocalBoxFuture, Shared};
use futures::{FutureExt, StreamExt, TryFutureExt};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, LazyLock};
use tokio::runtime;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{
    vortex_err, vortex_panic, SharedVortexResult, VortexError, VortexExpect, VortexResult,
};

/// Represents a handle to a Vortex runtime that can be used to enqueue CPU- or I/O-bound tasks.
///
// TODO(ngates): I think the handle should probably have a lifetime tied to the runtime?
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
    //
    // FIXME(ngates): this API can create a channel that is used for the entire lifetime of the
    //  file. We can then pass the other end of the channel to the runtime.
    pub fn open_file(&self, read: Arc<File>) -> FileIo {
        self.open(read)
    }

    pub fn open(&self, driver: Arc<dyn IoDriver>) -> FileIo {
        let (send, recv) = flume::unbounded();

        // Construct the size future in case we need it.
        let size = driver.size();

        let concurrency = driver.concurrency();

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

        FileIo { size, send }
    }
}

pub trait IoDriver: Send + Sync + 'static {
    fn coalescing_window(&self) -> Option<u64>;

    fn concurrency(&self) -> usize;

    /// Returns a shared future that resolves to the byte size of the underlying data source.
    fn size(&self) -> Shared<BoxFuture<'static, SharedVortexResult<u64>>>;

    /// Perform the actual I/O operation.
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

impl IoDriver for ByteBuffer {
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

impl IoDriver for File {
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
static TOKIO: LazyLock<runtime::Runtime> = LazyLock::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("vortex-object-store")
        .build()
        .vortex_expect("Failed to create Tokio runtime")
});

#[cfg(feature = "object_store")]
impl IoDriver for ObjectStoreIo {
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

/// Coalesces IoRequests that are within `coalesce_distance` bytes of each other
fn coalesce_requests(
    mut requests: Vec<IoRequest>,
    coalesce_distance: u64,
) -> Vec<CoalescedRequest> {
    if requests.is_empty() {
        return vec![];
    }

    // Sort requests by their start offset
    requests.sort_unstable_by_key(|req| req.offset);

    let mut coalesced = Vec::new();
    let mut current_requests = Vec::new();
    let mut current_start = requests[0].offset;
    let mut current_end = requests[0].offset + requests[0].length as u64;
    let mut current_alignment = requests[0].alignment.clone();

    let mut requests = requests.into_iter();
    current_requests.push(requests.next().vortex_expect("at least one"));

    for req in requests {
        let req_start = req.offset;
        let req_end = req.offset + req.length as u64;

        // Check if this request should be coalesced with the current group
        if req_start.saturating_sub(current_end) <= coalesce_distance {
            // Expand the current range
            current_end = current_end.max(req_end);
            current_requests.push(req);
        } else {
            // Start a new coalesced group
            coalesced.push(CoalescedRequest {
                range: current_start..current_end,
                alignment: current_alignment,
                requests: current_requests,
            });

            // Initialize the new group
            current_start = req_start;
            current_end = req_end;
            current_alignment = req.alignment.clone();
            current_requests = vec![req];
        }
    }

    // Don't forget the last group
    if !current_requests.is_empty() {
        coalesced.push(CoalescedRequest {
            range: current_start..current_end,
            alignment: current_alignment,
            requests: current_requests,
        });
    }

    coalesced
}
/// A file that can be read from using a Vortex runtime.
///
/// This essentially provides a wrapper to bind a handle to a read interface. It is optional, but
/// should be used carefully because the subsequent read operations must be driven on the same
/// runtime.
#[derive(Clone)]
pub struct FileIo {
    size: Shared<BoxFuture<'static, SharedVortexResult<u64>>>,
    send: flume::Sender<IoRequest>,
}

pub struct IoRequest {
    pub offset: u64,
    pub length: usize,
    pub alignment: Alignment,
    pub callback: ReadCompletion,
}

impl FileIo {
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
