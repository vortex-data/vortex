// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, IoTask, Read, ReadCompletion, Runtime};
use async_stream::stream;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, Stream, StreamExt, TryFutureExt};
use smol::lock::Semaphore;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{
    vortex_err, vortex_panic, SharedVortexResult, VortexError, VortexExpect, VortexResult,
};

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
    //
    // FIXME(ngates): this API can create a channel that is used for the entire lifetime of the
    //  file. We can then pass the other end of the channel to the runtime.
    pub fn open_file(&self, read: Arc<File>) -> FileIo {
        self.open(read)
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
        path: object_store::path::Path,
    ) -> FileIo {
        self.open(Arc::new(ObjectStoreIo { store, path }))
    }

    pub fn open<D: IoDriver>(&self, driver: Arc<D>) -> FileIo {
        let (send, recv) = flume::unbounded();

        let size = driver.size().map_err(Arc::new).boxed().shared();

        // We map the recv stream through the driver.
        let io_stream = driver.drive(recv.into_stream());
        self.0.spawn_io(io_stream.boxed());

        FileIo { size, send }
    }
}

pub trait IoDriver {
    fn size(&self) -> impl Future<Output = VortexResult<u64>> + Send + 'static;

    /// Convert the given stream of I/O requests into a stream of opaque tasks that each runtime
    /// can process.
    fn drive(
        self: Arc<Self>,
        stream: impl Stream<Item = IoRequest> + Unpin + Send + 'static,
    ) -> impl Stream<Item = IoTask> + Send + 'static;
}

impl IoDriver for ByteBuffer {
    fn size(&self) -> impl Future<Output = VortexResult<u64>> + Send + 'static {
        let len = self.len() as u64;
        async move { Ok(len) }
    }

    fn drive(
        self: Arc<Self>,
        stream: impl Stream<Item = IoRequest> + Unpin + Send + 'static,
    ) -> impl Stream<Item = IoTask> + Send + 'static {
        stream.map(move |req: IoRequest| {
            let buffer = self.clone();
            IoTask::non_local(move || async move {
                if req.offset + req.length as u64 > buffer.len() as u64 {
                    return req
                        .callback
                        .complete(Err(vortex_err!("Read out of bounds")));
                }
                let mut slice = ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                unsafe { slice.set_len(req.length) };
                slice.as_mut_slice().copy_from_slice(
                    &buffer.as_slice()[req.offset as usize..req.offset as usize + req.length],
                );
                req.callback.complete(Ok(slice.freeze()))
            })
        })
    }
}

impl IoDriver for File {
    fn size(&self) -> impl Future<Output = VortexResult<u64>> + Send + 'static {
        let metadata = match self.metadata() {
            Ok(m) => m,
            Err(e) => return async move { Err(VortexError::from(e)) }.boxed(),
        };
        let len = metadata.len();
        async move { Ok(len) }.boxed()
    }

    fn drive(
        self: Arc<Self>,
        stream: impl Stream<Item = IoRequest> + Unpin + Send + 'static,
    ) -> impl Stream<Item = IoTask> + Send + 'static {
        stream.map(move |req: IoRequest| {
            let file = self.clone();
            IoTask::non_local(move || async move {
                let mut buffer = ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                unsafe { buffer.set_len(req.length) };
                match file.read_exact_at(&mut buffer, req.offset) {
                    Ok(()) => req.callback.complete(Ok(buffer.freeze())),
                    Err(e) => req.callback.complete(Err(VortexError::from(e))),
                }
            })
        })
    }
}

#[cfg(feature = "object_store")]
pub struct ObjectStoreIo {
    pub store: Arc<dyn object_store::ObjectStore>,
    pub path: object_store::path::Path,
}
#[cfg(feature = "object_store")]
impl IoDriver for ObjectStoreIo {
    fn size(&self) -> impl Future<Output = VortexResult<u64>> + Send + 'static {
        let store = self.store.clone();
        let path = self.path.clone();
        async move { Ok(store.head(&path).await?.size) }
    }

    fn drive(
        self: Arc<Self>,
        stream: impl Stream<Item = IoRequest> + Unpin + Send + 'static,
    ) -> impl Stream<Item = IoTask> + Send + 'static {
        stream! {
            let semaphore = Arc::new(Semaphore::new(16));
            let mut stream = stream.ready_chunks(256).fuse();

            // Priority queue - maintains the order in which we should process requests
            let mut priority_queue: VecDeque<usize> = VecDeque::new();

            // Spatial index - allows us to find nearby requests for coalescing
            let mut requests_by_offset: BTreeMap<(u64, usize), IoRequest> = BTreeMap::new();

            // Map request ID to its key in the BTreeMap
            let mut id_to_key: HashMap<usize, (u64, usize)> = HashMap::new();

            let mut next_id = 0usize;

            loop {
                // First, wait for a permit to ensure we don't exceed concurrency. This allows us
                // to defer coalescing for as long as possible.
                let guard = semaphore.acquire_arc().await;

                match stream.next().await {
                    Some(reqs) => {
                        for req in reqs {
                            let req_id = next_id;
                            next_id += 1;

                            let key = (req.offset, req_id);

                            // Add to priority queue (FIFO order)
                            priority_queue.push_back(req_id);

                            // Add to spatial index
                            id_to_key.insert(req_id, key);
                            requests_by_offset.insert(key, req);
                        }
                    }
                    None if priority_queue.is_empty() => {
                        // No more requests and nothing pending
                        break;
                    }
                    None => {
                        // Stream exhausted but we have pending requests
                        // Fall through to process them
                    }
                }

                // Process the next request in priority order
                if let Some(next_id) = priority_queue.pop_front() {
                    // Check if this request still exists (might have been coalesced)
                    if let Some(&key) = id_to_key.get(&next_id) {
                        // Build a coalesced request including this priority request
                        let coalesced = build_coalesced_including(
                            &mut requests_by_offset,
                            &mut id_to_key,
                            &mut priority_queue,
                            key,
                            1024 * 1024  // coalesce distance
                        );

                        let store = self.store.clone();
                        let path = self.path.clone();
                        let range = coalesced.range.clone();

                        // Create the IoTask for this coalesced read
                        let task = async move {
                            println!("ObjectStoreIo: reading range {:?}", coalesced);
                            let result = store
                                .get_range(&path, range.clone())
                                .await
                                .map(|bytes| ByteBuffer::from(bytes).aligned(coalesced.alignment))
                                .map_err(VortexError::from);
                            coalesced.resolve(result);
                            drop(guard); // Release the permit
                        };

                        yield IoTask::non_local(|| task);
                    }

                    // If the request doesn't exist, it was already coalesced - continue
                }
            }
        }
    }
}

/// Build a coalesced request that includes the specified request and any nearby requests
fn build_coalesced_including(
    requests_by_offset: &mut BTreeMap<(u64, usize), IoRequest>,
    id_to_key: &mut HashMap<usize, (u64, usize)>,
    priority_queue: &mut VecDeque<usize>,
    must_include_key: (u64, usize),
    coalesce_distance: u64,
) -> CoalescedRequest {
    let (start_offset, start_id) = must_include_key;

    // Remove the mandatory request
    let first_req = requests_by_offset
        .remove(&must_include_key)
        .expect("must_include_key should exist");
    id_to_key.remove(&start_id);

    let mut requests = vec![first_req];
    let mut current_start = requests[0].offset;
    let mut current_end = requests[0].offset + requests[0].length as u64;
    let alignment = requests[0].alignment.clone();

    // Find the range we should scan for coalescing
    let scan_start = start_offset.saturating_sub(coalesce_distance);
    let scan_end = start_offset + requests[0].length as u64 + coalesce_distance;

    // Collect requests that can be coalesced (both before and after our mandatory request)
    let mut keys_to_remove = Vec::new();

    for (&key, req) in requests_by_offset.range((scan_start, 0)..=(scan_end, usize::MAX)) {
        let (req_offset, req_id) = key;
        let req_end = req_offset + req.length as u64;

        // Check if this request is within coalescing distance of our current range
        if (req_offset <= current_end + coalesce_distance && req_end >= current_start)
            || (req_end + coalesce_distance >= current_start && req_offset <= current_end)
        {
            keys_to_remove.push(key);
            current_start = current_start.min(req_offset);
            current_end = current_end.max(req_end);
        }
    }

    // Remove the coalesced requests
    for key in keys_to_remove {
        let (_, req_id) = key;
        if let Some(req) = requests_by_offset.remove(&key) {
            requests.push(req);
            id_to_key.remove(&req_id);
            // Remove from priority queue (this is O(n) but queue should be small)
            priority_queue.retain(|&id| id != req_id);
        }
    }

    // Sort requests by offset for correct slicing in resolve
    requests.sort_unstable_by_key(|r| r.offset);

    CoalescedRequest {
        range: current_start..current_end,
        alignment,
        requests,
    }
}

struct CoalescedRequest {
    range: Range<u64>,
    alignment: Alignment, // The alignment of the first request in the coalesced range.
    requests: Vec<IoRequest>,
}

impl Debug for CoalescedRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoalescedRequest")
            .field("#", &self.requests.len())
            .field("length", &(self.range.end - self.range.start))
            .field("range", &self.range)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl CoalescedRequest {
    pub(crate) fn resolve(self, result: VortexResult<ByteBuffer>) {
        match result {
            Ok(buffer) => {
                let buffer = buffer.aligned(Alignment::none());
                for req in self.requests.into_iter() {
                    let start = (req.offset - self.range.start) as usize;
                    let end = start + req.length;
                    let slice = buffer.slice(start..end).aligned(req.alignment);
                    req.callback.complete(Ok(slice));
                }
            }
            Err(e) => {
                let e = Arc::new(e);
                for req in self.requests.into_iter() {
                    req.callback.complete(Err(VortexError::from(e.clone())));
                }
            }
        }
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
