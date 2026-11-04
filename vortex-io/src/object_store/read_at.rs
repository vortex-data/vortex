// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
#[cfg(not(target_arch = "wasm32"))]
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

/// An object store backed I/O source.
///
/// The object must remain unchanged while the source is in use. Local file payloads retain
/// their opened file for subsequent reads, including when the path is renamed or replaced.
pub struct ObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: BufferAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    /// Keep the opened local object for subsequent ranges, as with `FileReadAt`.
    #[cfg(not(target_arch = "wasm32"))]
    local_file: Arc<OnceLock<Arc<File>>>,
}

impl ObjectStoreReadAt {
    /// Create a new object store source.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        Self::new_with_allocator(
            store,
            path,
            handle,
            BufferAllocatorRef::statically_allocated(),
        )
    }

    /// Create a new object store source with a custom writable buffer allocator.
    pub fn new_with_allocator(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        allocator: BufferAllocatorRef,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
            #[cfg(not(target_arch = "wasm32"))]
            local_file: Arc::new(OnceLock::new()),
        }
    }

    /// Set the concurrency for this source.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the coalesce config for this source.
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.coalesce_config = Some(config);
        self
    }
}

async fn read_object_store_range(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    allocator: BufferAllocatorRef,
    request: ReadAtRequest,
    #[cfg(not(target_arch = "wasm32"))] local_file: Arc<OnceLock<Arc<File>>>,
) -> VortexResult<BufferHandle> {
    let ReadAtRequest {
        offset,
        length,
        alignment,
    } = request;
    let range = offset..(offset + length as u64);
    let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
    // SAFETY: each return path checks that every byte was initialized.
    unsafe { buffer.set_len(length) };

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(file) = local_file.get() {
        return read_file_range(Arc::clone(file), io_handle, buffer, offset).await;
    }

    let response = store
        .get_opts(
            &path,
            GetOptions {
                range: Some(GetRange::Bounded(range.clone())),
                ..Default::default()
            },
        )
        .await?;

    match response.payload {
        #[cfg(not(target_arch = "wasm32"))]
        GetResultPayload::File(file, _) => {
            // Resolve the path through the store once, preserving prefixed/custom stores.
            // Concurrent first reads all use the same opened file once it is published.
            let file = local_file.get_or_init(|| Arc::new(file));
            read_file_range(Arc::clone(file), io_handle, buffer, offset).await
        }
        #[cfg(target_arch = "wasm32")]
        GetResultPayload::File(..) => {
            unreachable!("File payload not supported on wasm32")
        }
        GetResultPayload::Stream(mut byte_stream) => {
            let mut written = 0usize;
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                let end = written + bytes.len();
                vortex_ensure!(
                    end <= length,
                    "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                    end,
                    length,
                    range
                );
                buffer.as_mut_slice()[written..end].copy_from_slice(&bytes);
                written = end;
            }

            vortex_ensure_eq!(
                written,
                length,
                "Object store stream returned too few bytes (range: {:?})",
                range
            );

            Ok(BufferHandle::new_host(buffer.freeze()))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_file_range(
    file: Arc<File>,
    handle: Handle,
    mut buffer: BufferMut<u8>,
    offset: u64,
) -> VortexResult<BufferHandle> {
    handle
        .spawn_blocking(move || {
            read_exact_at(&file, buffer.as_mut_slice(), offset)?;
            Ok(BufferHandle::new_host(buffer.freeze()))
        })
        .await
}

impl VortexReadAt for ObjectStoreReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = self.allocator.clone();
        let io_handle = handle.clone();
        handle
            .spawn_io(read_object_store_range(
                store,
                path,
                io_handle,
                allocator,
                ReadAtRequest::new(offset, length, alignment),
                #[cfg(not(target_arch = "wasm32"))]
                Arc::clone(&self.local_file),
            ))
            .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        if requests.is_empty() {
            return stream::empty().boxed();
        }

        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = self.allocator.clone();
        let concurrency = self.concurrency.max(1);
        let (mut send, recv) = mpsc::channel(concurrency);
        let io_handle = handle.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let local_file = Arc::clone(&self.local_file);

        // A single runtime task drives all GETs, avoiding one spawn per range. Do not use
        // ObjectStore::get_ranges here: it returns one Vec after every range completes, whereas
        // VortexReadAt::read_ranges must expose each result as soon as it is ready.
        let task = handle.spawn_io(async move {
            let reads = requests.iter().copied().map(|request| {
                let store = Arc::clone(&store);
                let path = path.clone();
                let io_handle = io_handle.clone();
                let allocator = allocator.clone();
                #[cfg(not(target_arch = "wasm32"))]
                let local_file = Arc::clone(&local_file);
                async move {
                    let result = read_object_store_range(
                        store,
                        path,
                        io_handle,
                        allocator,
                        request,
                        #[cfg(not(target_arch = "wasm32"))]
                        local_file,
                    )
                    .await;
                    (request, result)
                }
            });

            let mut reads = stream::iter(reads).buffer_unordered(concurrency);
            while let Some(result) = reads.next().await {
                if send.send(result).await.is_err() {
                    break;
                }
            }
        });

        async_stream::stream! {
            let mut recv = recv;
            while let Some(result) = recv.next().await {
                yield result;
            }
            task.await;
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {

    #[cfg(unix)]
    use std::fs;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use object_store::PutPayload;
    #[cfg(unix)]
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    #[cfg(unix)]
    use tempfile::tempdir;

    use super::*;
    use crate::runtime::AbortHandle;
    use crate::runtime::AbortHandleRef;
    use crate::runtime::Executor;

    const TEST_DATA: &[u8] = b"object store test data";

    #[derive(Default)]
    struct CountingExecutor {
        spawn_count: AtomicUsize,
        spawn_io_count: AtomicUsize,
    }

    impl Executor for CountingExecutor {
        fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_io(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_io_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::spawn(async move { task() }).abort_handle())
        }

        fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::task::spawn_blocking(task).abort_handle())
        }
    }

    struct TokioAbortHandle(tokio::task::AbortHandle);

    impl TokioAbortHandle {
        fn new_handle(handle: tokio::task::AbortHandle) -> AbortHandleRef {
            Box::new(Self(handle))
        }
    }

    impl AbortHandle for TokioAbortHandle {
        fn abort(self: Box<Self>) {
            self.0.abort();
        }
    }

    #[tokio::test]
    async fn read_at_uses_spawn_io() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let reader = ObjectStoreReadAt::new(store, path, handle);
        let buffer = reader.read_at(7, 5, Alignment::new(1)).await?;

        assert_eq!(buffer.to_host().await.as_slice(), b"store");
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[tokio::test]
    async fn read_ranges_uses_one_io_task() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let reader = ObjectStoreReadAt::new(store, path, handle);
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
            ReadAtRequest::new(18, 4, Alignment::new(1)),
        ]);
        let results = reader.read_ranges(requests).collect::<Vec<_>>().await;

        assert_eq!(results.len(), 3);
        let mut ranges = Vec::new();
        for (request, result) in results {
            ranges.push((request.offset, request.length));
            let buffer = result?;
            let offset = usize::try_from(request.offset)?;
            assert_eq!(buffer.len(), request.length);
            assert_eq!(
                buffer.to_host().await.as_slice(),
                &TEST_DATA[offset..offset + request.length]
            );
        }
        ranges.sort_unstable();
        assert_eq!(ranges, [(0, 6), (7, 5), (18, 4)]);
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_ranges_keep_the_opened_file() -> VortexResult<()> {
        let directory = tempdir()?;
        let file_path = directory.path().join("test.bin");
        fs::write(&file_path, TEST_DATA)?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(directory.path())?);
        let executor = Arc::new(CountingExecutor::default()) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&executor));
        let reader = ObjectStoreReadAt::new(store, ObjectPath::from("test.bin"), handle);

        let buffer = reader.read_at(7, 5, Alignment::new(64)).await?;
        assert_eq!(buffer.to_host().await.as_slice(), b"store");
        fs::rename(&file_path, directory.path().join("old.bin"))?;
        fs::write(&file_path, b"replacement")?;

        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(18, 4, Alignment::new(64)),
            ReadAtRequest::new(0, 6, Alignment::new(64)),
            ReadAtRequest::new(7, 5, Alignment::new(64)),
        ]);
        let results = reader.read_ranges(requests).collect::<Vec<_>>().await;
        assert_eq!(results.len(), 3);
        for (request, result) in results {
            let buffer = result?.to_host().await;
            let offset = usize::try_from(request.offset)?;
            assert_eq!(
                buffer.as_slice(),
                &TEST_DATA[offset..offset + request.length]
            );
            assert_eq!(buffer.as_ptr() as usize % 64, 0);
        }

        // A retained descriptor must still report a short read rather than return uninitialized data.
        assert!(reader.read_at(20, 4, Alignment::new(64)).await.is_err());
        let buffer = reader.read_at(18, 4, Alignment::new(64)).await?;
        assert_eq!(buffer.to_host().await.as_slice(), b"data");
        Ok(())
    }
}
