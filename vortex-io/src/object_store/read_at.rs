// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::io;
use std::ops::Range;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::LazyLock;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Semaphore;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
#[cfg(not(target_arch = "wasm32"))]
use crate::request::trace::timestamp_ns;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

// Local object-store payloads use blocking pread rather than network GETs. An optional aggregate
// limit across files bounds allocation and blocking work; network payloads retain the GET limit.
#[cfg(not(target_arch = "wasm32"))]
fn local_read_limit() -> Option<Arc<Semaphore>> {
    static LIMIT: LazyLock<Option<Arc<Semaphore>>> = LazyLock::new(|| {
        let concurrency = std::env::var("VORTEX_LOCAL_READ_CONCURRENCY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        (concurrency > 0).then(|| Arc::new(Semaphore::new(concurrency)))
    });
    LIMIT.as_ref().map(Arc::clone)
}

/// An object store backed I/O source.
///
/// A positive `VORTEX_LOCAL_READ_CONCURRENCY` enables a process-wide blocking-read limit for local
/// file payloads. The default, `0`, leaves local admission unlimited. Network payloads use this
/// reader's object-store concurrency.
///
/// `VORTEX_LOCAL_FILE_HANDLE_REUSE=1` experimentally retains the first local file payload's
/// descriptor for this reader. Reads and size queries then refer to that opened inode, including
/// after the path is replaced. This removes repeated local GET/open/metadata preparation.
/// On Linux, `VORTEX_LOCAL_READ_RANDOM=1` disables kernel readahead on local file payloads;
/// with handle reuse this hint is applied once when the descriptor is first opened.
pub struct ObjectStoreReadAt {
    #[cfg(not(target_arch = "wasm32"))]
    local_limit: Option<Arc<Semaphore>>,
    #[cfg(not(target_arch = "wasm32"))]
    local_file: Option<Arc<OnceCell<Option<Arc<File>>>>>,
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: BufferAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
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
        #[cfg(not(target_arch = "wasm32"))]
        static REUSE_LOCAL_FILE: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("VORTEX_LOCAL_FILE_HANDLE_REUSE").is_ok_and(|value| value == "1")
        });
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            local_limit: local_read_limit(),
            #[cfg(not(target_arch = "wasm32"))]
            local_file: REUSE_LOCAL_FILE.then(Arc::default),
            store,
            path,
            uri,
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
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

#[cfg(not(target_arch = "wasm32"))]
enum ReadFile {
    Owned(File),
    Shared(Arc<File>),
}

#[cfg(not(target_arch = "wasm32"))]
impl AsRef<File> for ReadFile {
    fn as_ref(&self) -> &File {
        match self {
            Self::Owned(file) => file,
            Self::Shared(file) => file,
        }
    }
}

enum ReadPayload {
    #[cfg(not(target_arch = "wasm32"))]
    File(ReadFile),
    Stream(BoxStream<'static, object_store::Result<bytes::Bytes>>),
}

#[cfg(not(target_arch = "wasm32"))]
fn configure_local_file(file: File) -> File {
    #[cfg(target_os = "linux")]
    {
        static RANDOM_ACCESS: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("VORTEX_LOCAL_READ_RANDOM").is_ok_and(|value| value == "1")
        });
        if *RANDOM_ACCESS {
            // Coalescing already selects the ranges. Shared descriptors otherwise accumulate
            // readahead history across concurrently read, unrelated column segments.
            if let Err(error) = rustix::fs::fadvise(&file, 0, None, rustix::fs::Advice::Random) {
                tracing::debug!(%error, "local random-access hint unavailable");
            }
        }
    }
    file
}

async fn read_payload(
    store: &dyn ObjectStore,
    path: &ObjectPath,
    range: Range<u64>,
    #[cfg(not(target_arch = "wasm32"))] local_file: Option<&OnceCell<Option<Arc<File>>>>,
) -> VortexResult<(ReadPayload, bool)> {
    #[cfg(not(target_arch = "wasm32"))]
    let mut initial_payload = None;
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(local_file) = local_file {
        let mut opened_here = false;
        let file = local_file
            .get_or_try_init(|| async {
                opened_here = true;
                let response = store
                    .get_opts(
                        path,
                        GetOptions {
                            range: Some(GetRange::Bounded(range.clone())),
                            ..Default::default()
                        },
                    )
                    .await?;
                match response.payload {
                    GetResultPayload::File(file, _) => {
                        Ok::<_, VortexError>(Some(Arc::new(configure_local_file(file))))
                    }
                    stream => {
                        // The initializer keeps its stream; other requests do their own GETs.
                        initial_payload = Some(stream);
                        Ok(None)
                    }
                }
            })
            .await?;
        if let Some(file) = file {
            return Ok((
                ReadPayload::File(ReadFile::Shared(Arc::clone(file))),
                !opened_here,
            ));
        }
    }
    #[cfg(target_arch = "wasm32")]
    let initial_payload = None;
    let payload = match initial_payload {
        Some(payload) => payload,
        None => {
            store
                .get_opts(
                    path,
                    GetOptions {
                        range: Some(GetRange::Bounded(range)),
                        ..Default::default()
                    },
                )
                .await?
                .payload
        }
    };
    let payload = match payload {
        #[cfg(not(target_arch = "wasm32"))]
        GetResultPayload::File(file, _) => {
            ReadPayload::File(ReadFile::Owned(configure_local_file(file)))
        }
        #[cfg(target_arch = "wasm32")]
        GetResultPayload::File(..) => unreachable!("File payload not supported on wasm32"),
        GetResultPayload::Stream(stream) => ReadPayload::Stream(stream),
    };
    Ok((payload, false))
}

async fn read_object_store_range(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    allocator: BufferAllocatorRef,
    request: ReadAtRequest,
    #[cfg(not(target_arch = "wasm32"))] local_limit: Option<Arc<Semaphore>>,
    #[cfg(not(target_arch = "wasm32"))] local_file: Option<Arc<OnceCell<Option<Arc<File>>>>>,
) -> VortexResult<BufferHandle> {
    #[cfg(not(target_arch = "wasm32"))]
    let timing = tracing::enabled!(target: "vortex_io::read_timing", tracing::Level::DEBUG)
        .then(Instant::now);
    #[cfg(not(target_arch = "wasm32"))]
    let start_ns = timing.map(|_| timestamp_ns());
    let ReadAtRequest {
        offset,
        length,
        alignment,
    } = request;
    let range = offset..(offset + length as u64);
    let (payload, file_handle_reused) = read_payload(
        store.as_ref(),
        &path,
        range.clone(),
        #[cfg(not(target_arch = "wasm32"))]
        local_file.as_deref(),
    )
    .await?;
    #[cfg(target_arch = "wasm32")]
    let _ = file_handle_reused;
    #[cfg(not(target_arch = "wasm32"))]
    let received = timing.map(|_| Instant::now());
    #[cfg(not(target_arch = "wasm32"))]
    let local_permit = if matches!(&payload, ReadPayload::File(..)) {
        match local_limit {
            Some(limit) => Some(limit.acquire_owned().await.map_err(io::Error::other)?),
            None => None,
        }
    } else {
        None
    };
    #[cfg(not(target_arch = "wasm32"))]
    let admitted = timing.map(|_| Instant::now());
    let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
    // SAFETY: each return path checks that every byte was initialized.
    unsafe { buffer.set_len(length) };
    #[cfg(not(target_arch = "wasm32"))]
    let allocated = timing.map(|_| Instant::now());

    let buffer = match payload {
        #[cfg(not(target_arch = "wasm32"))]
        ReadPayload::File(file) => {
            let submitted = timing.map(|_| Instant::now());
            let queued_at = timing.map(|_| SystemTime::now());
            let (buffer, phases) = io_handle
                .spawn_blocking(move || {
                    // A cancelled async read must not free the slot while pread is still running.
                    let _permit = local_permit;
                    let started = submitted.map(|_| (Instant::now(), SystemTime::now()));
                    read_exact_at(file.as_ref(), buffer.as_mut_slice(), range.start)?;
                    let finished = started.map(|_| (Instant::now(), SystemTime::now()));
                    Ok::<_, io::Error>((buffer, submitted.zip(started).zip(finished)))
                })
                .await
                .map_err(io::Error::other)?;
            if let Some((
                (
                    (((start, received), admitted), allocated),
                    ((submitted, (started, reading_at)), (finished, completed_at)),
                ),
                queued_at,
            )) = timing
                .zip(received)
                .zip(admitted)
                .zip(allocated)
                .zip(phases)
                .zip(queued_at)
            {
                let resumed = Instant::now();
                tracing::debug!(
                    target: "vortex_io::read_timing",
                    path = %path,
                    ts_ns = timestamp_ns(),
                    start_ns = start_ns.unwrap_or_default(),
                    offset,
                    length,
                    queued_unix_ns = queued_at.duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(),
                    started_unix_ns = reading_at.duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(),
                    reading_unix_ns = reading_at.duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(),
                    completed_unix_ns = completed_at.duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(),
                    file_handle_reused,
                    prepare_ns = u64::try_from(submitted.duration_since(start).as_nanos()).unwrap_or(u64::MAX),
                    allocation_ns = u64::try_from(allocated.duration_since(admitted).as_nanos()).unwrap_or(u64::MAX),
                    get_ns = u64::try_from(received.duration_since(start).as_nanos()).unwrap_or(u64::MAX),
                    admission_ns = u64::try_from(admitted.duration_since(received).as_nanos()).unwrap_or(u64::MAX),
                    queue_ns = u64::try_from(started.duration_since(submitted).as_nanos()).unwrap_or(u64::MAX),
                    read_ns = u64::try_from(finished.duration_since(started).as_nanos()).unwrap_or(u64::MAX),
                    resume_ns = u64::try_from(resumed.duration_since(finished).as_nanos()).unwrap_or(u64::MAX),
                    "local object-store read"
                );
            }
            buffer
        }
        ReadPayload::Stream(mut byte_stream) => {
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

            buffer
        }
    };

    Ok(BufferHandle::new_host(buffer.freeze()))
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(file) = self
            .local_file
            .as_ref()
            .and_then(|cache| cache.get())
            .and_then(Option::as_ref)
        {
            let file = Arc::clone(file);
            return self
                .handle
                .spawn_blocking(move || Ok(file.metadata()?.len()))
                .boxed();
        }
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
        #[cfg(not(target_arch = "wasm32"))]
        let local_limit = self.local_limit.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let local_file = self.local_file.clone();
        handle
            .spawn_io(read_object_store_range(
                store,
                path,
                io_handle,
                allocator,
                ReadAtRequest::new(offset, length, alignment),
                #[cfg(not(target_arch = "wasm32"))]
                local_limit,
                #[cfg(not(target_arch = "wasm32"))]
                local_file,
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
        let local_limit = self.local_limit.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let local_file = self.local_file.clone();

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
                let local_limit = local_limit.clone();
                #[cfg(not(target_arch = "wasm32"))]
                let local_file = local_file.clone();
                async move {
                    let result = read_object_store_range(
                        store,
                        path,
                        io_handle,
                        allocator,
                        request,
                        #[cfg(not(target_arch = "wasm32"))]
                        local_limit,
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

    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::Duration;

    use object_store::PutPayload;
    #[cfg(not(target_arch = "wasm32"))]
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    #[cfg(not(target_arch = "wasm32"))]
    use parking_lot::Mutex;

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

    /// Holds submitted blocking work until the test explicitly executes it. Aborting its caller
    /// does not abort a blocking operation that has already been submitted.
    #[cfg(not(target_arch = "wasm32"))]
    #[derive(Default)]
    struct DeferredExecutor {
        executor: CountingExecutor,
        blocking: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl Executor for DeferredExecutor {
        fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.executor.spawn(fut)
        }

        fn spawn_io(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.executor.spawn_io(fut)
        }

        fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            self.executor.spawn_cpu(task)
        }

        fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            self.blocking.lock().push(task);
            TokioAbortHandle::new_handle(tokio::spawn(async {}).abort_handle())
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl DeferredExecutor {
        async fn take_blocking(&self) -> VortexResult<Box<dyn FnOnce() + Send>> {
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if let Some(task) = self.blocking.lock().pop() {
                        return task;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .map_err(|err| vortex_error::vortex_err!("{err}"))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn local_reads_keep_admission_until_cancelled_blocking_work_finishes() -> VortexResult<()>
    {
        let directory = tempfile::tempdir()?;
        let store =
            Arc::new(LocalFileSystem::new_with_prefix(directory.path())?) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;
        let executor = Arc::new(DeferredExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));
        let limit = Arc::new(Semaphore::new(1));
        let mut reader = ObjectStoreReadAt::new(store, path, handle);
        reader.local_limit = Some(Arc::clone(&limit));

        let first = reader.read_at(0, 6, Alignment::none());
        let blocking = executor.take_blocking().await?;
        drop(first);
        assert_eq!(limit.available_permits(), 0);
        let second = reader.read_at(7, 5, Alignment::none());
        assert!(executor.blocking.lock().is_empty());
        blocking();
        executor.take_blocking().await?();
        assert_eq!(second.await?.to_host().await.as_slice(), b"store");
        assert_eq!(limit.available_permits(), 1);
        Ok(())
    }

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

    #[cfg(not(target_arch = "wasm32"))]
    #[rstest::rstest]
    #[case(false)]
    #[case(true)]
    #[tokio::test]
    async fn local_handle_reuse_pins_the_opened_file(#[case] reuse: bool) -> VortexResult<()> {
        let directory = tempfile::tempdir()?;
        let store =
            Arc::new(LocalFileSystem::new_with_prefix(directory.path())?) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;
        let executor = Arc::new(CountingExecutor::default()) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&executor));
        let mut reader = ObjectStoreReadAt::new(Arc::clone(&store), path.clone(), handle);
        reader.local_file = reuse.then(Arc::default);
        assert!(reader.read_at(1024, 1, Alignment::none()).await.is_err());
        if let Some(cache) = &reader.local_file {
            assert!(cache.get().is_none());
        }
        let initial = reader
            .read_ranges(Arc::from([
                ReadAtRequest::new(1024, 1, Alignment::none()),
                ReadAtRequest::new(0, 6, Alignment::new(64)),
                ReadAtRequest::new(7, 5, Alignment::new(64)),
            ]))
            .collect::<Vec<_>>()
            .await;
        assert_eq!(initial.len(), 3);
        for (request, result) in initial {
            if request.offset == 1024 {
                assert!(result.is_err());
                continue;
            }
            let offset = usize::try_from(request.offset)?;
            assert_eq!(
                result?.to_host().await.as_slice(),
                &TEST_DATA[offset..offset + request.length]
            );
        }

        const REPLACEMENT: &[u8] = b"new file";
        // Object-store replacement uses another inode. Reuse deliberately keeps the opened one.
        store
            .put(&path, PutPayload::from_static(REPLACEMENT))
            .await?;
        let expected = if reuse { TEST_DATA } else { REPLACEMENT };
        assert_eq!(reader.size().await?, expected.len() as u64);
        let requests = Arc::from([
            ReadAtRequest::new(0, 4, Alignment::new(64)),
            ReadAtRequest::new(0, 64, Alignment::none()),
            ReadAtRequest::new(4, 4, Alignment::new(64)),
        ]);
        let results = reader.read_ranges(requests).collect::<Vec<_>>().await;
        assert_eq!(results.len(), 3);
        for (request, result) in results {
            if request.length == 64 {
                assert!(result.is_err());
                continue;
            }
            let buffer = result?.to_host().await;
            assert_eq!(buffer.alignment(), Alignment::new(64));
            let offset = usize::try_from(request.offset)?;
            assert_eq!(
                buffer.as_slice(),
                &expected[offset..offset + request.length]
            );
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn local_handle_reuse_keeps_stream_payloads_dynamic() -> VortexResult<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;
        let executor = Arc::new(CountingExecutor::default()) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&executor));
        let mut reader = ObjectStoreReadAt::new(Arc::clone(&store), path.clone(), handle);
        let local_file = Arc::new(OnceCell::new());
        reader.local_file = Some(Arc::clone(&local_file));
        reader.read_at(7, 5, Alignment::none()).await?;
        store
            .put(&path, PutPayload::from_static(b"new file"))
            .await?;
        assert_eq!(
            reader
                .read_at(0, 8, Alignment::none())
                .await?
                .to_host()
                .await
                .as_slice(),
            b"new file"
        );
        assert_eq!(reader.size().await?, 8);
        assert!(local_file.get().is_some_and(Option::is_none));
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
        for (request, result) in results {
            let buffer = result?;
            let offset = usize::try_from(request.offset)?;
            assert_eq!(buffer.len(), request.length);
            assert_eq!(
                buffer.to_host().await.as_slice(),
                &TEST_DATA[offset..offset + request.length]
            );
        }
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }
}
