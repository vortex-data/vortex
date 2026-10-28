// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use t4::CachedFile;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::object_store::DEFAULT_CONCURRENCY;
use crate::runtime::Handle;

/// A file read through t4, opened once ([`t4::T4Caching::open`]).
///
/// It reads as [`ObjectStoreReadAt`](crate::object_store::ObjectStoreReadAt) does, with the
/// same coalescing and concurrency unless set otherwise, but its bytes come from t4's memory,
/// its local drives, or the object store under it. t4 answers each read with one buffer whose
/// address agrees with the file offset modulo 4 KiB, and Vortex aligns its segments in the
/// file, so that buffer becomes the read's own without a copy.
pub struct T4ReadAt {
    file: Arc<CachedFile>,
    uri: Arc<str>,
    handle: Handle,
    allocator: BufferAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

impl T4ReadAt {
    /// Create a source over `file`, opened through t4.
    pub fn new(file: CachedFile, handle: Handle) -> Self {
        Self::new_with_allocator(file, handle, BufferAllocatorRef::statically_allocated())
    }

    /// Create a source over `file` with a custom writable buffer allocator, into which each read
    /// is copied.
    pub fn new_with_allocator(
        file: CachedFile,
        handle: Handle,
        allocator: BufferAllocatorRef,
    ) -> Self {
        let uri = Arc::from(file.meta().location.to_string());
        Self {
            file: Arc::new(file),
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

    /// Set how nearby reads are merged before they reach t4, or `None` to hand t4 each read as
    /// asked.
    pub fn with_coalesce_config(mut self, coalesce_config: Option<CoalesceConfig>) -> Self {
        self.coalesce_config = coalesce_config;
        self
    }
}

async fn read_t4_range(
    file: Arc<CachedFile>,
    allocator: BufferAllocatorRef,
    request: ReadAtRequest,
) -> VortexResult<BufferHandle> {
    let ReadAtRequest {
        offset,
        length,
        alignment,
    } = request;
    if length == 0 {
        return Ok(BufferHandle::new_host(ByteBuffer::empty_aligned(alignment)));
    }

    let bytes = file.read(offset..offset + length as u64, None).await?;
    vortex_ensure!(
        bytes.len() == length,
        "t4 returned {} bytes but expected {} bytes (offset: {})",
        bytes.len(),
        length,
        offset
    );

    let buffer = if allocator.ptr_eq(BufferAllocatorRef::static_ref())
        && alignment.is_ptr_aligned(bytes.as_ptr())
    {
        ByteBuffer::from_bytes_aligned(bytes, alignment)
    } else {
        let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
        buffer.extend_from_slice(&bytes);
        buffer.freeze()
    };
    Ok(BufferHandle::new_host(buffer))
}

impl VortexReadAt for T4ReadAt {
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
        let size = self.file.meta().size;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: vortex_buffer::Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.handle
            .spawn_io(read_t4_range(
                Arc::clone(&self.file),
                self.allocator.clone(),
                ReadAtRequest::new(offset, length, alignment),
            ))
            .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        if requests.is_empty() {
            return stream::empty().boxed();
        }

        let file = Arc::clone(&self.file);
        let allocator = self.allocator.clone();
        let concurrency = self.concurrency.max(1);
        let (mut send, recv) = mpsc::channel(concurrency);

        // As in `ObjectStoreReadAt`, a single runtime task drives every read of the batch, each
        // result sent on as soon as it is ready.
        let task = self.handle.spawn_io(async move {
            let reads = requests.iter().copied().map(|request| {
                let file = Arc::clone(&file);
                let allocator = allocator.clone();
                async move { (request, read_t4_range(file, allocator, request).await) }
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

// `TokioRuntime::current` gives these tests a runtime handle only under the `tokio` feature.
#[cfg(test)]
#[cfg(feature = "tokio")]
mod tests {
    use object_store::ObjectStore;
    use object_store::ObjectStoreExt;
    use object_store::PutPayload;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use rstest::rstest;
    use t4::T4Caching;
    use tempfile::TempDir;
    use vortex_buffer::Alignment;

    use super::*;
    use crate::runtime::tokio::TokioRuntime;

    const MIB: u64 = 1 << 20;

    /// A file's bytes, each a function of its offset.
    fn file_bytes(len: u64) -> Vec<u8> {
        (0..len).map(|offset| (offset % 251) as u8).collect()
    }

    /// A file of `len` bytes in an in-memory store, opened through t4 caching it in a temporary
    /// directory, which lives as long as the reader.
    async fn t4_read_at(len: u64) -> anyhow::Result<(TempDir, T4ReadAt, Vec<u8>)> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = Path::from("data/test.vortex");
        let data = file_bytes(len);
        store.put(&path, PutPayload::from(data.clone())).await?;
        let cache = T4Caching::builder(store)
            .with_memory_capacity(4 * MIB)
            .with_disk_file(dir.path().join("t4.cache"), 8 * MIB)
            .build()?;
        let file = cache.open(&path).await?;
        Ok((dir, T4ReadAt::new(file, TokioRuntime::current()), data))
    }

    #[rstest]
    #[case(0, 10, 1)]
    #[case(4096 + 7, 300, 1)]
    #[case(3 * 64, 1000, 64)]
    #[case(MIB - 10, 20, 1)]
    #[tokio::test]
    async fn test_read_at_returns_the_files_bytes(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] alignment: usize,
    ) -> anyhow::Result<()> {
        let (_dir, reader, data) = t4_read_at(2 * MIB).await?;
        let alignment = Alignment::new(alignment);

        // The first read fetches from the store under t4, the second is a hit in its memory.
        for _ in 0..2 {
            let buffer = reader.read_at(offset, length, alignment).await?;
            let buffer = buffer.to_host().await;
            let start = usize::try_from(offset)?;
            assert_eq!(buffer.as_slice(), &data[start..start + length]);
            assert!(alignment.is_ptr_aligned(buffer.as_ptr()));
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_read_ranges_answers_each_request() -> anyhow::Result<()> {
        let (_dir, reader, data) = t4_read_at(3 * MIB).await?;
        let requests: Arc<[ReadAtRequest]> = [
            ReadAtRequest::new(2 * MIB + 5, 100, Alignment::new(1)),
            ReadAtRequest::new(0, 64, Alignment::new(64)),
            ReadAtRequest::new(MIB - 3, 6, Alignment::new(1)),
            ReadAtRequest::new(17, 0, Alignment::new(8)),
        ]
        .into();

        let results: Vec<_> = reader.read_ranges(Arc::clone(&requests)).collect().await;
        assert_eq!(results.len(), requests.len());
        for (request, result) in results {
            let buffer = result?.to_host().await;
            let start = usize::try_from(request.offset)?;
            assert_eq!(buffer.as_slice(), &data[start..start + request.length]);
            assert!(request.alignment.is_ptr_aligned(buffer.as_ptr()));
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_size_is_the_files() -> anyhow::Result<()> {
        let (_dir, reader, _data) = t4_read_at(MIB + 3).await?;
        assert_eq!(reader.size().await?, MIB + 3);
        Ok(())
    }
}
