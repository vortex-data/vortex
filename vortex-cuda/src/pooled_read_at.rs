// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;
use vortex::io::std_file::read_exact_at;

use crate::pinned::PinnedByteBufferPool;
use crate::stream::VortexCudaStream;

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_FILE_CONCURRENCY: usize = 32;
/// Default number of concurrent requests to allow for object store I/O.
pub const DEFAULT_OBJECT_STORE_CONCURRENCY: usize = 192;

/// File reader that uses CUDA pinned host memory for I/O buffers and transfers
/// directly to the GPU.
///
/// Reads into a pooled pinned (page-locked) buffer, then submits a non-blocking
/// H2D DMA transfer and returns a device `BufferHandle`.
#[derive(Clone)]
pub struct PooledFileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
}

impl PooledFileReadAt {
    /// Open a file for pooled reading with direct device transfer.
    pub fn open(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = Arc::from(path.to_string_lossy().to_string());
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            pool,
            stream,
        })
    }
}

impl VortexReadAt for PooledFileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_FILE_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = Arc::clone(&self.file);
        async move {
            let metadata = file.metadata()?;
            Ok(metadata.len())
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);

        async move {
            let mut target = pool.get(length)?;
            let target = handle
                .spawn_blocking(move || {
                    read_exact_at(&file, target.as_mut_slice(), offset)?;
                    Ok::<_, io::Error>(target)
                })
                .await
                .map_err(VortexError::from)?;

            let cuda_buf = target.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }
        .boxed()
    }
}

/// Object store reader that uses CUDA pinned host memory for I/O buffers and
/// transfers directly to the GPU.
///
/// Reads into a pooled pinned (page-locked) buffer, then submits a non-blocking
/// H2D DMA transfer and returns a device `BufferHandle`.
#[derive(Clone)]
pub struct PooledObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

impl PooledObjectStoreReadAt {
    /// Create a new object-store source with pinned host-buffer allocations and direct device transfer.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            pool,
            stream,
            concurrency: DEFAULT_OBJECT_STORE_CONCURRENCY,
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

impl VortexReadAt for PooledObjectStoreReadAt {
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
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);

        async move {
            let end = offset.checked_add(length as u64).ok_or_else(|| {
                vortex_err!(
                    "Object store read range overflow: offset={}, length={}",
                    offset,
                    length
                )
            })?;
            let range = offset..end;
            let mut target = pool.get(length)?;
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
                    target = handle
                        .spawn_blocking(move || {
                            read_exact_at(&file, target.as_mut_slice(), range.start)?;
                            Ok::<_, io::Error>(target)
                        })
                        .await
                        .map_err(VortexError::from)?;
                }
                #[cfg(target_arch = "wasm32")]
                GetResultPayload::File(..) => {
                    unreachable!("File payload not supported on wasm32")
                }
                GetResultPayload::Stream(mut byte_stream) => {
                    let mut filled = 0usize;
                    while let Some(bytes) = byte_stream.next().await {
                        let bytes = bytes?;
                        let end = filled + bytes.len();
                        vortex_ensure!(
                            end <= length,
                            "Object store stream returned more bytes than expected (expected {} bytes, got at least {} bytes, range: {:?})",
                            length,
                            end,
                            range
                        );
                        target.as_mut_slice()[filled..end].copy_from_slice(&bytes);
                        filled = end;
                    }

                    vortex_ensure!(
                        filled == length,
                        "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                        filled,
                        length,
                        range
                    );
                }
            }

            let cuda_buf = target.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }
        .boxed()
    }
}

/// Default number of concurrent requests to allow for in-memory byte buffer I/O.
pub const DEFAULT_BYTE_BUFFER_CONCURRENCY: usize = 16;

/// In-memory byte buffer reader that uses CUDA pinned host memory for staging
/// and transfers directly to the GPU.
///
/// Slices the source `ByteBuffer`, copies into a pooled pinned (page-locked)
/// buffer, then submits a non-blocking H2D DMA transfer and returns a device
/// `BufferHandle`.
#[derive(Clone)]
pub struct PooledByteBufferReadAt {
    buffer: ByteBuffer,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
}

impl PooledByteBufferReadAt {
    /// Create a new in-memory reader with pinned host-buffer allocations and direct device transfer.
    pub fn new(
        buffer: ByteBuffer,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> Self {
        Self {
            buffer,
            pool,
            stream,
        }
    }
}

impl VortexReadAt for PooledByteBufferReadAt {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::in_memory())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_BYTE_BUFFER_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.buffer.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let buffer = self.buffer.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);

        async move {
            let offset = usize::try_from(offset)
                .map_err(|_| vortex_err!("Byte buffer read offset overflow: offset={}", offset))?;
            let src = &buffer.as_ref()[offset..offset + length];

            let mut target = pool.get(length)?;
            target.as_mut_slice().copy_from_slice(src);

            let cuda_buf = target.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }
        .boxed()
    }
}
