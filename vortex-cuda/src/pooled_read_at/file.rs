// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(target_os = "linux")]
mod direct;

use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use tokio::sync::Semaphore;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;
use vortex::io::std_file::read_exact_at;

#[cfg(target_os = "linux")]
use self::direct::DirectFileReadBackend;
use crate::CudaDeviceBuffer;
use crate::pinned::PinnedByteBufferPool;
use crate::pinned::PooledPinnedBuffer;
use crate::stream::VortexCudaStream;

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_FILE_CONCURRENCY: usize = 32;

// Physical segments can exceed the coalescing limit. Split their I/O, not their encoding,
// to limit blocking read sizes and start transfers before the whole segment is read.
const FILE_READ_CHUNK_BYTES: usize = 4 << 20;

/// Options controlling how [`PooledFileReadAt`] opens and reads a local file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PooledFileReadAtOptions {
    direct_io: bool,
}

impl PooledFileReadAtOptions {
    /// Bypass the operating system page cache for pooled file reads.
    ///
    /// This option is available only on Linux. Unaligned logical reads are widened to satisfy the
    /// filesystem's direct-I/O requirements and sliced back to the requested range after transfer
    /// to the device.
    #[cfg(target_os = "linux")]
    pub fn with_direct_io(mut self) -> Self {
        self.direct_io = true;
        self
    }

    fn open(self, path: &Path) -> VortexResult<Arc<dyn FileReadBackend>> {
        #[cfg(target_os = "linux")]
        if self.direct_io {
            return Ok(Arc::new(DirectFileReadBackend::open(path)?));
        }
        Ok(Arc::new(File::open(path)?))
    }
}

struct PooledHostRead {
    buffer: PooledPinnedBuffer,
    requested_range: Range<usize>,
}

trait FileReadBackend: Send + Sync {
    fn size(&self) -> VortexResult<u64>;

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead>;
}

impl FileReadBackend for File {
    fn size(&self) -> VortexResult<u64> {
        Ok(self.metadata()?.len())
    }

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead> {
        let mut buffer = pool.get(length)?;
        read_exact_at(self, buffer.as_mut_slice(), offset)?;
        Ok(PooledHostRead {
            buffer,
            requested_range: 0..length,
        })
    }
}

/// File reader that uses CUDA pinned host memory for I/O buffers and transfers
/// directly to the GPU.
///
/// Stages reads in pooled pinned buffers for non-blocking H2D transfer. Large reads use the
/// blocking I/O runtime, copying chunks into one device allocation as they finish. Concurrent
/// host reads are bounded per open file.
///
/// This is a data-plane reader. To open a complete local Vortex file, prefer
/// [`crate::CudaOpenOptionsExt::with_cuda`], which keeps the footer and zone maps on the host.
#[derive(Clone)]
pub struct PooledFileReadAt {
    uri: Arc<str>,
    backend: Arc<dyn FileReadBackend>,
    handle: Handle,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
    read_slots: Arc<Semaphore>,
}

impl PooledFileReadAt {
    /// Open a file for pooled reading with direct device transfer.
    pub fn open(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        Self::open_with_options(
            path,
            handle,
            pool,
            stream,
            PooledFileReadAtOptions::default(),
        )
    }

    /// Open a file for pooled reading with explicit options.
    pub fn open_with_options(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
        options: PooledFileReadAtOptions,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = Arc::from(path.to_string_lossy().to_string());
        let backend = options.open(path)?;
        Ok(Self {
            uri,
            backend,
            handle,
            pool,
            stream,
            read_slots: Arc::new(Semaphore::new(DEFAULT_FILE_CONCURRENCY)),
        })
    }

    /// Read into pinned memory under the file's shared concurrency limit.
    /// The returned `requested_range` excludes any backend alignment padding.
    async fn read_host(&self, offset: u64, length: usize) -> VortexResult<PooledHostRead> {
        let backend = Arc::clone(&self.backend);
        let pool = Arc::clone(&self.pool);
        let permit = Arc::clone(&self.read_slots)
            .acquire_owned()
            .await
            .map_err(|error| vortex_err!("file read semaphore closed: {error}"))?;
        self.handle
            .spawn_blocking(move || {
                // A started blocking read cannot be cancelled. Keep its permit and buffer owners
                // in the job even if the scan drops the awaiting future.
                let _permit = permit;
                backend.read(&pool, offset, length)
            })
            .await
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
        let backend = Arc::clone(&self.backend);
        async move { backend.size() }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let reader = self.clone();
        async move {
            vortex_ensure!(
                offset.checked_add(u64::try_from(length)?).is_some(),
                "file read range overflow: offset={offset}, length={length}"
            );
            if length <= FILE_READ_CHUNK_BYTES {
                let read = reader.read_host(offset, length).await?;
                let cuda_buf = read.buffer.transfer_to_device(&reader.stream)?;
                return Ok(BufferHandle::new_device(Arc::new(cuda_buf)).slice(read.requested_range));
            }

            let mut output = reader.stream.device_alloc::<u8>(length)?;
            let mut reads = stream::iter((0..length).step_by(FILE_READ_CHUNK_BYTES).map(|start| {
                let size = (length - start).min(FILE_READ_CHUNK_BYTES);
                reader
                    .read_host(offset + start as u64, size)
                    .map_ok(move |read| (start, read))
            }))
            .buffer_unordered(DEFAULT_FILE_CONCURRENCY);
            while let Some((start, read)) = reads.try_next().await? {
                let end = start + read.requested_range.len();
                read.buffer.copy_to_device(
                    &reader.stream,
                    read.requested_range,
                    &mut output.slice_mut(start..end),
                )?;
            }
            Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(
                output,
            ))))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests;
