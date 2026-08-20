// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(target_os = "linux")]
mod direct;

use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;
use vortex::io::std_file::read_exact_at;

#[cfg(target_os = "linux")]
use self::direct::DirectFileReadBackend;
use crate::pinned::PinnedByteBufferPool;
use crate::pinned::PooledPinnedBuffer;
use crate::stream::VortexCudaStream;

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_FILE_CONCURRENCY: usize = 32;

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

struct BufferedFileReadBackend {
    file: File,
}

impl BufferedFileReadBackend {
    fn open(path: &Path) -> VortexResult<Self> {
        Ok(Self {
            file: File::open(path)?,
        })
    }
}

impl FileReadBackend for BufferedFileReadBackend {
    fn size(&self) -> VortexResult<u64> {
        Ok(self.file.metadata()?.len())
    }

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead> {
        let mut buffer = pool.get(length)?;
        read_exact_at(&self.file, buffer.as_mut_slice(), offset)?;
        Ok(PooledHostRead {
            buffer,
            requested_range: 0..length,
        })
    }
}

#[cfg(target_os = "linux")]
fn open_backend(
    path: &Path,
    options: PooledFileReadAtOptions,
) -> VortexResult<Arc<dyn FileReadBackend>> {
    if options.direct_io {
        Ok(Arc::new(DirectFileReadBackend::open(path)?))
    } else {
        Ok(Arc::new(BufferedFileReadBackend::open(path)?))
    }
}

#[cfg(not(target_os = "linux"))]
fn open_backend(
    path: &Path,
    _options: PooledFileReadAtOptions,
) -> VortexResult<Arc<dyn FileReadBackend>> {
    Ok(Arc::new(BufferedFileReadBackend::open(path)?))
}

/// File reader that uses CUDA pinned host memory for I/O buffers and transfers
/// directly to the GPU.
///
/// Reads into a pooled pinned (page-locked) buffer, then submits a non-blocking
/// H2D DMA transfer and returns a device `BufferHandle`.
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
        let backend = open_backend(path, options)?;
        Ok(Self {
            uri,
            backend,
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
        let backend = Arc::clone(&self.backend);
        async move { backend.size() }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let backend = Arc::clone(&self.backend);
        let handle = self.handle.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);

        async move {
            let read = handle
                .spawn_blocking(move || backend.read(&pool, offset, length))
                .await?;
            let cuda_buf = read.buffer.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)).slice(read.requested_range))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooled_file_read_options_default_to_buffered_io() {
        assert!(!PooledFileReadAtOptions::default().direct_io);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pooled_file_read_options_enable_direct_io() {
        assert!(
            PooledFileReadAtOptions::default()
                .with_direct_io()
                .direct_io
        );
    }
}
