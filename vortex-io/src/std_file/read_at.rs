// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
#[cfg(all(not(unix), not(windows)))]
use std::io::Read;
#[cfg(all(not(unix), not(windows)))]
use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::FuturesUnordered;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::FILE_PREFERRED_READ_SIZE;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::runtime::Handle;

/// Read exactly `buffer.len()` bytes from `file` starting at `offset`.
/// This is a platform-specific helper that uses the most efficient method available.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buffer, offset)
    }
    #[cfg(windows)]
    {
        let mut bytes_read = 0;
        while bytes_read < buffer.len() {
            let read = file.seek_read(&mut buffer[bytes_read..], offset + bytes_read as u64)?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            bytes_read += read;
        }
        Ok(())
    }
    #[cfg(all(not(unix), not(windows)))]
    {
        use std::io::SeekFrom;
        let mut file_ref = file;
        file_ref.seek(SeekFrom::Start(offset))?;
        file_ref.read_exact(buffer)
    }
}

/// Bytes a single blocking task reads from a batch before the batch is split across more tasks.
///
/// Handing a read to the blocking pool costs a thread wake-up, which is far more than a small
/// read from the page cache. Batches of small ranges therefore share one task, while large ranges
/// still spread across threads.
const BLOCKING_TASK_READ_BYTES: usize = 256 << 10;

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    allocator: BufferAllocatorRef,
}

impl FileReadAt {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        Self::open_with_allocator(path, handle, BufferAllocatorRef::statically_allocated())
    }

    /// Open a file for reading using a custom writable buffer allocator.
    pub fn open_with_allocator(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: BufferAllocatorRef,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            allocator,
        })
    }
}

impl VortexReadAt for FileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn preferred_read_size(&self) -> Option<u64> {
        Some(FILE_PREFERRED_READ_SIZE)
    }

    fn concurrency(&self) -> usize {
        DEFAULT_CONCURRENCY
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
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        let allocator = self.allocator.clone();
        async move {
            handle
                .spawn_blocking(move || {
                    let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
                    // SAFETY: read_exact_at initializes every byte before the buffer is frozen.
                    unsafe { buffer.set_len(length) };
                    read_exact_at(&file, buffer.as_mut_slice(), offset)?;
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        let total_bytes: usize = requests.iter().map(|request| request.length).sum();
        let tasks = total_bytes
            .div_ceil(BLOCKING_TASK_READ_BYTES)
            .clamp(1, requests.len().min(DEFAULT_CONCURRENCY));
        let per_task = requests.len().div_ceil(tasks);

        let reads = FuturesUnordered::new();
        for start in (0..requests.len()).step_by(per_task) {
            let end = (start + per_task).min(requests.len());
            let requests = Arc::clone(&requests);
            let file = Arc::clone(&self.file);
            let allocator = self.allocator.clone();
            reads.push(self.handle.spawn_blocking(move || {
                requests[start..end]
                    .iter()
                    .map(|&request| {
                        let mut buffer = allocator
                            .with_capacity_aligned::<u8>(request.length, request.alignment);
                        // SAFETY: read_exact_at initializes every byte before the buffer is frozen.
                        unsafe { buffer.set_len(request.length) };
                        let result = read_exact_at(&file, buffer.as_mut_slice(), request.offset)
                            .map(|()| BufferHandle::new_host(buffer.freeze()))
                            .map_err(Into::into);
                        (request, result)
                    })
                    .collect::<Vec<_>>()
            }));
        }
        reads.flat_map(stream::iter).boxed()
    }
}
