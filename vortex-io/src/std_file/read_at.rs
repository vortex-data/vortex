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
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
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
        let uri = tracing::enabled!(target: "vortex_io::read_timing", tracing::Level::DEBUG)
            .then(|| Arc::clone(&self.uri));
        async move {
            let submitted = uri.as_ref().map(|_| Instant::now());
            handle
                .spawn_blocking(move || {
                    let started = submitted.map(|_| Instant::now());
                    let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
                    // SAFETY: read_exact_at initializes every byte before the buffer is frozen.
                    unsafe { buffer.set_len(length) };
                    let read_started = started.map(|_| Instant::now());
                    let result = read_exact_at(&file, buffer.as_mut_slice(), offset);
                    if let Some(((submitted, started), (read_started, uri))) =
                        submitted.zip(started).zip(read_started.zip(uri))
                    {
                        let finished = Instant::now();
                        tracing::debug!(
                            target: "vortex_io::read_timing",
                            path = %uri,
                            offset,
                            length,
                            completed_unix_ns = u64::try_from(
                                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
                            ).unwrap_or(u64::MAX),
                            queue_ns = u64::try_from(started.duration_since(submitted).as_nanos()).unwrap_or(u64::MAX),
                            allocation_ns = u64::try_from(read_started.duration_since(started).as_nanos()).unwrap_or(u64::MAX),
                            read_ns = u64::try_from(finished.duration_since(read_started).as_nanos()).unwrap_or(u64::MAX),
                            "local positional read"
                        );
                    }
                    result?;
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }
}
