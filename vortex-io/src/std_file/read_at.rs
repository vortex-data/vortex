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
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::read_buffer_pool::read_allocator;
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

/// The bytes of registered reads of a local file that are read before a scan waits for them.
/// Copying bytes out of the page cache takes long enough that a scan otherwise waits for each.
pub const DEFAULT_READAHEAD: usize = 256 << 20;

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

    fn readahead(&self) -> usize {
        DEFAULT_READAHEAD
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
        let allocator = read_allocator(self.allocator.clone());
        async move {
            // Bytes in the page cache are copied without waiting for a blocking thread, which
            // costs more than the copy. The copy runs as a task of its own, so that the threads
            // waiting for reads share the copies of a file, instead of the thread that drives
            // the reads of the file making every copy while the others wait.
            let cached_file = Arc::clone(&file);
            let (mut buffer, cached) = handle
                .spawn_cpu(move || {
                    let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
                    // SAFETY: the bytes are read before the buffer is frozen, by read_cached_at
                    // up to `cached` and by read_exact_at from there on.
                    unsafe { buffer.set_len(length) };
                    let cached = read_cached_at(&cached_file, buffer.as_mut_slice(), offset);
                    (buffer, cached)
                })
                .await;
            if cached == length {
                return Ok(BufferHandle::new_host(buffer.freeze()));
            }
            // Only the bytes that must come from the disk wait for a blocking thread.
            handle
                .spawn_blocking(move || {
                    read_exact_at(
                        &file,
                        &mut buffer.as_mut_slice()[cached..],
                        offset + cached as u64,
                    )?;
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }
}

/// Reads the start of `buffer` from `file` at `offset` as far as the page cache holds it,
/// without waiting for the disk, and returns the number of bytes read.
#[cfg(target_os = "linux")]
fn read_cached_at(file: &File, buffer: &mut [u8], offset: u64) -> usize {
    use std::io::IoSliceMut;

    use rustix::io::Errno;
    use rustix::io::ReadWriteFlags;

    let mut read = 0;
    while read < buffer.len() {
        let mut slices = [IoSliceMut::new(&mut buffer[read..])];
        match rustix::io::preadv2(
            file,
            &mut slices,
            offset + read as u64,
            ReadWriteFlags::NOWAIT,
        ) {
            // The end of the file is left for read_exact_at to report.
            Ok(0) => break,
            Ok(len) => read += len,
            Err(Errno::INTR) => {}
            // `EAGAIN` when the next page is not cached, or `EOPNOTSUPP` when the file system
            // cannot read without waiting.
            Err(_) => break,
        }
    }
    read
}

/// Platforms without non-blocking reads of cached bytes read everything on a blocking thread.
#[cfg(not(target_os = "linux"))]
fn read_cached_at(_file: &File, _buffer: &mut [u8], _offset: u64) -> usize {
    0
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use vortex_error::VortexResult;

    use super::*;
    use crate::runtime::BlockingRuntime;
    use crate::runtime::current::CurrentThreadRuntime;

    #[test]
    fn read_past_the_end_fails() -> VortexResult<()> {
        let directory = tempdir()?;
        let path = directory.path().join("read.vortex");
        std::fs::write(&path, [7u8; 16])?;

        let runtime = CurrentThreadRuntime::new();
        let reader = FileReadAt::open(&path, runtime.handle())?;
        // The cached bytes are read without waiting, and the rest hits the end of the file.
        let result = runtime.block_on(reader.read_at(8, 16, Alignment::none()));
        assert!(result.is_err());
        let buffer = runtime.block_on(reader.read_at(8, 8, Alignment::none()))?;
        assert_eq!(buffer.into_host_sync().as_slice(), &[7u8; 8]);
        Ok(())
    }
}
