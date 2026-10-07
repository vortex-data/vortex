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

/// Read exactly `buffer.len()` bytes from `file` at `offset`, but only if they are already resident
/// in the OS page cache.
///
/// Returns `Ok(true)` when the buffer was filled without blocking on storage, and `Ok(false)` when
/// any part of the range would require device IO. A `false` result may leave `buffer` partially
/// written. On Linux this uses `preadv2(RWF_NOWAIT)`; other platforms always return `Ok(false)`.
#[cfg(not(target_arch = "wasm32"))]
pub fn try_read_exact_at_cached(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        use rustix::io::Errno;
        use rustix::io::ReadWriteFlags;

        let mut filled = 0;
        while filled < buffer.len() {
            let bufs = &mut [io::IoSliceMut::new(&mut buffer[filled..])];
            match rustix::io::preadv2(file, bufs, offset + filled as u64, ReadWriteFlags::NOWAIT) {
                // EOF: let the regular read path report the error.
                Ok(0) => return Ok(false),
                Ok(n) => filled += n,
                Err(Errno::INTR) => {}
                // EAGAIN: data is not cached. EOPNOTSUPP/EINVAL: the kernel or filesystem does not
                // support RWF_NOWAIT for this file.
                Err(Errno::AGAIN | Errno::OPNOTSUPP | Errno::INVAL) => return Ok(false),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(true)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, buffer, offset);
        Ok(false)
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

    /// The underlying file handle.
    pub fn file(&self) -> &Arc<File> {
        &self.file
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
}
