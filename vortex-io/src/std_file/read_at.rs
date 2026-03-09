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
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
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
    direct_io: AtomicBool,
}

impl FileReadAt {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            direct_io: AtomicBool::new(false),
        })
    }

    /// Open a file with `O_DIRECT` for direct I/O (bypasses the page cache).
    ///
    /// Requires Linux. The implementation handles alignment requirements
    /// internally — offsets and lengths are rounded to 4096-byte boundaries,
    /// and the result is sliced to the originally requested range.
    #[cfg(target_os = "linux")]
    pub fn open_direct(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = Arc::new(
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)?,
        );
        Ok(Self {
            uri,
            file,
            handle,
            direct_io: AtomicBool::new(true),
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
        let file = self.file.clone();
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
        let file = self.file.clone();
        let handle = self.handle.clone();
        let direct_io = self.direct_io.load(Ordering::Relaxed);
        async move {
            handle
                .spawn_blocking(move || {
                    if direct_io {
                        read_at_direct(&file, offset, length, alignment)
                    } else {
                        let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
                        unsafe { buffer.set_len(length) };
                        read_exact_at(&file, &mut buffer, offset)?;
                        Ok(BufferHandle::new_host(buffer.freeze()))
                    }
                })
                .await
        }
        .boxed()
    }
}

/// Perform a read with O_DIRECT alignment handling.
///
/// O_DIRECT requires offset, length, and buffer address to be aligned to the
/// filesystem block size (typically 4096 bytes). We round down the offset and
/// round up the length, read into an aligned buffer, then copy out the
/// originally requested range.
#[cfg(target_os = "linux")]
#[allow(clippy::cast_possible_truncation)]
fn read_at_direct(
    file: &File,
    offset: u64,
    length: usize,
    alignment: Alignment,
) -> VortexResult<BufferHandle> {
    const BLOCK_SIZE: u64 = 4096;

    let aligned_offset = offset & !(BLOCK_SIZE - 1);
    let offset_adjustment = (offset - aligned_offset) as usize;
    let total_needed = offset_adjustment + length;
    let aligned_length = (total_needed + (BLOCK_SIZE as usize - 1)) & !(BLOCK_SIZE as usize - 1);

    // Allocate a page-aligned buffer for O_DIRECT.
    let mut aligned_buf = ByteBufferMut::with_capacity_aligned(
        aligned_length,
        Alignment::new(12), // 4096 = 2^12
    );
    unsafe { aligned_buf.set_len(aligned_length) };
    read_exact_at(file, &mut aligned_buf, aligned_offset)?;

    // Copy the requested slice into a buffer with the caller's requested alignment.
    let mut result = ByteBufferMut::with_capacity_aligned(length, alignment);
    unsafe { result.set_len(length) };
    result
        .as_mut_slice()
        .copy_from_slice(&aligned_buf.as_slice()[offset_adjustment..offset_adjustment + length]);
    Ok(BufferHandle::new_host(result.freeze()))
}
