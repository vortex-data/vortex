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
use std::sync::LazyLock;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(target_os = "linux")]
use crate::std_file::direct::DirectIoConstraints;
#[cfg(target_os = "linux")]
use crate::std_file::direct::open_direct;

#[cfg(test)]
mod tests;

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

/// Environment variable that opts local file reads into direct I/O.
pub const DIRECT_IO_ENV_VAR: &str = "VORTEX_DIRECT_IO";

static DIRECT_IO_FROM_ENV: LazyLock<bool> =
    LazyLock::new(|| std::env::var(DIRECT_IO_ENV_VAR).is_ok_and(|v| v == "1"));

/// Options controlling how [`FileReadAt`] opens and reads a local file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FileReadAtOptions {
    direct_io: bool,
}

impl FileReadAtOptions {
    /// Options taken from the process environment.
    ///
    /// Direct I/O is enabled when `VORTEX_DIRECT_IO=1`, letting a deployment or benchmark switch
    /// between page-cached and direct reads without recompiling.
    pub fn from_env() -> Self {
        Self {
            direct_io: *DIRECT_IO_FROM_ENV,
        }
    }

    /// Bypass the operating system page cache for file reads.
    ///
    /// This option is available only on Linux. Reads that do not meet the filesystem's direct-I/O
    /// alignment requirements are widened to the enclosing block boundaries and sliced back to the
    /// requested range, so files written by any Vortex version can be read this way. If the file
    /// cannot be opened with `O_DIRECT`, the reader falls back to buffered reads.
    #[cfg(target_os = "linux")]
    pub fn with_direct_io(mut self) -> Self {
        self.direct_io = true;
        self
    }

    /// Whether direct I/O was requested.
    pub fn direct_io(&self) -> bool {
        self.direct_io
    }
}

/// How [`FileReadAt`] transfers bytes out of the file.
enum ReadMode {
    /// Ordinary `pread`, served through the operating system page cache.
    Buffered,
    /// `O_DIRECT` `pread`, bypassing the page cache.
    #[cfg(target_os = "linux")]
    Direct(DirectIoConstraints),
}

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    mode: Arc<ReadMode>,
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
        Self::open_with_options(path, handle, allocator, FileReadAtOptions::from_env())
    }

    /// Open a file for reading with explicit options.
    pub fn open_with_options(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: BufferAllocatorRef,
        options: FileReadAtOptions,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri: Arc<str> = path.to_string_lossy().to_string().into();
        let (file, mode) = open_file(path, options, &uri)?;
        Ok(Self {
            uri,
            file: Arc::new(file),
            mode: Arc::new(mode),
            handle,
            allocator,
        })
    }

    /// Whether this reader bypasses the operating system page cache.
    pub fn is_direct(&self) -> bool {
        match self.mode.as_ref() {
            ReadMode::Buffered => false,
            #[cfg(target_os = "linux")]
            ReadMode::Direct(_) => true,
        }
    }
}

#[cfg(target_os = "linux")]
fn open_file(path: &Path, options: FileReadAtOptions, uri: &str) -> io::Result<(File, ReadMode)> {
    if !options.direct_io {
        return Ok((File::open(path)?, ReadMode::Buffered));
    }
    // Filesystems that cannot serve O_DIRECT reject the open outright, so degrade to buffered
    // reads rather than failing to open a file we can read perfectly well.
    match open_direct(path) {
        Ok(file) => match DirectIoConstraints::probe(&file) {
            Ok(constraints) => Ok((file, ReadMode::Direct(constraints))),
            Err(err) => {
                tracing::warn!(
                    "{uri}: direct I/O constraints unavailable, using buffered reads: {err}"
                );
                Ok((File::open(path)?, ReadMode::Buffered))
            }
        },
        Err(err) => {
            tracing::warn!("{uri}: cannot open with O_DIRECT, using buffered reads: {err}");
            Ok((File::open(path)?, ReadMode::Buffered))
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn open_file(path: &Path, options: FileReadAtOptions, uri: &str) -> io::Result<(File, ReadMode)> {
    if options.direct_io {
        tracing::warn!("{uri}: direct I/O is only supported on Linux, using buffered reads");
    }
    Ok((File::open(path)?, ReadMode::Buffered))
}

/// Read `length` bytes at `offset`, returning a buffer aligned to `alignment`.
fn read_buffer(
    file: &File,
    mode: &ReadMode,
    allocator: &BufferAllocatorRef,
    offset: u64,
    length: usize,
    alignment: Alignment,
) -> VortexResult<BufferHandle> {
    match mode {
        ReadMode::Buffered => {
            let mut buffer = allocator.with_capacity_aligned::<u8>(length, alignment);
            // SAFETY: read_exact_at initializes every byte before the buffer is frozen.
            unsafe { buffer.set_len(length) };
            read_exact_at(file, buffer.as_mut_slice(), offset)?;
            Ok(BufferHandle::new_host(buffer.freeze()))
        }
        #[cfg(target_os = "linux")]
        ReadMode::Direct(constraints) => {
            if length == 0 {
                let buffer = allocator.with_capacity_aligned::<u8>(0, alignment);
                return Ok(BufferHandle::new_host(buffer.freeze()));
            }

            let range = constraints.widen(offset, length)?;
            // The pointer must satisfy the filesystem's requirement as well as the caller's, and
            // over-aligning the base keeps the requested bytes aligned once sliced back out: a
            // segment offset that is a multiple of `alignment` stays one relative to a block
            // boundary, because both are powers of two and blocks are the larger of the pair.
            let alloc_alignment = alignment.max(Alignment::new(constraints.memory_alignment()));
            let mut buffer: ByteBufferMut = ByteBufferMut::with_capacity_aligned_in(
                range.read_length,
                alloc_alignment,
                allocator.clone(),
            );
            // SAFETY: the length is trimmed below to the prefix the kernel initialized.
            unsafe { buffer.set_len(range.read_length) };
            let initialized = constraints.read_at(
                file,
                buffer.as_mut_slice(),
                range.read_offset,
                range.requested_range.end,
            )?;
            unsafe { buffer.set_len(initialized) };

            Ok(BufferHandle::new_host(
                buffer
                    .freeze()
                    .slice_unaligned(range.requested_range)
                    .aligned(alignment),
            ))
        }
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
        let mode = Arc::clone(&self.mode);
        let handle = self.handle.clone();
        let allocator = self.allocator.clone();
        async move {
            handle
                .spawn_blocking(move || {
                    read_buffer(&file, &mode, &allocator, offset, length, alignment)
                })
                .await
        }
        .boxed()
    }
}
