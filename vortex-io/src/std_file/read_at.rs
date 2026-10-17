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
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
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

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    allocator: BufferAllocatorRef,
    /// A memory map of the file, which reads slice when present.
    mmap: Option<ByteBuffer>,
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
            mmap: None,
        })
    }

    /// Open a file for reading through a memory map of it.
    ///
    /// Reads return slices of the map rather than copies, which saves copying every read out of
    /// the page cache, allocating its buffer and waiting for a blocking thread to do so. Segments
    /// that the file does not align as requested are still copied.
    ///
    /// # Safety
    ///
    /// The file must not be truncated or modified while any buffer read from it is alive. A
    /// truncated file makes accesses to the unmapped pages fail with `SIGBUS`, and a modified
    /// file changes the contents of buffers that were already read.
    pub unsafe fn open_mmap(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        let mut reader = Self::open(path, handle)?;
        // SAFETY: the caller guarantees that the file does not change while it is mapped.
        let mmap = unsafe { memmap2::Mmap::map(reader.file.as_ref())? };
        reader.mmap = Some(ByteBuffer::from(mmap));
        Ok(reader)
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
        if let Some(mmap) = &self.mmap {
            let start = usize::try_from(offset).vortex_expect("offset fits in usize");
            let slice = mmap
                .slice_unaligned(start..start + length)
                .aligned(alignment);
            return futures::future::ready(Ok(BufferHandle::new_host(slice))).boxed();
        }
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        let allocator = read_allocator(self.allocator.clone());
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tempfile::tempdir;
    use vortex_error::VortexResult;

    use super::*;
    use crate::runtime::BlockingRuntime;
    use crate::runtime::current::CurrentThreadRuntime;

    #[rstest]
    #[case(0, 16, 1)]
    #[case(3, 100, 1)]
    #[case(8, 64, 8)]
    // An offset that the map does not align as requested is copied into an aligned buffer.
    #[case(3, 64, 8)]
    fn mmap_reads_match_file_reads(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] alignment: usize,
    ) -> VortexResult<()> {
        let directory = tempdir()?;
        let path = directory.path().join("read.vortex");
        let contents: Vec<u8> = (0..=255).collect();
        std::fs::write(&path, &contents)?;

        let runtime = CurrentThreadRuntime::new();
        let alignment = Alignment::new(alignment);
        let read = |reader: FileReadAt| {
            runtime.block_on(async {
                let buffer = reader.read_at(offset, length, alignment).await?;
                VortexResult::Ok(buffer.into_host_sync())
            })
        };

        let copied = read(FileReadAt::open(&path, runtime.handle())?)?;
        // SAFETY: the file is not modified while it is mapped.
        let mapped = read(unsafe { FileReadAt::open_mmap(&path, runtime.handle()) }?)?;

        let start = usize::try_from(offset).vortex_expect("offset fits in usize");
        assert_eq!(mapped.as_slice(), &contents[start..start + length]);
        assert_eq!(mapped.as_slice(), copied.as_slice());
        assert!(mapped.is_aligned(alignment));
        Ok(())
    }
}
