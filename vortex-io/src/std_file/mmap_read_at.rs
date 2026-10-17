// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::std_file::DEFAULT_CONCURRENCY;

/// A [`VortexReadAt`] over a memory map of a local file.
///
/// Reads return slices of the map rather than copies, which saves copying every read out of the
/// page cache, allocating its buffer and waiting for a blocking thread to do so. Segments that the
/// file does not align as requested are still copied.
pub struct MmapReadAt {
    uri: Arc<str>,
    mmap: ByteBuffer,
}

impl MmapReadAt {
    /// Maps the file at `path` for reading.
    ///
    /// # Safety
    ///
    /// The file must not be truncated or modified while any buffer read from it is alive. A
    /// truncated file makes accesses to the unmapped pages fail with `SIGBUS`, and a modified
    /// file changes the contents of buffers that were already read.
    pub unsafe fn open(path: impl AsRef<Path>) -> VortexResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: the caller guarantees that the file does not change while it is mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self {
            uri: path.to_string_lossy().to_string().into(),
            mmap: ByteBuffer::from(mmap),
        })
    }
}

impl VortexReadAt for MmapReadAt {
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
        futures::future::ready(Ok(self.mmap.len() as u64)).boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let range = usize::try_from(offset)
            .ok()
            .and_then(|start| Some(start..start.checked_add(length)?))
            .filter(|range| range.end <= self.mmap.len());
        let result = match range {
            Some(range) => Ok(BufferHandle::new_host(
                self.mmap.slice_unaligned(range).aligned(alignment),
            )),
            None => Err(vortex_err!(
                "read of {length} bytes at {offset} is out of bounds of {} ({} bytes)",
                self.uri,
                self.mmap.len()
            )),
        };
        futures::future::ready(result).boxed()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tempfile::tempdir;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::*;
    use crate::runtime::BlockingRuntime;
    use crate::runtime::current::CurrentThreadRuntime;
    use crate::std_file::FileReadAt;

    #[rstest]
    #[case(0, 16, 1)]
    #[case(3, 100, 1)]
    #[case(8, 64, 8)]
    // An offset that the map does not align as requested is copied into an aligned buffer.
    #[case(3, 64, 8)]
    #[case(200, 56, 1)]
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
        let read = |reader: &dyn VortexReadAt| {
            runtime.block_on(async {
                let buffer = reader.read_at(offset, length, alignment).await?;
                VortexResult::Ok(buffer.into_host_sync())
            })
        };

        let copied = read(&FileReadAt::open(&path, runtime.handle())?)?;
        // SAFETY: the file is not modified while it is mapped.
        let mapped_reader = unsafe { MmapReadAt::open(&path) }?;
        let mapped = read(&mapped_reader)?;

        let start = usize::try_from(offset).vortex_expect("offset fits in usize");
        assert_eq!(mapped.as_slice(), &contents[start..start + length]);
        assert_eq!(mapped.as_slice(), copied.as_slice());
        assert!(mapped.is_aligned(alignment));
        assert_eq!(runtime.block_on(mapped_reader.size())?, 256);
        Ok(())
    }

    #[test]
    fn mmap_read_out_of_bounds_fails() -> VortexResult<()> {
        let directory = tempdir()?;
        let path = directory.path().join("read.vortex");
        std::fs::write(&path, [0u8; 16])?;

        let runtime = CurrentThreadRuntime::new();
        // SAFETY: the file is not modified while it is mapped.
        let reader = unsafe { MmapReadAt::open(&path) }?;
        let result = runtime.block_on(reader.read_at(8, 16, Alignment::none()));
        assert!(result.is_err());
        Ok(())
    }
}
