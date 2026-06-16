// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use memmap2::Mmap;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::read_at::DEFAULT_CONCURRENCY;
use crate::CoalesceConfig;
use crate::VortexReadAt;

pub struct MmapReadAt {
    uri: Arc<str>,
    buffer: ByteBuffer,
}

impl MmapReadAt {
    /// Memory-map a file for reading.
    pub fn open(path: impl AsRef<Path>) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = File::open(path)?;
        // SAFETY: the file is opened read-only and is assumed not to be modified or truncated for
        // the lifetime of this mapping (the standard contract for read-only mmap of data files).
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Random)?;
        Ok(Self {
            uri,
            buffer: ByteBuffer::from(mmap),
        })
    }
}

impl VortexReadAt for MmapReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        None
    }

    fn concurrency(&self) -> usize {
        DEFAULT_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.buffer.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let buffer = self.buffer.clone();
        async move {
            let start = usize::try_from(offset).vortex_expect("offset too big for usize");
            let end =
                usize::try_from(offset + length as u64).vortex_expect("end too big for usize");
            if end > buffer.len() {
                vortex_bail!(
                    "Requested range {}..{} out of bounds for file of length {}",
                    start,
                    end,
                    buffer.len()
                );
            }
            Ok(BufferHandle::new_host(
                buffer.slice_unaligned(start..end).aligned(alignment),
            ))
        }
        .boxed()
    }
}
