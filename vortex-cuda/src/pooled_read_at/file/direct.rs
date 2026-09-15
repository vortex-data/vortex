// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::io::std_file::DirectIoConstraints;
use vortex::io::std_file::open_direct;

use super::FileReadBackend;
use super::PooledHostRead;
use crate::pinned::PinnedByteBufferPool;

pub(super) struct DirectFileReadBackend {
    file: File,
    constraints: DirectIoConstraints,
}

impl DirectFileReadBackend {
    pub(super) fn open(path: &Path) -> VortexResult<Self> {
        let file = open_direct(path)?;
        let constraints = DirectIoConstraints::probe(&file)?;
        Ok(Self { file, constraints })
    }
}

impl FileReadBackend for DirectFileReadBackend {
    fn size(&self) -> VortexResult<u64> {
        Ok(self.file.metadata()?.len())
    }

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead> {
        let direct_range = self.constraints.widen(offset, length)?;
        let mut buffer = pool.get(direct_range.read_length)?;
        let address = buffer.as_mut_slice().as_ptr() as usize;
        vortex_ensure!(
            address.is_multiple_of(self.constraints.memory_alignment()),
            "pinned buffer address {address:#x} is not aligned to {} bytes",
            self.constraints.memory_alignment()
        );

        let bytes_read = self.constraints.read_at(
            &self.file,
            buffer.as_mut_slice(),
            direct_range.read_offset,
            direct_range.requested_range.end,
        )?;
        buffer.truncate(bytes_read);
        Ok(PooledHostRead {
            buffer,
            requested_range: direct_range.requested_range,
        })
    }
}
