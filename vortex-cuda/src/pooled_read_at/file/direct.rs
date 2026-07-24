// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use rustix::fs::AtFlags;
use rustix::fs::Mode;
use rustix::fs::OFlags;
use rustix::fs::StatxFlags;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use super::FileReadBackend;
use super::PooledHostRead;
use crate::pinned::PinnedByteBufferPool;

/// Conservative direct-I/O alignment used when Linux cannot report the filesystem constraints.
///
/// A page-sized fallback is accepted by common block devices and filesystems. If the actual
/// requirement is stricter, the read fails with the underlying `EINVAL`.
const FALLBACK_DIRECT_IO_ALIGNMENT: usize = 4096;

#[derive(Clone, Copy)]
struct DirectIoConstraints {
    /// Required alignment of the address of the userspace I/O buffer.
    memory_alignment: usize,
    /// Required alignment of both the file offset and the I/O length.
    offset_alignment: usize,
}

impl Default for DirectIoConstraints {
    fn default() -> Self {
        Self {
            memory_alignment: FALLBACK_DIRECT_IO_ALIGNMENT,
            offset_alignment: FALLBACK_DIRECT_IO_ALIGNMENT,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DirectIoRange {
    read_offset: u64,
    read_length: usize,
    requested_range: Range<usize>,
}

pub(super) struct DirectFileReadBackend {
    file: File,
    constraints: DirectIoConstraints,
}

impl DirectFileReadBackend {
    pub(super) fn open(path: &Path) -> VortexResult<Self> {
        let file = File::from(
            rustix::fs::open(
                path,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECT,
                Mode::empty(),
            )
            .map_err(io::Error::from)?,
        );
        let constraints = direct_io_constraints(&file)?;
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
        let direct_range = direct_io_range(offset, length, self.constraints.offset_alignment)?;
        let mut buffer = pool.get(direct_range.read_length)?;
        let address = buffer.as_mut_slice().as_ptr() as usize;
        vortex_ensure!(
            address.is_multiple_of(self.constraints.memory_alignment),
            "pinned buffer address {address:#x} is not aligned to {} bytes",
            self.constraints.memory_alignment
        );

        let bytes_read = read_direct_at(
            &self.file,
            buffer.as_mut_slice(),
            direct_range.read_offset,
            direct_range.requested_range.end,
            self.constraints,
        )?;
        buffer.truncate(bytes_read);
        Ok(PooledHostRead {
            buffer,
            requested_range: direct_range.requested_range,
        })
    }
}

fn direct_io_range(offset: u64, length: usize, alignment: usize) -> VortexResult<DirectIoRange> {
    vortex_ensure!(alignment > 0, "direct I/O alignment must be non-zero");
    if length == 0 {
        return Ok(DirectIoRange {
            read_offset: offset,
            read_length: 0,
            requested_range: 0..0,
        });
    }

    let alignment_u64 = u64::try_from(alignment)?;
    let length_u64 = u64::try_from(length)?;
    let requested_end = offset.checked_add(length_u64).ok_or_else(|| {
        vortex_err!("direct I/O range overflow: offset={offset}, length={length}")
    })?;
    let read_offset = offset - offset % alignment_u64;
    let read_end = requested_end
        .checked_next_multiple_of(alignment_u64)
        .ok_or_else(|| vortex_err!("direct I/O aligned end overflow"))?;
    let read_length = usize::try_from(read_end - read_offset)?;
    let slice_start = usize::try_from(offset - read_offset)?;
    let slice_end = slice_start.checked_add(length).ok_or_else(|| {
        vortex_err!("direct I/O range overflow: offset={offset}, length={length}")
    })?;

    Ok(DirectIoRange {
        read_offset,
        read_length,
        requested_range: slice_start..slice_end,
    })
}

fn read_direct_at(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
    required_bytes: usize,
    constraints: DirectIoConstraints,
) -> io::Result<usize> {
    let mut initialized = 0;
    while initialized < required_bytes {
        let initialized_u64 = u64::try_from(initialized)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        let read_offset = offset
            .checked_add(initialized_u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        let bytes_read = match file.read_at(&mut buffer[initialized..], read_offset) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "direct read returned {initialized} bytes, but {required_bytes} bytes were required"
                ),
            ));
        }
        initialized += bytes_read;
        if initialized < required_bytes
            && (!initialized.is_multiple_of(constraints.offset_alignment)
                || !initialized.is_multiple_of(constraints.memory_alignment))
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "direct read returned an unaligned short read of {initialized} bytes before the required {required_bytes} bytes"
                ),
            ));
        }
    }

    Ok(initialized)
}

fn direct_io_constraints(file: &File) -> VortexResult<DirectIoConstraints> {
    let Ok(stat) = rustix::fs::statx(
        file,
        c"",
        AtFlags::EMPTY_PATH | AtFlags::STATX_DONT_SYNC,
        StatxFlags::DIOALIGN,
    ) else {
        return Ok(DirectIoConstraints::default());
    };
    if stat.stx_mask & StatxFlags::DIOALIGN.bits() == 0 {
        return Ok(DirectIoConstraints::default());
    }

    let Ok(memory_alignment) = usize::try_from(stat.stx_dio_mem_align) else {
        return Ok(DirectIoConstraints::default());
    };
    let Ok(offset_alignment) = usize::try_from(stat.stx_dio_offset_align) else {
        return Ok(DirectIoConstraints::default());
    };
    if memory_alignment == 0 || offset_alignment == 0 {
        return Ok(DirectIoConstraints::default());
    }
    vortex_ensure!(
        memory_alignment.is_power_of_two(),
        "direct I/O memory alignment must be a power of two, got {memory_alignment}"
    );
    vortex_ensure!(
        offset_alignment.is_power_of_two(),
        "direct I/O offset alignment must be a power of two, got {offset_alignment}"
    );

    Ok(DirectIoConstraints {
        memory_alignment,
        offset_alignment,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0, 0, 4096, 0, 0, 0)]
    #[case(5, 0, 4096, 5, 0, 0)]
    #[case(5, 10, 4096, 0, 4096, 5)]
    #[case(4090, 20, 4096, 0, 8192, 4090)]
    #[case(4096, 4096, 4096, 4096, 4096, 0)]
    #[case(513, 1, 512, 512, 512, 1)]
    #[case(4096, 8193, 4096, 4096, 12288, 0)]
    fn widens_direct_read_to_block_boundaries(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] alignment: usize,
        #[case] expected_offset: u64,
        #[case] expected_length: usize,
        #[case] expected_prefix: usize,
    ) -> VortexResult<()> {
        assert_eq!(
            direct_io_range(offset, length, alignment)?,
            DirectIoRange {
                read_offset: expected_offset,
                read_length: expected_length,
                requested_range: expected_prefix..expected_prefix + length,
            }
        );
        Ok(())
    }

    #[rstest]
    #[case(u64::MAX, 2, 4096)]
    #[case(0, 1, 0)]
    fn rejects_invalid_direct_read_range(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] alignment: usize,
    ) {
        assert!(direct_io_range(offset, length, alignment).is_err());
    }

    #[test]
    fn aligned_ranges_cover_requested_bytes() -> VortexResult<()> {
        for alignment in [512, 4096] {
            for offset in 0..alignment * 2 {
                for length in [0, 1, alignment - 1, alignment, alignment + 1] {
                    let range = direct_io_range(offset as u64, length, alignment)?;
                    if length == 0 {
                        assert_eq!(range.read_length, 0);
                        continue;
                    }

                    assert_eq!(range.read_offset % alignment as u64, 0);
                    assert_eq!(range.read_length % alignment, 0);
                    assert_eq!(range.requested_range.len(), length);
                    assert!(range.requested_range.end <= range.read_length);
                    assert_eq!(
                        range.read_offset + range.requested_range.start as u64,
                        offset as u64
                    );
                }
            }
        }
        Ok(())
    }
}
