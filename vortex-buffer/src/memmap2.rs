// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::Bytes;
use memmap2::Mmap;
use memmap2::MmapMut;

use crate::ByteBuffer;
use crate::ByteBufferMut;

impl From<Mmap> for ByteBuffer {
    fn from(value: Mmap) -> Self {
        ByteBuffer::from(Bytes::from_owner(value))
    }
}

impl From<MmapMut> for ByteBufferMut {
    /// Adopt a writable mapping without copying it, to write through in place.
    fn from(value: MmapMut) -> Self {
        ByteBufferMut::from_owner(value)
    }
}

#[cfg(test)]
mod tests {
    use memmap2::MmapMut;

    use crate::ByteBufferMut;

    #[test]
    #[cfg_attr(miri, ignore = "Miri does not support mmap")]
    fn writable_mappings_are_written_in_place() -> std::io::Result<()> {
        let mut mapping = MmapMut::map_anon(16)?;
        mapping[0] = 1;
        let ptr = mapping.as_ptr();

        let mut buffer = ByteBufferMut::from(mapping);
        assert_eq!(buffer.as_ptr(), ptr);
        buffer[1] = 2;
        assert_eq!(&buffer[..3], &[1, 2, 0]);
        Ok(())
    }
}
