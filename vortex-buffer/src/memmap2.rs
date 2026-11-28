// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bytes::Bytes;
use memmap2::Mmap;

use crate::ByteBuffer;

impl From<Mmap> for ByteBuffer {
    fn from(value: Mmap) -> Self {
        ByteBuffer::from(Bytes::from_owner(value))
    }
}

#[cfg(test)]
mod tests {
    use memmap2::MmapMut;

    use crate::ByteBuffer;

    #[test]
    fn from_mmap_preserves_pointer_and_contents() {
        let mut mmap_mut = MmapMut::map_anon(16).unwrap();
        mmap_mut[..5].copy_from_slice(b"hello");
        let mmap = mmap_mut.make_read_only().unwrap();
        let ptr = mmap.as_ptr();

        let buf = ByteBuffer::from(mmap);
        assert_eq!(buf.as_ptr(), ptr);
        assert_eq!(buf.len(), 16);
        assert_eq!(&buf[..5], b"hello");
    }
}
