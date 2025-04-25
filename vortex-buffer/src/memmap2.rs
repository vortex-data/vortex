use bytes::Bytes;
use memmap2::Mmap;

use crate::ByteBuffer;

impl From<Mmap> for ByteBuffer {
    fn from(value: Mmap) -> Self {
        ByteBuffer::from(Bytes::from_owner(value))
    }
}
