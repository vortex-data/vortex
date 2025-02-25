use std::io;
use std::ops::Range;

use compio::buf::{IoBuf, IoBufMut, SetBufInit};
use compio::fs::File;
use compio::io::AsyncReadAtExt;
use compio::BufResult;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::VortexExpect;

use crate::VortexReadAt;

/// Compio uses buffer capacity instead of buffer len (that everyone else uses) when reading
/// to fill a buffer. Since [`ByteBufferMut`] cannot be allocated with a precise capacity,
/// we need to wrap it in a struct that will keep track of the capacity.
struct FixedCapacityByteBufferMut {
    buffer: ByteBufferMut,
    capacity: usize,
}

unsafe impl IoBuf for FixedCapacityByteBufferMut {
    fn as_buf_ptr(&self) -> *const u8 {
        self.buffer.as_ptr()
    }

    fn buf_len(&self) -> usize {
        self.buffer.len()
    }

    fn buf_capacity(&self) -> usize {
        self.capacity
    }
}

impl SetBufInit for FixedCapacityByteBufferMut {
    unsafe fn set_buf_init(&mut self, len: usize) {
        unsafe { self.buffer.set_len(len) }
    }
}

unsafe impl IoBufMut for FixedCapacityByteBufferMut {
    fn as_buf_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }
}

impl VortexReadAt for File {
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        let len = usize::try_from(range.end - range.start).vortex_expect("range too big for usize");
        let buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
        let BufResult(result, buffer) = self
            .read_exact_at(
                FixedCapacityByteBufferMut {
                    buffer,
                    capacity: len,
                },
                range.start,
            )
            .await;
        result?;
        Ok(buffer.buffer.freeze())
    }

    async fn size(&self) -> io::Result<u64> {
        self.metadata().await.map(|metadata| metadata.len())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use compio::fs::File;
    use tempfile::NamedTempFile;
    use vortex_buffer::Alignment;

    use crate::VortexReadAt;

    #[cfg_attr(miri, ignore)]
    #[compio::test]
    async fn test_read_at_compio_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        write!(tmpfile, "0123456789").unwrap();

        // Open up a file handle in compio land
        let file = File::open(tmpfile.path()).await.unwrap();

        // Use the file as a VortexReadAt instance.
        let read = file.read_byte_range(2..6, Alignment::none()).await.unwrap();
        assert_eq!(read.as_ref(), "2345".as_bytes());
    }
}
