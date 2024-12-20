use bytes::Buf;

use crate::ByteBuffer;

impl Buf for ByteBuffer {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(
            cnt <= self.remaining(),
            "cannot advance past `remaining`: {:?} <= {:?}",
            cnt,
            self.remaining(),
        );
        unsafe {
            // SAFETY: We've checked that `cnt` <= `self.remaining()` and we know that
            // `self.remaining()` <= `self.cap`.
            self.advance_unchecked(cnt);
        }
    }

    // TODO(ngates): implement copy_to_bytes
}

/// An extension to the Buf trait to return an aligned `ByteBuffer`.
pub trait AlignedBuf: Buf {
    fn copy_to_byte_buffer(&mut self, len: usize) -> ByteBuffer {
        ByteBuffer::from(self.copy_to_bytes(len))
    }
}
