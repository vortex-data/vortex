use crate::ByteBufferMut;

unsafe impl compio::buf::IoBuf for ByteBufferMut {
    fn as_buf_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn buf_len(&self) -> usize {
        self.len()
    }

    fn buf_capacity(&self) -> usize {
        self.capacity()
    }
}

impl compio::buf::SetBufInit for ByteBufferMut {
    unsafe fn set_buf_init(&mut self, len: usize) {
        unsafe { self.set_len(len) }
    }
}

unsafe impl compio::buf::IoBufMut for ByteBufferMut {
    fn as_buf_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}
