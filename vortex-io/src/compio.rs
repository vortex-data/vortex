use std::future::Future;
use std::io;

use bytes::Bytes;
use compio::buf::{IoBuf, IoBufMut, SetBufInit};
use compio::fs::File;
use compio::io::AsyncReadAtExt;
use compio::BufResult;

use crate::aligned::{AlignedBytesMut, PowerOfTwo};
use crate::{VortexReadAt, ALIGNMENT};

unsafe impl<const ALIGN: usize> IoBuf for AlignedBytesMut<ALIGN>
where
    usize: PowerOfTwo<ALIGN>,
{
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

impl<const ALIGN: usize> SetBufInit for AlignedBytesMut<ALIGN>
where
    usize: PowerOfTwo<ALIGN>,
{
    unsafe fn set_buf_init(&mut self, len: usize) {
        // The contract of this trait specifies that providing a `len` <= the current len should
        // do nothing. AlignedBytesMut by default will set the len directly without checking this.
        if self.len() < len {
            unsafe {
                self.set_len(len);
            }
        }
    }
}

unsafe impl<const ALIGN: usize> IoBufMut for AlignedBytesMut<ALIGN>
where
    usize: PowerOfTwo<ALIGN>,
{
    fn as_buf_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}

impl VortexReadAt for File {
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        let this = self.clone();
        let buffer = AlignedBytesMut::<ALIGNMENT>::with_capacity(len as usize);
        async move {
            // Turn the buffer into a static slice.
            let BufResult(res, buffer) = this.read_exact_at(buffer, pos).await;
            res.map(|_| buffer.freeze())
        }
    }

    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        let this = self.clone();
        async move { this.metadata().await.map(|metadata| metadata.len()) }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use compio::fs::File;
    use tempfile::NamedTempFile;

    use crate::VortexReadAt;

    #[cfg_attr(miri, ignore)]
    #[compio::test]
    async fn test_read_at_compio_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        write!(tmpfile, "0123456789").unwrap();

        // Open up a file handle in compio land
        let file = File::open(tmpfile.path()).await.unwrap();

        // Use the file as a VortexReadAt instance.
        let read = file.read_byte_range(2, 4).await.unwrap();
        assert_eq!(&read, "2345".as_bytes());
    }
}
