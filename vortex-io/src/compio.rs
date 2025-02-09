use std::io;
use std::ops::Range;

use bytes::{Bytes, BytesMut};
use compio::fs::File;
use compio::io::AsyncReadAtExt;
use compio::BufResult;
use vortex_error::VortexExpect;

use crate::VortexReadAt;

impl VortexReadAt for File {
    async fn read_byte_range(&self, range: Range<u64>) -> io::Result<Bytes> {
        let len = usize::try_from(range.end - range.start).vortex_expect("range too big for usize");
        let buffer = BytesMut::with_capacity(len);
        let BufResult(result, buffer) = self.read_exact_at(buffer, range.start).await;
        result?;
        Ok(buffer.freeze())
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

    use crate::VortexReadAt;

    #[cfg_attr(miri, ignore)]
    #[compio::test]
    async fn test_read_at_compio_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        write!(tmpfile, "0123456789").unwrap();

        // Open up a file handle in compio land
        let file = File::open(tmpfile.path()).await.unwrap();

        // Use the file as a VortexReadAt instance.
        let read = file.read_byte_range(2..6).await.unwrap();
        assert_eq!(read.as_ref(), "2345".as_bytes());
    }
}
