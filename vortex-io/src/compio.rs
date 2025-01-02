use std::future::Future;
use std::io;

use bytes::Bytes;
use compio::fs::File;
use compio::io::AsyncReadAtExt;
use compio::BufResult;
use vortex_error::VortexUnwrap;

use crate::VortexReadAt;

impl VortexReadAt for File {
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        let this = self.clone();
        let buffer = Vec::with_capacity(len.try_into().vortex_unwrap());
        async move {
            // Turn the buffer into a static slice.
            let BufResult(res, buffer) = this.read_exact_at(buffer, pos).await;
            res.map(|_| Bytes::from(buffer))
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
        assert_eq!(read.as_ref(), "2345".as_bytes());
    }
}
