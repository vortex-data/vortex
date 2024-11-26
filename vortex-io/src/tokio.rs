use std::fs::File;
use std::future::{self, Future};
use std::io;
use std::ops::Deref;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use vortex_buffer::io_buf::IoBuf;

use crate::aligned::AlignedBytesMut;
use crate::{VortexReadAt, VortexWrite, ALIGNMENT};

pub struct TokioAdapter<IO>(pub IO);

impl<W: AsyncWrite + Unpin> VortexWrite for TokioAdapter<W> {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.0.write_all(buffer.as_slice()).await?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.0.flush().await
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        self.0.shutdown().await
    }
}

/// A cheaply cloneable, readonly file that executes operations
/// on a tokio blocking threadpool.
///
/// We use this because the builtin tokio `File` type is not `Clone` and
/// also does actually implement a `read_exact_at` operation.
#[derive(Debug, Clone)]
pub struct TokioFile(Arc<File>);

impl TokioFile {
    /// Open a file on the current file system.
    ///
    /// The `TokioFile` takes ownership of the file descriptor, and can be cloned
    /// many times without opening a new file descriptor. When the last instance
    /// of the `TokioFile` is dropped, the file descriptor is closed.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let f = File::open(path)?;

        Ok(Self(Arc::new(f)))
    }
}

// Implement deref coercion for non-mut `File` methods on `TokioFile`.
impl Deref for TokioFile {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl VortexReadAt for TokioFile {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        let this = self.clone();

        let mut buffer = AlignedBytesMut::<ALIGNMENT>::with_capacity(len as usize);
        unsafe {
            buffer.set_len(len as usize);
        }
        match this.read_exact_at(&mut buffer, pos) {
            Ok(()) => future::ready(Ok(buffer.freeze())),
            Err(e) => future::ready(Err(e)),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        let this = self.clone();

        async move { this.metadata().map(|metadata| metadata.len()) }
    }
}

impl VortexWrite for tokio::fs::File {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        AsyncWriteExt::write_all(self, buffer.as_slice()).await?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        AsyncWriteExt::flush(self).await
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        AsyncWriteExt::shutdown(self).await
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::ops::Deref;
    use std::os::unix::fs::FileExt;

    use tempfile::NamedTempFile;

    use crate::{TokioFile, VortexReadAt};

    #[tokio::test]
    async fn test_shared_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        write!(tmpfile, "0123456789").unwrap();

        let shared_file = TokioFile::open(tmpfile.path()).unwrap();

        let first_half = shared_file.read_byte_range(0, 5).await.unwrap();

        let second_half = shared_file.read_byte_range(5, 5).await.unwrap();

        assert_eq!(&first_half, "01234".as_bytes());
        assert_eq!(&second_half, "56789".as_bytes());
    }

    #[test]
    fn test_drop_semantics() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "test123").unwrap();

        // Transfer ownership of the file into our Tokio file.
        let tokio_file = TokioFile::open(file.path()).unwrap();
        // Delete the file, so that tokio_file's owned FD is the only thing keeping it around.
        std::fs::remove_file(file.path()).unwrap();

        // Create a function to test if we can read from the file
        let can_read = |file: &File| {
            let mut buffer = vec![0; 7];
            file.read_exact_at(&mut buffer, 0).is_ok()
        };

        // Test initial read
        assert!(can_read(tokio_file.deref()));

        // Clone the old tokio_file, then drop the old one. Because the refcount
        // of the Inner is > 0, the file handle should not be dropped.
        let tokio_file_cloned = tokio_file.clone();
        drop(tokio_file);

        // File handle should still be open and readable
        assert!(can_read(tokio_file_cloned.deref()));

        // Now, drop the cloned handle. The file should be deleted after the drop.
        drop(tokio_file_cloned);
        assert!(!std::fs::exists(file.path()).unwrap());
    }
}
