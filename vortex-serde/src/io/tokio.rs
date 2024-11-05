#![cfg(feature = "tokio")]

use std::fs::File;
use std::future::Future;
use std::io;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use bytes::BytesMut;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::runtime::{Handle, Runtime};
use vortex_buffer::io_buf::IoBuf;
use vortex_error::vortex_panic;

use super::VortexReadAt;
use crate::file::AsyncRuntime;
use crate::io::{VortexRead, VortexWrite};

pub struct TokioAdapter<IO>(pub IO);

impl<IO: AsyncRead + Unpin> VortexRead for TokioAdapter<IO> {
    async fn read_into(&mut self, mut buffer: BytesMut) -> io::Result<BytesMut> {
        self.0.read_exact(buffer.as_mut()).await?;
        Ok(buffer)
    }
}

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

impl AsyncRuntime for Runtime {
    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.block_on(fut)
    }
}

impl AsyncRuntime for Handle {
    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.block_on(fut)
    }
}

/// A cheaply cloneable, readonly file that executes operations
/// on a tokio blocking threadpool.
///
/// We use this because the builtin tokio `File` type is not `Clone` and
/// also does actually implement a `read_exact_at` operation.
#[derive(Debug, Clone)]
pub struct TokioFile(Arc<Inner>);

// Implement an inner file.
#[derive(Debug)]
struct Inner {
    file: ManuallyDrop<File>,
}

impl TokioFile {
    /// Open a file on the current file system.
    ///
    /// The `TokioFile` takes ownership of the file descriptor, and can be cloned
    /// many times without opening a new file descriptor. When the last instance
    /// of the `TokioFile` is dropped, the file descriptor is closed.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let f = File::open(path)?;
        let inner = Arc::new(Inner {
            file: ManuallyDrop::new(f),
        });

        Ok(Self(inner))
    }
}

// Implement deref coercion for non-mut `File` methods on `TokioFile`.
impl Deref for TokioFile {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0.file
    }
}

impl VortexReadAt for TokioFile {
    fn read_at_into(
        &self,
        pos: u64,
        buffer: BytesMut,
    ) -> impl Future<Output = io::Result<BytesMut>> + 'static {
        let this = self.clone();

        async move {
            let res = tokio::task::spawn_blocking(move || {
                let mut buffer = buffer;
                match this.read_exact_at(&mut buffer, pos) {
                    Ok(()) => Ok(buffer),
                    Err(e) => Err(e),
                }
            });

            res.await.map_err(|_canceled| {
                io::Error::new(
                    io::ErrorKind::Other,
                    "blocking read_exact_at task was canceled",
                )
            })?
        }
    }

    fn size(&self) -> impl Future<Output = u64> + 'static {
        let this = self.clone();

        async move {
            let res = tokio::task::spawn_blocking(move || {
                this.metadata()
                    .unwrap_or_else(|e| vortex_panic!("access TokioFile metadata: {e}"))
                    .len()
            })
            .await;

            res.unwrap_or_else(|e| vortex_panic!("Joining spawn_blocking: size: {e}"))
        }
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
    use std::io::Write;

    use bytes::BytesMut;
    use tempfile::NamedTempFile;

    use crate::io::{TokioFile, VortexReadAt};

    #[tokio::test]
    async fn test_shared_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        write!(tmpfile, "0123456789").unwrap();

        let shared_file = TokioFile::open(tmpfile.path()).unwrap();

        let first_half = BytesMut::zeroed(5);
        let first_half = shared_file.read_at_into(0, first_half).await.unwrap();

        let second_half = BytesMut::zeroed(5);
        let second_half = shared_file.read_at_into(5, second_half).await.unwrap();

        assert_eq!(first_half.freeze(), "01234".as_bytes());
        assert_eq!(second_half.freeze(), "56789".as_bytes());
    }
}
