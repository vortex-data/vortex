// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{ResultExt, VortexResult};

use crate::dispatcher::Dispatch;
use crate::dispatcher::tokio::TokioDispatcher;
use crate::{IoBuf, ReadAt, VortexWrite};

static DISPATCHER: LazyLock<TokioDispatcher> = LazyLock::new(|| TokioDispatcher::new(1));

/// A generic (unsealed) trait for implementing read-at operations via dispatched I/O.
///
/// Note that this trait does not require a `Send` bound on the returned future since it is
/// dispatched onto a Tokio [`LocalSet`].
///
/// See [`TokioDispatchedIo`] to wrap this implementation into a Vortex [`ReadAt`].
pub trait TokioReadAt: Send + Sync + 'static {
    fn read_at(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> impl Future<Output = VortexResult<ByteBuffer>> + Send;

    fn size(&self) -> impl Future<Output = VortexResult<u64>> + Send;
}

/// A wrapper for dispatching I/O operations to a Tokio runtime.
// TODO(ngates): the current implementation creates an `Arc<dyn TokioReadAt>` and send it into
//  the dispatcher on each call. An alternative would be to send the read object once during
//  construction, and then use a mpsc channel to send read requests into the runtime. This would
//  allow us to support `TokioReadAt` implementations that return `!Send` futures.
#[derive(Clone)]
pub struct TokioDispatchedIo<R>(Arc<R>);

impl<R: TokioReadAt> TokioDispatchedIo<R> {
    /// Wraps an existing [`TokioReadAt`] implementation to provide a Vortex-compatible `ReadAt`.
    pub fn new(read: R) -> Self {
        Self(Arc::new(read))
    }
}

#[async_trait]
impl<R: TokioReadAt> ReadAt for TokioDispatchedIo<R> {
    async fn read_range(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let read = self.0.clone();
        DISPATCHER
            .dispatch(|| async move { read.read_at(offset, len, alignment).await })?
            .await
            .unnest()
    }

    async fn size(&self) -> VortexResult<u64> {
        let read = self.0.clone();
        DISPATCHER
            .dispatch(|| async move { read.size().await })?
            .await
            .unnest()
    }
}

/// A cheaply cloneable, readonly file that executes operations
/// on a tokio blocking threadpool.
///
/// We use this because tokio's [`File`](tokio::fs::File) type is not `Clone` and
/// also does not implement a `read_exact_at` operation.
#[derive(Debug, Clone)]
pub struct TokioFile(Arc<File>);

impl TokioFile {
    pub fn new(file: File) -> Self {
        Self(Arc::new(file))
    }

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

    use crate::tokio::TokioFile;

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
