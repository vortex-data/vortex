// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use std::fs::File;
use std::io;
use std::ops::{Deref, Range};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::oneshot;
use tokio::task::spawn_blocking;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{ResultExt, VortexExpect, VortexResult, vortex_err};

use crate::dispatcher::Dispatch;
use crate::dispatcher::tokio::TOKIO_DISPATCHER;
use crate::{IoBuf, PerformanceHint, ReadAt, VortexReadAt, VortexWrite};

/// A generic (unsealed) trait for implementing read-at operations via dispatched I/O.
///
/// Note that this trait does not require a `Send` bound on the returned future since it is
/// dispatched onto a Tokio [`LocalSet`].
///
/// See [`TokioDispatchedIo`] to wrap this implementation into a Vortex [`ReadAt`].
pub trait TokioReadAt {
    fn read_at(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> impl Future<Output = VortexResult<ByteBuffer>>;
}

/// A wrapper for dispatching I/O operations to a Tokio runtime.
#[derive(Clone)]
pub struct TokioDispatchedIo {
    send: flume::Sender<ReadRequest>,
    performance_hint: PerformanceHint,
}

/// A read request that is sent to the Tokio runtime for processing.
struct ReadRequest {
    offset: u64,
    len: usize,
    alignment: Alignment,
    response: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl TokioDispatchedIo {
    /// Wraps an existing [`TokioReadAt`] implementation to provide a Vortex-compatible `ReadAt`.
    pub fn new<R: TokioReadAt + Send + 'static>(
        read: R,
        performance_hint: PerformanceHint,
    ) -> Self {
        let (send, recv) = flume::unbounded::<ReadRequest>();
        TOKIO_DISPATCHER
            .dispatch(move || {
                async move {
                    // `recv.recv_async()` returns error only if all senders have been dropped.
                    while let Ok(request) = recv.recv_async().await {
                        let ReadRequest {
                            offset,
                            len,
                            alignment,
                            response,
                        } = request;

                        let result = read.read_at(offset, len, alignment).await;
                        if response.send(result).is_err() {
                            log::trace!("Failed to send Tokio read result back to requester");
                        }
                    }
                }
            })
            .vortex_expect("Failed to dispatch Tokio read task, dispatcher fatally dead");
        Self {
            send,
            performance_hint,
        }
    }
}

#[async_trait]
impl ReadAt for TokioDispatchedIo {
    async fn read_range(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        // TODO(ngates): we should find a stack-based oneshot channel to avoid a heap allocation
        //  on every read request. I think internally the Tokio dispatcher may also be more
        //  expensive than perhaps it needs to be. That said, this async_trait also heap-allocates
        //  the future!
        let (send, recv) = oneshot::channel();
        self.send
            .send(ReadRequest {
                offset,
                len,
                alignment,
                response: send,
            })
            .map_err(|e| vortex_err!("Tokio dispatcher send error: {e}"))?;
        recv.await
            .map_err(|e| vortex_err!("Tokio dispatcher died: {e}"))
            .unnest()
    }

    async fn size(&self) -> VortexResult<u64> {
        todo!("TokioDispatchedIo does not support size yet");
    }

    fn performance_hint(&self) -> PerformanceHint {
        self.performance_hint.clone()
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

impl VortexReadAt for TokioFile {
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(range, alignment))
    )]
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        let len = usize::try_from(range.end - range.start).vortex_expect("range too big for usize");
        let this = self.clone();

        spawn_blocking(move || {
            let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
            unsafe { buffer.set_len(len) }
            this.read_exact_at(&mut buffer, range.start)?;
            Ok(buffer.freeze())
        })
        .await?
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    async fn size(&self) -> io::Result<u64> {
        self.metadata().map(|metadata| metadata.len())
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
    use vortex_buffer::Alignment;

    use crate::VortexReadAt;
    use crate::tokio::TokioFile;

    #[tokio::test]
    async fn test_shared_file() {
        let mut tmpfile = NamedTempFile::new().unwrap();
        write!(tmpfile, "0123456789").unwrap();

        let shared_file = TokioFile::open(tmpfile.path()).unwrap();

        let first_half = shared_file
            .read_byte_range(0..5, Alignment::none())
            .await
            .unwrap();

        let second_half = shared_file
            .read_byte_range(5..10, Alignment::none())
            .await
            .unwrap();

        assert_eq!(first_half.as_ref(), "01234".as_bytes());
        assert_eq!(second_half.as_ref(), "56789".as_bytes());
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
