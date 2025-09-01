// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::{BoxFuture, LocalBoxFuture};
use futures::{FutureExt, StreamExt};
use smol::unblock;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};

/// A future representing an in-flight read operation.
///
/// If this [`Read`] object is dropped prior to the I/O operation being submitted, it **may** be
/// skipped by the runtime. If it has already been submitted, it will continue to completion.
pub struct Read(ReadState);

impl Read {
    pub fn ready(result: VortexResult<ByteBuffer>) -> Self {
        Read(ReadState::Ready(Some(result)))
    }

    pub fn future() -> (Self, ReadCompletion) {
        let (send, recv) = oneshot::channel();
        (Read(ReadState::Future(recv)), ReadCompletion(send))
    }
}

enum ReadState {
    Ready(Option<VortexResult<ByteBuffer>>),
    Future(oneshot::Receiver<VortexResult<ByteBuffer>>),
}

/// A handle to complete a pending read operation.
pub struct ReadCompletion(oneshot::Sender<VortexResult<ByteBuffer>>);

impl ReadCompletion {
    /// Returns true if the read has been canceled and the result will not be delivered.
    pub fn is_canceled(&self) -> bool {
        self.0.is_closed()
    }

    pub fn complete(self, result: VortexResult<ByteBuffer>) {
        if let Err(e) = self.0.send(result) {
            log::trace!("I/O request cancelled while in-flight: {e}");
        }
    }
}

impl Future for Read {
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut self.0 {
            ReadState::Ready(maybe_result) => Poll::Ready(
                maybe_result
                    .take()
                    .vortex_expect("Read future polled after completion"),
            ),
            ReadState::Future(fut) => match ready!(fut.poll_unpin(cx)) {
                Ok(result) => Poll::Ready(result),
                Err(e) => Poll::Ready(Err(vortex_err!(
                    "Failed to read from file, IoTask dropped by runtime: {e}"
                ))),
            },
        }
    }
}

pub trait IoSource: Send + Sync + 'static {
    // FIXME(ngates): non-owned
    fn name(&self) -> String;

    fn coalescing_window(&self) -> Option<u64>;

    fn concurrency(&self) -> usize;

    /// Returns a shared future that resolves to the byte size of the underlying data source.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;

    /// Perform a single read operation.
    ///
    /// The returned future must be `Send`, and should not require a specific runtime to drive it.
    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>>;

    fn read_local(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> LocalBoxFuture<'static, VortexResult<ByteBuffer>> {
        self.read_send(offset, length, alignment).boxed_local()
    }
}

impl IoSource for ByteBuffer {
    fn name(&self) -> String {
        format!("ByteBuffer({})", self.len())
    }

    fn coalescing_window(&self) -> Option<u64> {
        None
    }

    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let buffer = self.clone();
        async move {
            if offset + length as u64 > buffer.len() as u64 {
                return Err(vortex_err!("Read out of bounds"));
            }
            let mut slice = ByteBufferMut::with_capacity_aligned(length, alignment);
            unsafe { slice.set_len(length) };
            slice
                .as_mut_slice()
                .copy_from_slice(&buffer.as_slice()[offset as usize..offset as usize + length]);
            Ok(slice.freeze())
        }
        .boxed()
    }
}

pub struct FileIoSource {
    file: Arc<File>,
    path: String,
}

impl FileIoSource {
    pub fn try_new<P: AsRef<Path>>(path: P) -> VortexResult<Self> {
        let path = path.as_ref();
        let name = path.to_string_lossy().to_string();
        let file = File::open(path).map_err(VortexError::from)?;
        Ok(Self {
            file: Arc::new(file),
            path: name,
        })
    }
}

impl IoSource for FileIoSource {
    fn name(&self) -> String {
        self.path.clone()
    }

    fn coalescing_window(&self) -> Option<u64> {
        Some(8192) // 8 KB
    }

    fn concurrency(&self) -> usize {
        128
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        async move {
            let metadata = file.metadata().map_err(VortexError::from)?;
            Ok(metadata.len())
        }
        .boxed()
    }

    #[tracing::instrument(level = "debug", skip(self), fields(path = %self.path, offset, length, alignment = ?alignment))]
    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let file = self.file.clone();
        unblock(move || {
            let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
            unsafe { buffer.set_len(length) };
            match file.read_exact_at(&mut buffer, offset) {
                Ok(()) => Ok(buffer.freeze()),
                Err(e) => Err(VortexError::from(e)),
            }
        })
        .boxed()
    }
}

#[cfg(feature = "object_store")]
pub struct ObjectStoreIoSource {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
    concurrency: usize,
    coalesce_window: u64, // In bytes
}

#[cfg(feature = "object_store")]
impl ObjectStoreIoSource {
    pub fn new(store: Arc<dyn object_store::ObjectStore>, path: object_store::path::Path) -> Self {
        Self {
            store,
            path,
            concurrency: 128,
            coalesce_window: 1 * 1024 * 1024, // 1 MB
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    pub fn with_coalesce_window(mut self, window: u64) -> Self {
        // Currently a no-op, as coalescing is always enabled with a fixed window.
        self.coalesce_window = window;
        self
    }
}

#[cfg(feature = "object_store")]
impl IoSource for ObjectStoreIoSource {
    fn name(&self) -> String {
        self.path.to_string()
    }

    fn coalescing_window(&self) -> Option<u64> {
        Some(2 * 1024 * 1024) // +/- 2 MB
    }

    fn concurrency(&self) -> usize {
        192
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = self.store.clone();
        let path = self.path.clone();
        async move {
            Ok(store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)?)
        }
        .boxed()
    }

    #[tracing::instrument(level = "debug", skip(self), fields(path = %self.path, offset, length, alignment = ?alignment))]
    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let store = self.store.clone();
        let path = self.path.clone();

        async move {
            // Instead of calling `ObjectStore::get_range`, we expand the implementation and run it
            // ourselves to avoid a second copy to align the buffer. Instead, we can write directly
            // into the aligned buffer.
            let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);

            let response = store
                .get_opts(
                    &path,
                    object_store::GetOptions {
                        range: Some(object_store::GetRange::Bounded(
                            offset..offset + length as u64,
                        )),
                        ..Default::default()
                    },
                )
                .await?;

            let buffer = match response.payload {
                object_store::GetResultPayload::File(file, _) => {
                    // SAFETY: We're setting the length to the exact size we're about to read.
                    // The read_exact_at call will either fill the entire buffer or return an error,
                    // ensuring no uninitialized memory is exposed.
                    unsafe { buffer.set_len(length) };
                    unblock(move || {
                        file.read_exact_at(&mut buffer, offset)?;
                        Ok::<_, io::Error>(buffer)
                    })
                    .await
                    .map_err(io::Error::other)?
                }
                object_store::GetResultPayload::Stream(mut byte_stream) => {
                    while let Some(bytes) = byte_stream.next().await {
                        buffer.extend_from_slice(&bytes?);
                    }
                    buffer
                }
            };

            Ok(buffer.freeze())
        }
        .boxed()
    }
}
