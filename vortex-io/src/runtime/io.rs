// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::{BoxFuture, LocalBoxFuture};
use futures::FutureExt;
use smol::unblock;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};

pub struct Read(pub(crate) ReadState);

impl Read {
    pub fn ready(result: VortexResult<ByteBuffer>) -> Self {
        Read(ReadState::Ready(Some(result)))
    }

    pub fn future() -> (Self, ReadCompletion) {
        let (send, recv) = oneshot::channel();
        (Read(ReadState::Future(recv)), ReadCompletion(send))
    }
}

pub enum ReadState {
    Ready(Option<VortexResult<ByteBuffer>>),
    Future(oneshot::Receiver<VortexResult<ByteBuffer>>),
}

pub struct ReadCompletion(oneshot::Sender<VortexResult<ByteBuffer>>);

impl ReadCompletion {
    pub fn complete(self, result: VortexResult<ByteBuffer>) {
        self.0
            .send(result)
            .map_err(|e| vortex_err!("Sender dropped: {e}"))
            .vortex_expect("Failed to send read completion");
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
        Some(1024 * 1024) // 1 MB
    }

    fn concurrency(&self) -> usize {
        64
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

    fn read_send(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let store = self.store.clone();
        let path = self.path.clone();

        async move {
            let range = offset..offset + length as u64;
            // FIXME(ngates): see object_store.rs
            let bytes = store.get_range(&path, range).await?;
            let buffer = ByteBuffer::from(bytes).aligned(alignment);
            Ok(buffer)
        }
        .boxed()
    }
}
