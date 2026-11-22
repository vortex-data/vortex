// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::BytesMut;
use futures::TryStreamExt;
use futures::stream::FuturesUnordered;
use object_store::MultipartUpload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::path::Path;
use vortex_error::VortexResult;

use crate::IoBuf;
use crate::VortexWrite;
#[cfg(not(target_arch = "wasm32"))]
use crate::compat::Compat;

/// Adapter type to write data through an [`ObjectStore`] instance.
///
/// Each active instance exclusively owns its underlying [`MultipartUpload`]. After writing, the
/// caller must call [`VortexWrite::shutdown`] to complete and publish the object.
///
/// Multipart initiation, part uploads, and completion run on the caller's async runtime so
/// concurrent writers scale across the caller's worker threads. If a part upload or completion
/// fails, the writer schedules a background abort once, returns the part/completion error, and
/// rejects any subsequent writes or completion.
///
/// Dropping an uncompleted writer (or cancelling an in-flight `write_all`, `flush`, or `shutdown`
/// future) schedules an asynchronous abort on a process-lifetime executor (`smol` with a `Compat`
/// Tokio context on native targets) so cleanup survives both task cancellation and shutdown of the
/// caller's Tokio runtime. A cancelled `shutdown()` aborts in the background and does not report
/// whether the object was published. Process termination can still prevent cleanup. If the
/// underlying [`ObjectStore`] holds pooled keep-alive connections opened on a caller Tokio runtime
/// that later goes idle (without shutting down) and retries are disabled (`max_retries: 0`), an
/// abort that reuses such a connection still depends on that runtime remaining driven or retries
/// opening a fresh connection.
pub struct ObjectStoreWrite {
    state: UploadState,
}

enum UploadState {
    Active {
        upload: Box<dyn MultipartUpload>,
        buffer: BytesMut,
    },
    Completed(PutResult),
    Aborted,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;
const BUFFER_SIZE: usize = 128 * 1024 * 1024;

impl ObjectStoreWrite {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            state: UploadState::Active {
                upload,
                buffer: BytesMut::with_capacity(CHUNK_SIZE),
            },
        })
    }

    pub fn put_result(&self) -> Option<&PutResult> {
        match &self.state {
            UploadState::Completed(put_result) => Some(put_result),
            _ => None,
        }
    }

    /// Schedule detached cleanup when a part/completion fails, a future is cancelled, or the
    /// writer is dropped before completing.
    fn abort_in_background(&mut self) {
        if !matches!(self.state, UploadState::Active { .. }) {
            return;
        }
        let UploadState::Active { mut upload, .. } =
            std::mem::replace(&mut self.state, UploadState::Aborted)
        else {
            vortex_error::vortex_panic!("checked active state above");
        };
        spawn_detached_cleanup(async move {
            if let Err(err) = upload.abort().await {
                tracing::warn!(%err, "failed to abort multipart upload in background");
            }
        });
    }

    fn ensure_active(&mut self) -> io::Result<(&mut Box<dyn MultipartUpload>, &mut BytesMut)> {
        match &mut self.state {
            UploadState::Active { upload, buffer } => Ok((upload, buffer)),
            UploadState::Completed(_) => {
                Err(io::Error::other("multipart upload already completed"))
            }
            UploadState::Aborted => Err(io::Error::other(
                "multipart upload already failed or aborted",
            )),
        }
    }

    /// Flush buffered chunks of up to `CHUNK_SIZE` while more than `min_remaining` bytes remain.
    ///
    /// Arms [`AbortOnCancel`] once chunks are removed from the buffer; on failure or cancellation,
    /// dropping the guard schedules a background abort once and returns the part error.
    async fn flush_parts(&mut self, min_remaining: usize) -> io::Result<()> {
        let (_, buffer) = self.ensure_active()?;
        if buffer.len() <= min_remaining {
            return Ok(());
        }

        let guard = AbortOnCancel::new(self);
        let part_result = {
            let (upload, buffer) = guard.writer.ensure_active()?;
            let parts = FuturesUnordered::new();
            while buffer.len() > min_remaining {
                let size = buffer.len().min(CHUNK_SIZE);
                let payload = buffer.split_to(size).freeze();
                parts.push(upload.put_part(PutPayload::from_bytes(payload)));
            }
            parts.try_collect::<Vec<_>>().await
        };

        part_result?;
        guard.disarm();
        Ok(())
    }
}

fn spawn_detached_cleanup(fut: impl Future<Output = ()> + Send + 'static) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        smol::spawn(Compat::new(fut)).detach();
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm-bindgen"))]
    {
        wasm_bindgen_futures::spawn_local(fut);
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
    {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(fut);
            return;
        }
        drop(fut);
        tracing::warn!("no async runtime available to abort multipart upload on drop");
    }
}

/// Drop guard that aborts an active upload if an in-flight `write_all`, `flush`, or `shutdown`
/// future is dropped before completing.
struct AbortOnCancel<'a> {
    writer: &'a mut ObjectStoreWrite,
    armed: bool,
}

impl<'a> AbortOnCancel<'a> {
    fn new(writer: &'a mut ObjectStoreWrite) -> Self {
        Self {
            writer,
            armed: true,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for AbortOnCancel<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.writer.abort_in_background();
        }
    }
}

impl Drop for ObjectStoreWrite {
    fn drop(&mut self) {
        self.abort_in_background();
    }
}

impl VortexWrite for ObjectStoreWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        let (_, buf) = self.ensure_active()?;
        buf.extend_from_slice(buffer.as_slice());

        if buf.len() > BUFFER_SIZE {
            self.flush_parts(CHUNK_SIZE).await?;
        }

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.flush_parts(CHUNK_SIZE).await
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        if matches!(self.state, UploadState::Completed(_)) {
            return Ok(());
        }

        self.flush_parts(0).await?;

        let guard = AbortOnCancel::new(self);
        let (upload, _) = guard.writer.ensure_active()?;
        let put_result = upload.complete().await?;
        guard.writer.state = UploadState::Completed(put_result);
        guard.disarm();
        Ok(())
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests;
