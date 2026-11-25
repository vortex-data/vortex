// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::BytesMut;
use futures::StreamExt;
use futures::channel::oneshot;
use futures::stream::FuturesUnordered;
use object_store::MultipartUpload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::UploadPart;
use object_store::path::Path;
use vortex_error::VortexResult;

use crate::IoBuf;
use crate::VortexWrite;
#[cfg(not(target_arch = "wasm32"))]
use crate::compat::Compat;

/// Adapter type to write data through an [`ObjectStore`] instance.
///
/// Call [`VortexWrite::shutdown`] to publish the object or [`Self::abort`] to discard it.
/// Upload errors preserve the original error and attempt cleanup. Dropping an uncompleted writer
/// or cancelling a write, flush, or shutdown schedules background cleanup and rejects further writes.
/// A cancelled shutdown does not report whether the object was published.
///
/// Background cleanup survives shutdown of the caller's runtime, but process termination can
/// prevent cleanup. If the underlying store reuses connections from a caller runtime that goes
/// idle without shutting down, cleanup still depends on that runtime being driven or retries
/// opening a fresh connection.
pub struct ObjectStoreWrite {
    state: UploadState,
    concurrency: usize,
}

struct ActiveUpload {
    upload: Box<dyn MultipartUpload>,
    buffer: BytesMut,
    parts: FuturesUnordered<UploadPart>,
}

enum UploadState {
    Active(ActiveUpload),
    Aborting(oneshot::Receiver<(Box<dyn MultipartUpload>, io::Result<()>)>),
    Failed(Box<dyn MultipartUpload>),
    Completed(PutResult),
    Aborted,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;

impl ObjectStoreWrite {
    /// Start a multipart upload. The object is not published until shutdown succeeds.
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            state: UploadState::Active(ActiveUpload {
                upload,
                buffer: BytesMut::with_capacity(CHUNK_SIZE),
                parts: FuturesUnordered::new(),
            }),
            concurrency: 0,
        })
    }

    /// Set the maximum number of in-flight parts. Defaults to zero, meaning unlimited.
    ///
    /// Upload concurrency controls network throughput, not CPU parallelism. The default avoids
    /// imposing an arbitrary throughput cap. Set a positive limit to bound adapter-owned buffering
    /// to that many 16 MiB parts plus one staging part, excluding caller buffers and store internals.
    /// Configure this before writing; lowering the limit does not cancel existing uploads.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// The result of a successful shutdown, or `None` before publication.
    pub fn put_result(&self) -> Option<&PutResult> {
        match &self.state {
            UploadState::Completed(put_result) => Some(put_result),
            _ => None,
        }
    }

    /// Abort this upload, waiting for pending parts before cleanup.
    ///
    /// Call this for external errors or empty output, or to await background cleanup after
    /// cancellation. Cleanup failures can be retried, including after completion fails. Aborting
    /// again after success is harmless; aborting a published object is an error.
    pub async fn abort(&mut self) -> io::Result<()> {
        match self.state {
            UploadState::Aborted => return Ok(()),
            UploadState::Completed(_) => {
                return Err(io::Error::other("multipart upload already completed"));
            }
            _ => self.abort_in_background(),
        }
        let UploadState::Aborting(receiver) = &mut self.state else {
            vortex_error::vortex_panic!("abort schedules cleanup before waiting");
        };
        let (upload, result) = receiver.await.map_err(io::Error::other)?;
        self.state = UploadState::Failed(upload);
        result?;
        self.state = UploadState::Aborted;
        Ok(())
    }

    /// Schedule cleanup independently of the caller's future and runtime. Retain its result so
    /// explicit abort can await it and retry failures without racing another cleanup attempt.
    fn abort_in_background(&mut self) {
        let (mut upload, mut parts) = match std::mem::replace(&mut self.state, UploadState::Aborted)
        {
            UploadState::Active(active) => (active.upload, active.parts),
            UploadState::Failed(upload) => (upload, FuturesUnordered::new()),
            state => {
                self.state = state;
                return;
            }
        };
        let (sender, receiver) = oneshot::channel();
        self.state = UploadState::Aborting(receiver);
        spawn_detached_cleanup(async move {
            // Stores can spawn work internally. Settle parts before aborting so late uploads
            // cannot race with cleanup, even if the caller cancelled the operation.
            while parts.next().await.is_some() {}
            let result = upload.abort().await.map_err(io::Error::from);
            if let Err(err) = &result {
                tracing::warn!(%err, "failed to abort multipart upload in background");
            }
            drop(sender.send((upload, result)));
        });
    }

    fn ensure_active(&mut self) -> io::Result<&mut ActiveUpload> {
        match &mut self.state {
            UploadState::Active(active) => Ok(active),
            UploadState::Completed(_) => {
                Err(io::Error::other("multipart upload already completed"))
            }
            _ => Err(io::Error::other(
                "multipart upload already failed or aborted",
            )),
        }
    }

    async fn upload_buffer(&mut self) -> io::Result<()> {
        let concurrency = self.concurrency;
        let active = self.ensure_active()?;
        while concurrency != 0 && active.parts.len() >= concurrency {
            active.parts.next().await.transpose()?;
        }
        // Independent, fixed-size allocations keep a submitted part from retaining a larger buffer.
        let payload = std::mem::replace(&mut active.buffer, BytesMut::with_capacity(CHUNK_SIZE));
        active.parts.push(
            active
                .upload
                .put_part(PutPayload::from_bytes(payload.freeze())),
        );
        Ok(())
    }

    async fn cleanup_error(&mut self, error: io::Error) -> io::Error {
        // Background cleanup logs its own errors; never replace the upload error with one.
        drop(self.abort().await);
        error
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
        if self.armed && matches!(self.writer.state, UploadState::Active(_)) {
            self.writer.abort_in_background();
        }
    }
}

impl Drop for ObjectStoreWrite {
    fn drop(&mut self) {
        if matches!(self.state, UploadState::Active(_)) {
            self.abort_in_background();
        }
    }
}

impl VortexWrite for ObjectStoreWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.ensure_active()?;
        let guard = AbortOnCancel::new(self);
        let mut remaining = buffer.as_slice();
        while !remaining.is_empty() {
            let active = guard.writer.ensure_active()?;
            let len = remaining.len().min(CHUNK_SIZE - active.buffer.len());
            active.buffer.extend_from_slice(&remaining[..len]);
            remaining = &remaining[len..];
            if active.buffer.len() == CHUNK_SIZE
                && let Err(error) = guard.writer.upload_buffer().await
            {
                return Err(guard.writer.cleanup_error(error).await);
            }
        }
        guard.disarm();
        Ok(buffer)
    }

    /// Wait for submitted parts without uploading a partial tail or publishing the object.
    async fn flush(&mut self) -> io::Result<()> {
        if matches!(self.state, UploadState::Completed(_)) {
            return Ok(());
        }
        self.ensure_active()?;
        let guard = AbortOnCancel::new(self);
        while let Some(result) = guard.writer.ensure_active()?.parts.next().await {
            if let Err(error) = result {
                return Err(guard.writer.cleanup_error(error.into()).await);
            }
        }
        guard.disarm();
        Ok(())
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        if matches!(self.state, UploadState::Completed(_)) {
            return Ok(());
        }
        self.ensure_active()?;
        let guard = AbortOnCancel::new(self);
        if !guard.writer.ensure_active()?.buffer.is_empty()
            && let Err(error) = guard.writer.upload_buffer().await
        {
            return Err(guard.writer.cleanup_error(error).await);
        }
        guard.writer.flush().await?;
        match guard.writer.ensure_active()?.upload.complete().await {
            Ok(result) => guard.writer.state = UploadState::Completed(result),
            Err(error) => return Err(guard.writer.cleanup_error(error.into()).await),
        }
        guard.disarm();
        Ok(())
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests;
