// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::BytesMut;
use futures::TryStreamExt;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-bindgen"))]
use futures::channel::oneshot;
#[cfg(not(target_arch = "wasm32"))]
use futures::future::Either;
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
/// Multipart initiation and part uploads run on the caller's async runtime so concurrent writers
/// scale across the caller's worker threads. If a part upload or completion fails, the writer
/// aborts the multipart upload once (preserving the primary error if abort itself fails) and
/// rejects any subsequent writes or completion.
///
/// Dropping an uncompleted writer (or cancelling an in-flight `write_all`, `flush`, or `shutdown`
/// future) schedules an asynchronous abort on a process-lifetime executor (`smol` with a `Compat`
/// Tokio context on native targets) so cleanup survives both task cancellation and shutdown of the
/// caller's Tokio runtime. On native targets, `shutdown()` also tracks remote completion on that
/// executor so a ready publication result or failed cleanup remains observable across cancellation.
/// Process termination can still prevent cleanup. If the underlying [`ObjectStore`] holds pooled
/// keep-alive connections opened on a caller Tokio runtime that later goes idle (without shutting
/// down) and retries are disabled (`max_retries: 0`), an abort that reuses such a connection still
/// depends on that runtime remaining driven or retries opening a fresh connection.
pub struct ObjectStoreWrite {
    state: UploadState,
    buffer: BytesMut,
    put_result: Option<PutResult>,
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
    cleanup_runtime: Option<tokio::runtime::Handle>,
}

#[cfg(not(target_arch = "wasm32"))]
enum CompletionOutcome {
    Completed(PutResult),
    Failed {
        complete_err: object_store::Error,
        abort_result: object_store::Result<()>,
    },
    Aborted(object_store::Result<()>),
}

#[cfg(not(target_arch = "wasm32"))]
struct PendingCompletion {
    cancel_tx: Option<oneshot::Sender<()>>,
    result_rx: oneshot::Receiver<CompletionOutcome>,
}

enum UploadState {
    Active(Box<dyn MultipartUpload>),
    #[cfg(not(target_arch = "wasm32"))]
    Completing(PendingCompletion),
    Completed,
    Aborted,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;
const BUFFER_SIZE: usize = 128 * 1024 * 1024;

impl ObjectStoreWrite {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            state: UploadState::Active(upload),
            buffer: BytesMut::with_capacity(CHUNK_SIZE),
            put_result: None,
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
            cleanup_runtime: tokio::runtime::Handle::try_current().ok(),
        })
    }

    pub fn put_result(&self) -> Option<&PutResult> {
        self.put_result.as_ref()
    }

    /// Explicitly abort the multipart upload if it is still active or completing.
    ///
    /// Calling `abort` on an already-aborted or completed writer is a no-op. If a previous
    /// `shutdown()` was cancelled while remote completion or post-failure cleanup was in flight on
    /// a native target, `abort` signals cancellation and waits for the completion task to settle:
    /// it returns `Ok(())` only if abort succeeded, returns the cleanup error if abort failed, or
    /// returns an error (and populates [`Self::put_result`]) if remote completion had already
    /// published the object. On native targets, cleanup runs on the global `smol` executor wrapped
    /// in [`Compat`] so it outlives both caller task cancellation and shutdown of the caller's
    /// Tokio runtime.
    pub async fn abort(&mut self) -> io::Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if let UploadState::Completing(pending) = &mut self.state {
            if let Some(cancel_tx) = pending.cancel_tx.take() {
                let _ = cancel_tx.send(());
            }
            let outcome = (&mut pending.result_rx).await;
            return match outcome {
                Ok(CompletionOutcome::Aborted(res)) => {
                    self.state = UploadState::Aborted;
                    res.map_err(io::Error::from)
                }
                Ok(CompletionOutcome::Failed { abort_result, .. }) => {
                    self.state = UploadState::Aborted;
                    abort_result.map_err(io::Error::from)
                }
                Ok(CompletionOutcome::Completed(put_result)) => {
                    self.state = UploadState::Completed;
                    self.put_result = Some(put_result);
                    Err(io::Error::other(
                        "multipart upload already completed before abort",
                    ))
                }
                Err(_) => {
                    self.state = UploadState::Aborted;
                    Err(io::Error::other("completion task cancelled"))
                }
            };
        }

        if !matches!(self.state, UploadState::Active(_)) {
            return Ok(());
        }
        self.buffer.clear();
        let UploadState::Active(mut upload) =
            std::mem::replace(&mut self.state, UploadState::Aborted)
        else {
            unreachable!("checked active state above");
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = oneshot::channel();
            smol::spawn(Compat::new(async move {
                let res = upload.abort().await;
                drop(tx.send(res));
            }))
            .detach();
            match rx.await {
                Ok(res) => res.map_err(io::Error::from),
                Err(_) => Err(io::Error::other("abort task cancelled")),
            }
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm-bindgen"))]
        {
            let (tx, rx) = oneshot::channel();
            wasm_bindgen_futures::spawn_local(async move {
                let res = upload.abort().await;
                drop(tx.send(res));
            });
            match rx.await {
                Ok(res) => res.map_err(io::Error::from),
                Err(_) => Err(io::Error::other("abort task cancelled")),
            }
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
        {
            if let Some(handle) = self.take_live_tokio_handle() {
                let join_handle = handle.spawn(async move { upload.abort().await });
                return match join_handle.await {
                    Ok(res) => res.map_err(io::Error::from),
                    Err(err) => Err(io::Error::other(format!("abort task failed: {err}"))),
                };
            }
            upload.abort().await.map_err(io::Error::from)
        }
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
    fn take_live_tokio_handle(&mut self) -> Option<tokio::runtime::Handle> {
        if let Ok(handle) = tokio::runtime::Handle::try_current()
            && is_tokio_handle_alive(&handle)
        {
            self.cleanup_runtime = None;
            return Some(handle);
        }
        self.cleanup_runtime.take().filter(is_tokio_handle_alive)
    }

    /// Transition an active upload to `Aborted` and schedule detached remote cleanup on a
    /// process-lifetime executor. Used by `Drop` and by [`AbortOnCancel`] when an in-flight write,
    /// flush, or shutdown future is cancelled while the caller still owns the writer.
    fn abort_in_background(&mut self) {
        self.buffer.clear();
        #[cfg(not(target_arch = "wasm32"))]
        if let UploadState::Completing(pending) = &mut self.state {
            // Signal the in-flight completion task to cancel `complete()` and abort the upload,
            // while keeping `UploadState::Completing` so a subsequent `abort()` or `shutdown()`
            // still observes whether publication won or cleanup failed.
            if let Some(cancel_tx) = pending.cancel_tx.take() {
                let _ = cancel_tx.send(());
            }
            return;
        }
        let UploadState::Active(mut upload) =
            std::mem::replace(&mut self.state, UploadState::Aborted)
        else {
            return;
        };

        let abort_future = async move {
            if let Err(err) = upload.abort().await {
                tracing::warn!(%err, "failed to abort multipart upload in background");
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            smol::spawn(Compat::new(abort_future)).detach();
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm-bindgen"))]
        {
            wasm_bindgen_futures::spawn_local(abort_future);
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
        {
            if let Some(handle) = self.take_live_tokio_handle() {
                handle.spawn(abort_future);
                return;
            }
            drop(abort_future);
            tracing::warn!("no async runtime available to abort multipart upload on drop");
        }
    }

    fn active_upload_mut(&mut self) -> io::Result<&mut Box<dyn MultipartUpload>> {
        match &mut self.state {
            UploadState::Active(upload) => Ok(upload),
            #[cfg(not(target_arch = "wasm32"))]
            UploadState::Completing(_) => {
                Err(io::Error::other("multipart upload completion in progress"))
            }
            UploadState::Completed => Err(io::Error::other("multipart upload already completed")),
            UploadState::Aborted => Err(io::Error::other(
                "multipart upload already failed or aborted",
            )),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn finish_pending_completion(&mut self) -> io::Result<()> {
        let outcome = {
            let UploadState::Completing(pending) = &mut self.state else {
                unreachable!("checked completing state above");
            };
            (&mut pending.result_rx).await
        };
        match outcome {
            Ok(CompletionOutcome::Completed(put_result)) => {
                self.state = UploadState::Completed;
                self.put_result = Some(put_result);
                Ok(())
            }
            Ok(CompletionOutcome::Failed { complete_err, .. }) => {
                self.state = UploadState::Aborted;
                Err(complete_err.into())
            }
            Ok(CompletionOutcome::Aborted(_)) => {
                self.state = UploadState::Aborted;
                Err(io::Error::other(
                    "multipart upload already failed or aborted",
                ))
            }
            Err(_) => {
                self.state = UploadState::Aborted;
                Err(io::Error::other("completion task cancelled"))
            }
        }
    }

    /// Attempt to abort the active upload once after a primary error, logging and discarding any
    /// secondary abort error so the caller receives the primary failure.
    async fn abort_after_error(&mut self) {
        if let Err(abort_err) = self.abort().await {
            tracing::warn!(%abort_err, "failed to abort multipart upload after write error");
        }
    }
}

/// Checks whether a Tokio runtime handle can still execute spawned tasks.
///
/// On a shut-down runtime, `Handle::spawn` cancels the task immediately so `is_finished()` is
/// already true even for a future that never resolves.
#[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
fn is_tokio_handle_alive(handle: &tokio::runtime::Handle) -> bool {
    let probe = handle.spawn(std::future::pending::<()>());
    let alive = !probe.is_finished();
    probe.abort();
    alive
}

/// Drop guard that aborts an active upload if an in-flight `write_all`, `flush`, or `shutdown`
/// future is cancelled after buffered bytes have been removed or a remote call has started.
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
        let _ = self.active_upload_mut()?;
        self.buffer.extend_from_slice(buffer.as_slice());

        // If the buffer is full
        if self.buffer.len() > BUFFER_SIZE {
            let guard = AbortOnCancel::new(self);
            let part_result = {
                let parts = FuturesUnordered::new();
                let UploadState::Active(upload) = &mut guard.writer.state else {
                    unreachable!("checked active state above");
                };
                // Split off chunks while buffer is larger than CHUNK_SIZE
                while guard.writer.buffer.len() > CHUNK_SIZE {
                    let payload = guard.writer.buffer.split_to(CHUNK_SIZE).freeze();
                    parts.push(upload.put_part(PutPayload::from_bytes(payload)));
                }
                parts.try_collect::<Vec<_>>().await
            };

            if let Err(err) = part_result {
                guard.writer.abort_after_error().await;
                guard.disarm();
                return Err(err.into());
            }
            guard.disarm();
        }

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        let _ = self.active_upload_mut()?;
        let guard = AbortOnCancel::new(self);
        let part_result = {
            let parts = FuturesUnordered::new();
            let UploadState::Active(upload) = &mut guard.writer.state else {
                unreachable!("checked active state above");
            };
            // Split off chunks while buffer is larger than CHUNK_SIZE
            while guard.writer.buffer.len() > CHUNK_SIZE {
                let payload = guard.writer.buffer.split_to(CHUNK_SIZE).freeze();
                parts.push(upload.put_part(PutPayload::from_bytes(payload)));
            }
            parts.try_collect::<Vec<_>>().await
        };

        if let Err(err) = part_result {
            guard.writer.abort_after_error().await;
            guard.disarm();
            return Err(err.into());
        }
        guard.disarm();

        Ok(())
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        if matches!(self.state, UploadState::Completed) {
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(self.state, UploadState::Completing(_)) {
            let guard = AbortOnCancel::new(self);
            let res = guard.writer.finish_pending_completion().await;
            guard.disarm();
            return res;
        }
        let _ = self.active_upload_mut()?;
        let guard = AbortOnCancel::new(self);

        let part_result = {
            let parts = FuturesUnordered::new();
            let UploadState::Active(upload) = &mut guard.writer.state else {
                unreachable!("checked active state above");
            };
            while !guard.writer.buffer.is_empty() {
                let size = guard.writer.buffer.len().min(CHUNK_SIZE);
                let payload = guard.writer.buffer.split_to(size).freeze();
                parts.push(upload.put_part(PutPayload::from_bytes(payload)));
            }
            parts.try_collect::<Vec<_>>().await
        };

        if let Err(err) = part_result {
            guard.writer.abort_after_error().await;
            guard.disarm();
            return Err(err.into());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Track remote completion in `UploadState::Completing` so cancelling `shutdown()`
            // preserves either a ready `PutResult` or any post-failure cleanup error for a
            // subsequent `abort()` or `shutdown()` call.
            let (cancel_tx, mut cancel_rx) = oneshot::channel::<()>();
            let (result_tx, result_rx) = oneshot::channel();
            let UploadState::Active(mut upload) = std::mem::replace(
                &mut guard.writer.state,
                UploadState::Completing(PendingCompletion {
                    cancel_tx: Some(cancel_tx),
                    result_rx,
                }),
            ) else {
                guard.disarm();
                return Err(io::Error::other(
                    "multipart upload already failed or aborted",
                ));
            };
            smol::spawn(Compat::new(async move {
                let complete_outcome = if !matches!(cancel_rx.try_recv(), Ok(None)) {
                    None
                } else {
                    let complete_fut = upload.complete();
                    futures::pin_mut!(complete_fut);
                    futures::pin_mut!(cancel_rx);
                    match futures::future::select(complete_fut, cancel_rx).await {
                        Either::Left((res, _)) => Some(res),
                        Either::Right((..)) => None,
                    }
                };
                let outcome = match complete_outcome {
                    Some(Ok(put_result)) => CompletionOutcome::Completed(put_result),
                    Some(Err(complete_err)) => {
                        let abort_result = upload.abort().await;
                        if let Err(abort_err) = &abort_result {
                            tracing::warn!(
                                %abort_err,
                                "failed to abort multipart upload after completion error"
                            );
                        }
                        CompletionOutcome::Failed {
                            complete_err,
                            abort_result,
                        }
                    }
                    None => CompletionOutcome::Aborted(upload.abort().await),
                };
                drop(result_tx.send(outcome));
            }))
            .detach();

            let res = guard.writer.finish_pending_completion().await;
            guard.disarm();
            res
        }

        #[cfg(target_arch = "wasm32")]
        {
            let complete_result = guard.writer.active_upload_mut()?.complete().await;
            match complete_result {
                Ok(put_result) => {
                    // Release upload ownership without aborting now that the object is published.
                    guard.writer.state = UploadState::Completed;
                    guard.writer.put_result = Some(put_result);
                    guard.disarm();
                    Ok(())
                }
                Err(err) => {
                    guard.writer.abort_after_error().await;
                    guard.disarm();
                    Err(err.into())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
