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

enum UploadState {
    Active {
        upload: Box<dyn MultipartUpload>,
        buffer: BytesMut,
    },
    #[cfg(not(target_arch = "wasm32"))]
    Completing {
        cancel_tx: Option<oneshot::Sender<()>>,
        outcome_rx: oneshot::Receiver<CompletionOutcome>,
    },
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-bindgen"))]
    Aborting(oneshot::Receiver<object_store::Result<()>>),
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

    /// Explicitly abort the multipart upload if it is still active or completing.
    ///
    /// Calling `abort` on an already-aborted or completed writer is a no-op. If a previous
    /// `shutdown()` was cancelled while remote completion or post-failure cleanup was in flight on
    /// a native target, `abort` signals cancellation and waits for the completion task to settle:
    /// it returns `Ok(())` only if abort succeeded, returns the cleanup error if abort failed, or
    /// returns an error (and populates [`Self::put_result`]) if remote completion had already
    /// published the object.
    pub async fn abort(&mut self) -> io::Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(self.state, UploadState::Completing { .. }) {
            return match self.settle_completion(true).await? {
                CompletionOutcome::Aborted(res) => res.map_err(io::Error::from),
                CompletionOutcome::Failed { abort_result, .. } => {
                    abort_result.map_err(io::Error::from)
                }
                CompletionOutcome::Completed(_) => Err(io::Error::other(
                    "multipart upload already completed before abort",
                )),
            };
        }

        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-bindgen"))]
        {
            if matches!(self.state, UploadState::Active { .. }) {
                self.start_background_abort();
            }
            if let UploadState::Aborting(outcome_rx) = &mut self.state {
                let outcome = outcome_rx.await;
                self.state = UploadState::Aborted;
                return match outcome {
                    Ok(res) => res.map_err(io::Error::from),
                    Err(_) => Err(io::Error::other("abort task cancelled")),
                };
            }
            Ok(())
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
        {
            let Some(mut upload) = self.take_active_for_abort() else {
                return Ok(());
            };
            upload.abort().await.map_err(io::Error::from)
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-bindgen"))]
    fn start_background_abort(&mut self) {
        if !matches!(self.state, UploadState::Active { .. }) {
            return;
        }
        let (tx, rx) = oneshot::channel();
        let UploadState::Active { mut upload, .. } =
            std::mem::replace(&mut self.state, UploadState::Aborting(rx))
        else {
            unreachable!("checked active state above");
        };
        spawn_detached_cleanup(async move {
            let res = upload.abort().await;
            if let Err(Err(err)) = tx.send(res) {
                tracing::warn!(%err, "failed to abort multipart upload in background");
            }
        });
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
    fn take_active_for_abort(&mut self) -> Option<Box<dyn MultipartUpload>> {
        if !matches!(self.state, UploadState::Active { .. }) {
            return None;
        }
        match std::mem::replace(&mut self.state, UploadState::Aborted) {
            UploadState::Active { upload, .. } => Some(upload),
            _ => unreachable!("checked active state above"),
        }
    }

    /// Schedule detached cleanup when `Drop` or [`AbortOnCancel`] fires.
    fn abort_in_background(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let UploadState::Completing { cancel_tx, .. } = &mut self.state {
            // Keep `Completing` so a subsequent `abort()` or `shutdown()` still observes whether
            // publication won or cleanup failed.
            if let Some(tx) = cancel_tx.take() {
                let _ = tx.send(());
            }
            return;
        }
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-bindgen"))]
        {
            self.start_background_abort();
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
        {
            let Some(mut upload) = self.take_active_for_abort() else {
                return;
            };
            spawn_detached_cleanup(async move {
                if let Err(err) = upload.abort().await {
                    tracing::warn!(%err, "failed to abort multipart upload in background");
                }
            });
        }
    }

    fn ensure_active(&mut self) -> io::Result<(&mut Box<dyn MultipartUpload>, &mut BytesMut)> {
        match &mut self.state {
            UploadState::Active { upload, buffer } => Ok((upload, buffer)),
            #[cfg(not(target_arch = "wasm32"))]
            UploadState::Completing { .. } => {
                Err(io::Error::other("multipart upload completion in progress"))
            }
            UploadState::Completed(_) => {
                Err(io::Error::other("multipart upload already completed"))
            }
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-bindgen"))]
            UploadState::Aborting(_) => Err(io::Error::other(
                "multipart upload already failed or aborted",
            )),
            UploadState::Aborted => Err(io::Error::other(
                "multipart upload already failed or aborted",
            )),
        }
    }

    /// Flush buffered chunks of up to `CHUNK_SIZE` while more than `min_remaining` bytes remain.
    ///
    /// Arms [`AbortOnCancel`] once chunks are removed from the buffer and aborts once on failure,
    /// preserving the primary part error if abort itself fails.
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

        if let Err(err) = part_result {
            guard.writer.abort_after_error().await;
            guard.disarm();
            return Err(err.into());
        }
        guard.disarm();
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn settle_completion(&mut self, cancel: bool) -> io::Result<CompletionOutcome> {
        let outcome = {
            let UploadState::Completing {
                cancel_tx,
                outcome_rx,
            } = &mut self.state
            else {
                return Err(io::Error::other(
                    "multipart upload is not awaiting completion",
                ));
            };
            if cancel && let Some(tx) = cancel_tx.take() {
                let _ = tx.send(());
            }
            outcome_rx.await
        };

        match outcome {
            Ok(CompletionOutcome::Completed(put_result)) => {
                self.state = UploadState::Completed(put_result.clone());
                Ok(CompletionOutcome::Completed(put_result))
            }
            Ok(other) => {
                self.state = UploadState::Aborted;
                Ok(other)
            }
            Err(_) => {
                self.state = UploadState::Aborted;
                Err(io::Error::other("completion task cancelled"))
            }
        }
    }

    async fn abort_after_error(&mut self) {
        if let Err(abort_err) = self.abort().await {
            tracing::warn!(%abort_err, "failed to abort multipart upload after write error");
        }
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

/// Drop guard that aborts an active upload (or signals cancellation of a pending completion) if an
/// in-flight `write_all`, `flush`, or `shutdown` future is dropped before completing.
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

        #[cfg(not(target_arch = "wasm32"))]
        {
            if !matches!(self.state, UploadState::Completing { .. }) {
                self.flush_parts(0).await?;

                let (cancel_tx, mut cancel_rx) = oneshot::channel::<()>();
                let (outcome_tx, outcome_rx) = oneshot::channel();
                let UploadState::Active { mut upload, .. } = std::mem::replace(
                    &mut self.state,
                    UploadState::Completing {
                        cancel_tx: Some(cancel_tx),
                        outcome_rx,
                    },
                ) else {
                    return Err(io::Error::other(
                        "multipart upload already failed or aborted",
                    ));
                };

                spawn_detached_cleanup(async move {
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
                    drop(outcome_tx.send(outcome));
                });
            }

            let guard = AbortOnCancel::new(self);
            let outcome = guard.writer.settle_completion(false).await;
            guard.disarm();
            match outcome? {
                CompletionOutcome::Completed(_) => Ok(()),
                CompletionOutcome::Failed { complete_err, .. } => Err(complete_err.into()),
                CompletionOutcome::Aborted(_) => Err(io::Error::other(
                    "multipart upload already failed or aborted",
                )),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.flush_parts(0).await?;
            let guard = AbortOnCancel::new(self);
            let (upload, _) = guard.writer.ensure_active()?;
            match upload.complete().await {
                Ok(put_result) => {
                    guard.writer.state = UploadState::Completed(put_result);
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
