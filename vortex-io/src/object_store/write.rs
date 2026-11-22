// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::BytesMut;
use futures::TryStreamExt;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-bindgen"))]
use futures::channel::oneshot;
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

/// Adapter type to write data through a [`ObjectStore`] instance.
///
/// Each active instance exclusively owns its underlying [`MultipartUpload`]. After writing, the
/// caller must call [`VortexWrite::shutdown`] to complete and publish the object.
///
/// If a part upload or completion fails, the writer aborts the multipart upload once (preserving
/// the primary error if abort itself fails) and rejects any subsequent writes or completion. Note
/// that if `complete` fails due to a lost response after the remote store already committed the
/// object, the commit status is unknown; the writer still attempts a single best-effort abort and
/// returns the original completion error.
///
/// Dropping an uncompleted writer (for example on an upstream input error or task cancellation)
/// schedules an asynchronous abort on a live executor (the current Tokio runtime, a captured
/// multi-thread Tokio runtime, or the global `smol` executor) so cleanup survives cancellation of
/// the file task. Process termination can still prevent cleanup.
pub struct ObjectStoreWrite {
    state: UploadState,
    buffer: BytesMut,
    put_result: Option<PutResult>,
    cleanup_runtime: Option<tokio::runtime::Handle>,
}

enum UploadState {
    Active(Box<dyn MultipartUpload>),
    Completed,
    Aborted,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;
const BUFFER_SIZE: usize = 128 * 1024 * 1024;

impl ObjectStoreWrite {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        // Only capture a multi-thread Tokio runtime for off-runtime cleanup: it has its own worker
        // threads and can make progress independently, whereas a current-thread Tokio runtime may
        // be alive and idle with nothing polling it.
        let cleanup_runtime = tokio::runtime::Handle::try_current()
            .ok()
            .filter(|handle| handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread);
        Ok(Self {
            state: UploadState::Active(upload),
            buffer: BytesMut::with_capacity(CHUNK_SIZE),
            put_result: None,
            cleanup_runtime,
        })
    }

    pub fn put_result(&self) -> Option<&PutResult> {
        self.put_result.as_ref()
    }

    /// Explicitly abort the multipart upload if it is still active.
    ///
    /// Calling `abort` on an already-aborted or completed writer is a no-op. Cleanup is spawned on
    /// a live executor (the current Tokio runtime, a captured multi-thread Tokio runtime, or the
    /// global `smol` executor) so cancelling the caller while `abort` is in flight does not drop
    /// the upload before remote cleanup completes.
    pub async fn abort(&mut self) -> io::Result<()> {
        if !matches!(self.state, UploadState::Active(_)) {
            return Ok(());
        }
        self.buffer.clear();
        let UploadState::Active(mut upload) =
            std::mem::replace(&mut self.state, UploadState::Aborted)
        else {
            unreachable!("checked active state above");
        };

        if let Some(handle) = self.take_live_tokio_handle() {
            let join_handle = handle.spawn(async move { upload.abort().await });
            return match join_handle.await {
                Ok(res) => res.map_err(io::Error::from),
                Err(err) => Err(io::Error::other(format!("abort task failed: {err}"))),
            };
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = oneshot::channel();
            smol::spawn(async move {
                let res = upload.abort().await;
                drop(tx.send(res));
            })
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
            upload.abort().await.map_err(io::Error::from)
        }
    }

    fn take_live_tokio_handle(&mut self) -> Option<tokio::runtime::Handle> {
        if let Ok(handle) = tokio::runtime::Handle::try_current()
            && is_tokio_handle_alive(&handle)
        {
            self.cleanup_runtime = None;
            return Some(handle);
        }
        self.cleanup_runtime.take().filter(is_tokio_handle_alive)
    }

    /// Transition an active upload to `Aborted` and schedule detached remote cleanup on a live
    /// executor. Used by `Drop` and by [`AbortOnCancel`] when an in-flight write, flush, or
    /// shutdown future is cancelled while the caller still owns the writer.
    fn abort_in_background(&mut self) {
        self.buffer.clear();
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

        if let Some(handle) = self.take_live_tokio_handle() {
            handle.spawn(abort_future);
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            smol::spawn(abort_future).detach();
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm-bindgen"))]
        {
            wasm_bindgen_futures::spawn_local(abort_future);
        }

        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-bindgen")))]
        {
            drop(abort_future);
            tracing::warn!("no async runtime available to abort multipart upload on drop");
        }
    }

    fn active_upload_mut(&mut self) -> io::Result<&mut Box<dyn MultipartUpload>> {
        match &mut self.state {
            UploadState::Active(upload) => Ok(upload),
            UploadState::Completed => Err(io::Error::other("multipart upload already completed")),
            UploadState::Aborted => Err(io::Error::other(
                "multipart upload already failed or aborted",
            )),
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
                    let part_fut = upload.put_part(PutPayload::from_bytes(payload));

                    parts.push(part_fut);
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
        if self.buffer.len() <= CHUNK_SIZE {
            return Ok(());
        }

        let guard = AbortOnCancel::new(self);
        let part_result = {
            let parts = FuturesUnordered::new();
            let UploadState::Active(upload) = &mut guard.writer.state else {
                unreachable!("checked active state above");
            };

            while guard.writer.buffer.len() > CHUNK_SIZE {
                let payload = guard.writer.buffer.split_to(CHUNK_SIZE).freeze();
                let part_fut = upload.put_part(PutPayload::from_bytes(payload));

                parts.push(part_fut);
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
        self.flush().await?;

        let guard = AbortOnCancel::new(self);

        if !guard.writer.buffer.is_empty() {
            let payload = std::mem::take(&mut guard.writer.buffer).freeze();
            let part_result = guard
                .writer
                .active_upload_mut()?
                .put_part(PutPayload::from_bytes(payload))
                .await;
            if let Err(err) = part_result {
                guard.writer.abort_after_error().await;
                guard.disarm();
                return Err(err.into());
            }
        }

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use rstest::rstest;
    use tempfile::tempdir;

    use super::*;

    // Note: Concurrent writes test removed because &mut self in write_all already ensures
    // exclusive access. Multiple writers would need to be created with separate buffers,
    // which is not the intended use case.

    #[tokio::test]
    #[rstest]
    #[case(100)]
    #[case(8 * 1024 * 1024)]
    #[case(25 * 1024 * 1024)]
    #[case(26 * 1024 * 1024)]
    async fn test_object_store_writer_multiple_flushes(
        #[case] chunk_size: usize,
    ) -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let local_store =
            Arc::new(LocalFileSystem::new_with_prefix(temp_dir.path())?) as Arc<dyn ObjectStore>;
        let memory_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let location = Path::from("test.bin");

        for test_store in [memory_store, local_store] {
            let mut writer = ObjectStoreWrite::new(Arc::clone(&test_store), &location).await?;

            #[expect(clippy::cast_possible_truncation)]
            let data = (0..3)
                .map(|i| vec![i as u8; chunk_size])
                .collect::<Vec<_>>();

            // Write and flush multiple times
            for i in 0..3 {
                let data = data[i].clone();
                writer.write_all(data).await?;
                writer.flush().await?;
            }

            // Shutdown the writer to make sure data actually gets persisted.
            writer.shutdown().await?;

            // Verify all data was written
            let result = test_store.get(&location).await?;
            let bytes = result.bytes().await?;

            let expected_data = itertools::concat(data.into_iter());
            assert_eq!(bytes, expected_data);
        }

        Ok(())
    }

    #[allow(clippy::unimplemented)]
    mod multipart_lifecycle_tests {
        use std::fmt;
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering;
        use std::time::Duration;

        use async_trait::async_trait;
        use futures::stream::BoxStream;
        use object_store::GetOptions;
        use object_store::GetResult;
        use object_store::ListResult;
        use object_store::MultipartUpload;
        use object_store::ObjectMeta;
        use object_store::ObjectStore;
        use object_store::PutMultipartOptions;
        use object_store::PutOptions;
        use object_store::PutPayload;
        use object_store::PutResult;
        use object_store::UploadPart;
        use object_store::path::Path;
        use tokio::sync::Notify;
        use tokio::task::JoinSet;
        use tokio::time::timeout;

        use super::*;
        use crate::runtime::BlockingRuntime;
        use crate::runtime::current::CurrentThreadRuntime;

        #[derive(Default)]
        struct UploadBehavior {
            fail_part_at: Option<usize>,
            fail_complete: bool,
            fail_abort: bool,
            gate_part_at: Option<(usize, Arc<Notify>, Arc<Notify>)>,
            gate_abort: Option<(Arc<Notify>, Arc<Notify>)>,
        }

        #[derive(Default)]
        struct UploadCounters {
            part_calls: AtomicUsize,
            complete_calls: AtomicUsize,
            abort_calls: AtomicUsize,
            abort_completed: AtomicUsize,
            abort_in_tokio: AtomicUsize,
            abort_notify: Notify,
            abort_completed_notify: Notify,
            staged_bytes: parking_lot::Mutex<Vec<u8>>,
            published_bytes: parking_lot::Mutex<Option<Vec<u8>>>,
        }

        struct RecordingStore {
            behavior: Arc<UploadBehavior>,
            counters: Arc<UploadCounters>,
        }

        impl fmt::Debug for RecordingStore {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("RecordingStore").finish()
            }
        }

        impl fmt::Display for RecordingStore {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "RecordingStore")
            }
        }

        #[async_trait]
        impl ObjectStore for RecordingStore {
            async fn put_opts(
                &self,
                _location: &Path,
                _payload: PutPayload,
                _opts: PutOptions,
            ) -> object_store::Result<PutResult> {
                unimplemented!()
            }

            async fn put_multipart_opts(
                &self,
                _location: &Path,
                _opts: PutMultipartOptions,
            ) -> object_store::Result<Box<dyn MultipartUpload>> {
                Ok(Box::new(RecordingUpload {
                    behavior: Arc::clone(&self.behavior),
                    counters: Arc::clone(&self.counters),
                }))
            }

            async fn get_opts(
                &self,
                _location: &Path,
                _options: GetOptions,
            ) -> object_store::Result<GetResult> {
                unimplemented!()
            }

            fn delete_stream(
                &self,
                _locations: BoxStream<'static, object_store::Result<Path>>,
            ) -> BoxStream<'static, object_store::Result<Path>> {
                unimplemented!()
            }

            fn list(
                &self,
                _prefix: Option<&Path>,
            ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
                unimplemented!()
            }

            async fn list_with_delimiter(
                &self,
                _prefix: Option<&Path>,
            ) -> object_store::Result<ListResult> {
                unimplemented!()
            }

            async fn copy_opts(
                &self,
                _from: &Path,
                _to: &Path,
                _options: object_store::CopyOptions,
            ) -> object_store::Result<()> {
                unimplemented!()
            }
        }

        struct RecordingUpload {
            behavior: Arc<UploadBehavior>,
            counters: Arc<UploadCounters>,
        }

        impl fmt::Debug for RecordingUpload {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("RecordingUpload").finish()
            }
        }

        #[async_trait]
        impl MultipartUpload for RecordingUpload {
            fn put_part(&mut self, data: PutPayload) -> UploadPart {
                let part_num = self.counters.part_calls.fetch_add(1, Ordering::SeqCst) + 1;
                let fail = self.behavior.fail_part_at == Some(part_num);
                let gate =
                    self.behavior.gate_part_at.as_ref().and_then(|(n, e, r)| {
                        (*n == part_num).then(|| (Arc::clone(e), Arc::clone(r)))
                    });
                let counters = Arc::clone(&self.counters);
                Box::pin(async move {
                    if let Some((entered, release)) = gate {
                        entered.notify_one();
                        release.notified().await;
                    }
                    if fail {
                        return Err(object_store::Error::Generic {
                            store: "RecordingUpload",
                            source: io::Error::other("injected part failure").into(),
                        });
                    }
                    let mut staged = counters.staged_bytes.lock();
                    for chunk in data {
                        staged.extend_from_slice(&chunk);
                    }
                    Ok(())
                })
            }

            async fn complete(&mut self) -> object_store::Result<PutResult> {
                self.counters.complete_calls.fetch_add(1, Ordering::SeqCst);
                if self.behavior.fail_complete {
                    return Err(object_store::Error::Generic {
                        store: "RecordingUpload",
                        source: io::Error::other("injected complete failure").into(),
                    });
                }
                let published = self.counters.staged_bytes.lock().clone();
                *self.counters.published_bytes.lock() = Some(published);
                Ok(PutResult {
                    e_tag: Some("etag-ok".to_string()),
                    version: None,
                })
            }

            async fn abort(&mut self) -> object_store::Result<()> {
                self.counters.abort_calls.fetch_add(1, Ordering::SeqCst);
                if tokio::runtime::Handle::try_current().is_ok() {
                    self.counters.abort_in_tokio.fetch_add(1, Ordering::SeqCst);
                }
                self.counters.abort_notify.notify_waiters();
                if let Some((entered, release)) = &self.behavior.gate_abort {
                    entered.notify_one();
                    release.notified().await;
                }
                self.counters.abort_completed.fetch_add(1, Ordering::SeqCst);
                self.counters.abort_completed_notify.notify_waiters();
                if self.behavior.fail_abort {
                    return Err(object_store::Error::Generic {
                        store: "RecordingUpload",
                        source: io::Error::other("injected abort failure").into(),
                    });
                }
                Ok(())
            }
        }

        #[tokio::test]
        async fn test_part_failure_aborts_once_and_rejects_completion() -> anyhow::Result<()> {
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior {
                    fail_part_at: Some(1),
                    fail_abort: true,
                    ..Default::default()
                }),
                counters: Arc::clone(&counters),
            });

            let mut writer = ObjectStoreWrite::new(store, &Path::from("part-fail.vortex")).await?;
            writer.write_all(vec![1u8; CHUNK_SIZE + 1]).await?;
            let err = writer
                .flush()
                .await
                .expect_err("part failure must surface from flush");
            assert!(
                err.to_string().contains("injected part failure"),
                "primary part error must win over abort failure: {err}"
            );
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);

            let shutdown_err = writer
                .shutdown()
                .await
                .expect_err("shutdown after part failure must be rejected");
            assert!(
                !shutdown_err.to_string().is_empty(),
                "shutdown error must be non-empty"
            );
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);

            drop(writer);
            tokio::task::yield_now().await;
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            Ok(())
        }

        #[tokio::test]
        async fn test_completion_failure_aborts_once_and_preserves_primary_error()
        -> anyhow::Result<()> {
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior {
                    fail_complete: true,
                    fail_abort: true,
                    ..Default::default()
                }),
                counters: Arc::clone(&counters),
            });

            let mut writer =
                ObjectStoreWrite::new(store, &Path::from("complete-fail.vortex")).await?;
            writer.write_all(vec![7u8; 64]).await?;
            let err = writer
                .shutdown()
                .await
                .expect_err("completion failure must surface from shutdown");
            assert!(
                err.to_string().contains("injected complete failure"),
                "primary complete error must win over abort failure: {err}"
            );
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            assert!(writer.put_result().is_none());

            drop(writer);
            tokio::task::yield_now().await;
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            Ok(())
        }

        #[tokio::test]
        async fn test_drop_before_shutdown_aborts_once() -> anyhow::Result<()> {
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&counters),
            });

            let mut writer = ObjectStoreWrite::new(store, &Path::from("dropped.vortex")).await?;
            writer.write_all(vec![3u8; 128]).await?;
            drop(writer);

            timeout(Duration::from_secs(2), async {
                while counters.abort_calls.load(Ordering::SeqCst) == 0 {
                    counters.abort_notify.notified().await;
                }
            })
            .await
            .expect("drop must trigger async abort");

            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);
            Ok(())
        }

        #[tokio::test]
        async fn test_cancelled_task_during_gated_part_aborts_on_surviving_runtime()
        -> anyhow::Result<()> {
            let entered = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior {
                    gate_part_at: Some((1, Arc::clone(&entered), Arc::clone(&release))),
                    ..Default::default()
                }),
                counters: Arc::clone(&counters),
            });

            let mut tasks = JoinSet::new();
            tasks.spawn(async move {
                let mut writer =
                    ObjectStoreWrite::new(store, &Path::from("cancelled.vortex")).await?;
                writer.write_all(vec![9u8; CHUNK_SIZE + 1]).await?;
                writer.flush().await?;
                writer.shutdown().await?;
                anyhow::Ok(())
            });

            timeout(Duration::from_secs(2), entered.notified())
                .await
                .expect("gated part must start");

            // Dropping the JoinSet cancels the file task mid-part, simulating a sibling failure.
            drop(tasks);

            timeout(Duration::from_secs(2), async {
                while counters.abort_calls.load(Ordering::SeqCst) == 0 {
                    counters.abort_notify.notified().await;
                }
            })
            .await
            .expect("cancelled task must abort multipart upload on surviving runtime");

            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);
            Ok(())
        }

        #[tokio::test]
        async fn test_success_publishes_result_without_abort() -> anyhow::Result<()> {
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&counters),
            });

            let mut writer = ObjectStoreWrite::new(store, &Path::from("ok.vortex")).await?;
            let expected = vec![5u8; 256];
            writer.write_all(expected.clone()).await?;
            writer.shutdown().await?;

            assert_eq!(
                writer.put_result().and_then(|r| r.e_tag.as_deref()),
                Some("etag-ok")
            );
            assert_eq!(
                counters.published_bytes.lock().as_deref(),
                Some(expected.as_slice())
            );
            assert_eq!(counters.part_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 0);

            drop(writer);
            tokio::task::yield_now().await;
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 0);
            Ok(())
        }

        #[tokio::test]
        async fn test_cancelled_shutdown_aborts_and_rejects_retry_with_live_writer()
        -> anyhow::Result<()> {
            let entered = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior {
                    gate_part_at: Some((1, Arc::clone(&entered), Arc::clone(&release))),
                    ..Default::default()
                }),
                counters: Arc::clone(&counters),
            });

            let mut writer =
                ObjectStoreWrite::new(store, &Path::from("cancelled-shutdown.vortex")).await?;
            writer.write_all(vec![0xABu8; 64]).await?;

            {
                let shutdown_fut = writer.shutdown();
                tokio::pin!(shutdown_fut);
                tokio::select! {
                    res = &mut shutdown_fut => panic!("gated shutdown should not complete: {res:?}"),
                    () = entered.notified() => {}
                }
                // Dropping `shutdown_fut` cancels the pending shutdown while `writer` stays alive.
            }

            release.notify_one();

            // Retrying shutdown on the live writer must fail rather than publishing a 0-byte file.
            assert!(
                writer.shutdown().await.is_err(),
                "retrying shutdown after cancellation must fail"
            );
            assert!(counters.published_bytes.lock().is_none());
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);

            timeout(Duration::from_secs(2), async {
                while counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    counters.abort_completed_notify.notified().await;
                }
            })
            .await
            .expect("cancelled shutdown must abort the upload");

            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            Ok(())
        }

        #[tokio::test]
        async fn test_cancelled_flush_aborts_and_rejects_shutdown_with_live_writer()
        -> anyhow::Result<()> {
            let entered = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior {
                    gate_part_at: Some((1, Arc::clone(&entered), Arc::clone(&release))),
                    ..Default::default()
                }),
                counters: Arc::clone(&counters),
            });

            let mut writer =
                ObjectStoreWrite::new(store, &Path::from("cancelled-flush.vortex")).await?;
            writer.write_all(vec![0xCDu8; CHUNK_SIZE + 1]).await?;

            {
                let flush_fut = writer.flush();
                tokio::pin!(flush_fut);
                tokio::select! {
                    res = &mut flush_fut => panic!("gated flush should not complete: {res:?}"),
                    () = entered.notified() => {}
                }
                // Dropping `flush_fut` cancels the pending flush while `writer` stays alive.
            }

            release.notify_one();

            // Calling shutdown on the live writer must fail rather than publishing corrupt data.
            assert!(
                writer.shutdown().await.is_err(),
                "shutdown after cancelled flush must fail"
            );
            assert!(counters.published_bytes.lock().is_none());
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);

            timeout(Duration::from_secs(2), async {
                while counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    counters.abort_completed_notify.notified().await;
                }
            })
            .await
            .expect("cancelled flush must abort the upload");

            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            Ok(())
        }

        #[test]
        fn test_cancelled_abort_on_current_thread_runtime_completes_cleanup() -> anyhow::Result<()>
        {
            let runtime = CurrentThreadRuntime::new();
            let entered = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior {
                    gate_abort: Some((Arc::clone(&entered), Arc::clone(&release))),
                    ..Default::default()
                }),
                counters: Arc::clone(&counters),
            });

            runtime.block_on(async move {
                let mut writer =
                    ObjectStoreWrite::new(store, &Path::from("current-thread-abort.vortex"))
                        .await?;
                writer.write_all(vec![1u8; 64]).await?;

                {
                    let abort_fut = writer.abort();
                    futures::pin_mut!(abort_fut);
                    let entered_fut = entered.notified();
                    futures::pin_mut!(entered_fut);
                    match futures::future::select(abort_fut, entered_fut).await {
                        futures::future::Either::Left((res, _)) => {
                            panic!("gated abort should not complete: {res:?}");
                        }
                        futures::future::Either::Right(((), _)) => {}
                    }
                    // Dropping `abort_fut` cancels explicit `abort()` on CurrentThreadRuntime.
                }

                // Retry abort and drop writer; detached cleanup must still finish once released.
                writer.abort().await?;
                drop(writer);
                release.notify_one();

                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "detached abort on CurrentThreadRuntime did not complete"
                    );
                    smol::Timer::after(Duration::from_millis(5)).await;
                }

                assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(counters.abort_completed.load(Ordering::SeqCst), 1);
                anyhow::Ok(())
            })
        }

        #[test]
        fn test_stopped_tokio_runtime_falls_back_to_live_smol_executor() -> anyhow::Result<()> {
            let abort_counters = Arc::new(UploadCounters::default());
            let abort_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&abort_counters),
            });

            let drop_counters = Arc::new(UploadCounters::default());
            let drop_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&drop_counters),
            });

            // Create both writers on a Tokio runtime, then shut that runtime down.
            let (mut writer_for_abort, writer_for_drop) = {
                let tokio_rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                tokio_rt.block_on(async move {
                    let mut w1 =
                        ObjectStoreWrite::new(abort_store, &Path::from("stopped-rt-abort.vortex"))
                            .await?;
                    w1.write_all(vec![2u8; 64]).await?;
                    let mut w2 =
                        ObjectStoreWrite::new(drop_store, &Path::from("stopped-rt-drop.vortex"))
                            .await?;
                    w2.write_all(vec![3u8; 64]).await?;
                    anyhow::Ok((w1, w2))
                })?
            };

            // Drive both explicit abort and drop on Vortex's CurrentThreadRuntime (smol).
            let current_rt = CurrentThreadRuntime::new();
            current_rt.block_on(async move {
                writer_for_abort.abort().await?;
                assert_eq!(abort_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(abort_counters.abort_completed.load(Ordering::SeqCst), 1);

                drop(writer_for_drop);
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while drop_counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "drop cleanup from stopped Tokio runtime did not complete on smol"
                    );
                    smol::Timer::after(Duration::from_millis(5)).await;
                }
                assert_eq!(drop_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(drop_counters.abort_completed.load(Ordering::SeqCst), 1);
                anyhow::Ok(())
            })
        }

        #[test]
        fn test_idle_tokio_runtime_aborts_independently_on_smol() -> anyhow::Result<()> {
            let abort_counters = Arc::new(UploadCounters::default());
            let abort_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&abort_counters),
            });

            let drop_counters = Arc::new(UploadCounters::default());
            let drop_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&drop_counters),
            });

            // Create both writers on a Tokio current-thread runtime and keep that runtime alive
            // but idle throughout the rest of the test.
            let tokio_rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let (mut writer_for_abort, writer_for_drop) = tokio_rt.block_on(async move {
                let mut w1 =
                    ObjectStoreWrite::new(abort_store, &Path::from("idle-rt-abort.vortex")).await?;
                w1.write_all(vec![4u8; 64]).await?;
                let mut w2 =
                    ObjectStoreWrite::new(drop_store, &Path::from("idle-rt-drop.vortex")).await?;
                w2.write_all(vec![5u8; 64]).await?;
                anyhow::Ok((w1, w2))
            })?;

            // Drive both explicit abort and drop on Vortex's CurrentThreadRuntime while `tokio_rt`
            // stays alive and idle (never polled again).
            let current_rt = CurrentThreadRuntime::new();
            current_rt.block_on(async move {
                writer_for_abort.abort().await?;
                assert_eq!(abort_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(abort_counters.abort_completed.load(Ordering::SeqCst), 1);

                drop(writer_for_drop);
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while drop_counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "drop cleanup with idle Tokio runtime did not complete on smol"
                    );
                    smol::Timer::after(Duration::from_millis(5)).await;
                }
                assert_eq!(drop_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(drop_counters.abort_completed.load(Ordering::SeqCst), 1);
                anyhow::Ok(())
            })?;

            // Keep `tokio_rt` alive until after all assertions pass.
            drop(tokio_rt);
            Ok(())
        }

        #[test]
        fn test_multithread_tokio_runtime_preserves_tokio_context_for_off_runtime_cleanup()
        -> anyhow::Result<()> {
            let abort_counters = Arc::new(UploadCounters::default());
            let abort_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&abort_counters),
            });

            let drop_counters = Arc::new(UploadCounters::default());
            let drop_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&drop_counters),
            });

            let tokio_rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()?;
            let (mut writer_for_abort, writer_for_drop) = tokio_rt.block_on(async move {
                let mut w1 =
                    ObjectStoreWrite::new(abort_store, &Path::from("mt-abort.vortex")).await?;
                w1.write_all(vec![6u8; 64]).await?;
                let mut w2 =
                    ObjectStoreWrite::new(drop_store, &Path::from("mt-drop.vortex")).await?;
                w2.write_all(vec![7u8; 64]).await?;
                anyhow::Ok((w1, w2))
            })?;

            // Drive abort and drop outside Tokio while the multi-thread Tokio runtime stays live;
            // cleanup should run on that Tokio runtime so Tokio-dependent object stores keep their
            // reactor context.
            let current_rt = CurrentThreadRuntime::new();
            current_rt.block_on(async move {
                writer_for_abort.abort().await?;
                assert_eq!(abort_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(abort_counters.abort_completed.load(Ordering::SeqCst), 1);
                assert_eq!(abort_counters.abort_in_tokio.load(Ordering::SeqCst), 1);

                drop(writer_for_drop);
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while drop_counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "drop cleanup did not complete on captured multi-thread Tokio runtime"
                    );
                    smol::Timer::after(Duration::from_millis(5)).await;
                }
                assert_eq!(drop_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(drop_counters.abort_completed.load(Ordering::SeqCst), 1);
                assert_eq!(drop_counters.abort_in_tokio.load(Ordering::SeqCst), 1);
                anyhow::Ok(())
            })?;

            drop(tokio_rt);
            Ok(())
        }

        #[test]
        fn test_entered_shutdown_tokio_handle_falls_back_to_smol() -> anyhow::Result<()> {
            let abort_counters = Arc::new(UploadCounters::default());
            let abort_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&abort_counters),
            });

            let drop_counters = Arc::new(UploadCounters::default());
            let drop_store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior::default()),
                counters: Arc::clone(&drop_counters),
            });

            let tokio_rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()?;
            let stale_handle = tokio_rt.handle().clone();
            let (mut writer_for_abort, writer_for_drop) = tokio_rt.block_on(async move {
                let mut w1 =
                    ObjectStoreWrite::new(abort_store, &Path::from("stale-handle-abort.vortex"))
                        .await?;
                w1.write_all(vec![8u8; 64]).await?;
                let mut w2 =
                    ObjectStoreWrite::new(drop_store, &Path::from("stale-handle-drop.vortex"))
                        .await?;
                w2.write_all(vec![9u8; 64]).await?;
                anyhow::Ok((w1, w2))
            })?;
            drop(tokio_rt);

            // Enter the shut-down Tokio handle so Handle::try_current() succeeds while spawn()
            // would immediately cancel any task; both abort() and Drop must fall back to smol.
            let _guard = stale_handle.enter();
            let current_rt = CurrentThreadRuntime::new();
            current_rt.block_on(async move {
                writer_for_abort.abort().await?;
                assert_eq!(abort_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(abort_counters.abort_completed.load(Ordering::SeqCst), 1);

                drop(writer_for_drop);
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while drop_counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "drop cleanup under entered shut-down Tokio handle did not complete on smol"
                    );
                    smol::Timer::after(Duration::from_millis(5)).await;
                }
                assert_eq!(drop_counters.abort_calls.load(Ordering::SeqCst), 1);
                assert_eq!(drop_counters.abort_completed.load(Ordering::SeqCst), 1);
                anyhow::Ok(())
            })
        }

        #[tokio::test]
        async fn test_cancelled_abort_completes_on_surviving_runtime() -> anyhow::Result<()> {
            let entered = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let counters = Arc::new(UploadCounters::default());
            let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
                behavior: Arc::new(UploadBehavior {
                    gate_abort: Some((Arc::clone(&entered), Arc::clone(&release))),
                    ..Default::default()
                }),
                counters: Arc::clone(&counters),
            });

            let mut tasks = JoinSet::new();
            tasks.spawn(async move {
                let mut writer =
                    ObjectStoreWrite::new(store, &Path::from("cancelled-abort.vortex")).await?;
                writer.write_all(vec![1u8; 64]).await?;
                writer.abort().await?;
                anyhow::Ok(())
            });

            timeout(Duration::from_secs(2), entered.notified())
                .await
                .expect("gated abort must start");

            // Cancel the caller while `abort()` is awaiting the remote store.
            drop(tasks);
            release.notify_one();

            timeout(Duration::from_secs(2), async {
                while counters.abort_completed.load(Ordering::SeqCst) == 0 {
                    counters.abort_completed_notify.notified().await;
                }
            })
            .await
            .expect("spawned abort must still finish on surviving runtime after cancellation");

            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.abort_completed.load(Ordering::SeqCst), 1);
            Ok(())
        }
    }
}
