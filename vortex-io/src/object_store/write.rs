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
/// schedules an asynchronous abort on the runtime captured at construction time so cleanup
/// survives cancellation of the file task. Process termination can still prevent cleanup.
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
        let cleanup_runtime = tokio::runtime::Handle::try_current().ok();
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
    /// Calling `abort` on an already-aborted or completed writer is a no-op.
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
        upload.abort().await.map_err(io::Error::from)
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

impl Drop for ObjectStoreWrite {
    fn drop(&mut self) {
        let UploadState::Active(mut upload) =
            std::mem::replace(&mut self.state, UploadState::Aborted)
        else {
            return;
        };

        let abort_future = async move {
            if let Err(err) = upload.abort().await {
                tracing::warn!(%err, "failed to abort multipart upload on drop");
            }
        };

        if let Some(handle) = self
            .cleanup_runtime
            .take()
            .or_else(|| tokio::runtime::Handle::try_current().ok())
        {
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
}

impl VortexWrite for ObjectStoreWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        let _ = self.active_upload_mut()?;
        self.buffer.extend_from_slice(buffer.as_slice());

        // If the buffer is full
        if self.buffer.len() > BUFFER_SIZE {
            let part_result = {
                let parts = FuturesUnordered::new();
                let UploadState::Active(upload) = &mut self.state else {
                    unreachable!("checked active state above");
                };
                // Split off chunks while buffer is larger than CHUNK_SIZE
                while self.buffer.len() > CHUNK_SIZE {
                    let payload = self.buffer.split_to(CHUNK_SIZE).freeze();
                    let part_fut = upload.put_part(PutPayload::from_bytes(payload));

                    parts.push(part_fut);
                }
                parts.try_collect::<Vec<_>>().await
            };

            if let Err(err) = part_result {
                self.abort_after_error().await;
                return Err(err.into());
            }
        }

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        let _ = self.active_upload_mut()?;
        let part_result = {
            let parts = FuturesUnordered::new();
            let UploadState::Active(upload) = &mut self.state else {
                unreachable!("checked active state above");
            };

            while self.buffer.len() > CHUNK_SIZE {
                let payload = self.buffer.split_to(CHUNK_SIZE).freeze();
                let part_fut = upload.put_part(PutPayload::from_bytes(payload));

                parts.push(part_fut);
            }

            parts.try_collect::<Vec<_>>().await
        };

        if let Err(err) = part_result {
            self.abort_after_error().await;
            return Err(err.into());
        }

        Ok(())
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        if matches!(self.state, UploadState::Completed) {
            return Ok(());
        }
        self.flush().await?;

        if !self.buffer.is_empty() {
            let payload = std::mem::take(&mut self.buffer).freeze();
            let part_result = self
                .active_upload_mut()?
                .put_part(PutPayload::from_bytes(payload))
                .await;
            if let Err(err) = part_result {
                self.abort_after_error().await;
                return Err(err.into());
            }
        }

        let complete_result = self.active_upload_mut()?.complete().await;
        match complete_result {
            Ok(put_result) => {
                // Release upload ownership without aborting now that the object is published.
                self.state = UploadState::Completed;
                self.put_result = Some(put_result);
                Ok(())
            }
            Err(err) => {
                self.abort_after_error().await;
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

        #[derive(Default)]
        struct UploadBehavior {
            fail_part_at: Option<usize>,
            fail_complete: bool,
            fail_abort: bool,
            gate_part_at: Option<(usize, Arc<Notify>, Arc<Notify>)>,
        }

        #[derive(Default)]
        struct UploadCounters {
            part_calls: AtomicUsize,
            complete_calls: AtomicUsize,
            abort_calls: AtomicUsize,
            abort_notify: Notify,
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
            fn put_part(&mut self, _data: PutPayload) -> UploadPart {
                let part_num = self.counters.part_calls.fetch_add(1, Ordering::SeqCst) + 1;
                let fail = self.behavior.fail_part_at == Some(part_num);
                let gate =
                    self.behavior.gate_part_at.as_ref().and_then(|(n, e, r)| {
                        (*n == part_num).then(|| (Arc::clone(e), Arc::clone(r)))
                    });
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
                Ok(PutResult {
                    e_tag: Some("etag-ok".to_string()),
                    version: None,
                })
            }

            async fn abort(&mut self) -> object_store::Result<()> {
                self.counters.abort_calls.fetch_add(1, Ordering::SeqCst);
                self.counters.abort_notify.notify_waiters();
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
            writer.write_all(vec![5u8; 256]).await?;
            writer.shutdown().await?;

            assert_eq!(
                writer.put_result().and_then(|r| r.e_tag.as_deref()),
                Some("etag-ok")
            );
            assert_eq!(counters.part_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 1);
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 0);

            drop(writer);
            tokio::task::yield_now().await;
            assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 0);
            Ok(())
        }
    }
}
