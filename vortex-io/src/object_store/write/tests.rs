// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::future::Future;
use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::task::Poll;
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
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::path::Path;
use parking_lot::Mutex;
use rstest::rstest;
use tempfile::tempdir;
use tokio::sync::Notify;
use tokio::time::timeout;

use super::*;
use crate::runtime::BlockingRuntime;
use crate::runtime::current::CurrentThreadRuntime;

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
    use super::*;

    #[derive(Default)]
    struct UploadCounters {
        part_calls: AtomicUsize,
        complete_calls: AtomicUsize,
        abort_calls: AtomicUsize,
        abort_notify: Notify,
        abort_finished: AtomicUsize,
        abort_finished_notify: Notify,
    }

    #[derive(Default)]
    struct UploadBehavior {
        fail_part_on_call: Option<usize>,
        fail_complete: bool,
        fail_abort: bool,
        gate_part: Option<(Arc<Notify>, Arc<Notify>)>,
        gate_complete: Option<(Arc<Notify>, Arc<Notify>)>,
        gate_abort: Option<(Arc<Notify>, Arc<Notify>)>,
        abort_uses_tokio_timer: bool,
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
            let call = self.counters.part_calls.fetch_add(1, Ordering::SeqCst) + 1;
            let behavior = Arc::clone(&self.behavior);
            Box::pin(async move {
                if let Some((entered, release)) = &behavior.gate_part {
                    entered.notify_one();
                    release.notified().await;
                }
                if behavior.fail_part_on_call == Some(call) {
                    return Err(object_store::Error::Generic {
                        store: "test",
                        source: format!("synthetic part failure on call {call}").into(),
                    });
                }
                Ok(())
            })
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            self.counters.complete_calls.fetch_add(1, Ordering::SeqCst);
            if let Some((entered, release)) = &self.behavior.gate_complete {
                entered.notify_one();
                release.notified().await;
            }
            if self.behavior.fail_complete {
                return Err(object_store::Error::Generic {
                    store: "test",
                    source: "synthetic complete failure".into(),
                });
            }
            Ok(PutResult {
                e_tag: Some("test-etag".to_owned()),
                version: None,
            })
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            if self.behavior.abort_uses_tokio_timer {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            self.counters.abort_calls.fetch_add(1, Ordering::SeqCst);
            self.counters.abort_notify.notify_one();
            if let Some((entered, release)) = &self.behavior.gate_abort {
                entered.notify_one();
                release.notified().await;
            }
            let res = if self.behavior.fail_abort {
                Err(object_store::Error::Generic {
                    store: "test",
                    source: "synthetic abort failure".into(),
                })
            } else {
                Ok(())
            };
            self.counters.abort_finished.fetch_add(1, Ordering::SeqCst);
            self.counters.abort_finished_notify.notify_one();
            res
        }
    }

    fn make_store(behavior: UploadBehavior) -> (Arc<dyn ObjectStore>, Arc<UploadCounters>) {
        let counters = Arc::new(UploadCounters::default());
        let store: Arc<dyn ObjectStore> = Arc::new(RecordingStore {
            behavior: Arc::new(behavior),
            counters: Arc::clone(&counters),
        });
        (store, counters)
    }

    async fn wait_for_abort(counters: &UploadCounters) {
        timeout(Duration::from_secs(2), async {
            loop {
                let mut notified = std::pin::pin!(counters.abort_notify.notified());
                notified.as_mut().enable();
                if counters.abort_calls.load(Ordering::SeqCst) > 0 {
                    break;
                }
                notified.await;
            }
        })
        .await
        .expect("timed out waiting for background multipart abort");
    }

    #[tokio::test]
    async fn test_part_failure_aborts_once_and_rejects_subsequent_calls() -> anyhow::Result<()> {
        let (store, counters) = make_store(UploadBehavior {
            fail_part_on_call: Some(1),
            fail_abort: true,
            ..Default::default()
        });

        let mut writer = ObjectStoreWrite::new(store, &Path::from("part-fail.vortex")).await?;
        writer.write_all(vec![1u8; CHUNK_SIZE + 1]).await?;

        let err = writer
            .flush()
            .await
            .expect_err("part upload failure must propagate");
        assert!(
            err.to_string().contains("synthetic part failure"),
            "primary part error must not be masked by abort failure: {err}"
        );
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);

        // Subsequent operations fail immediately without calling abort or complete again.
        assert!(writer.write_all(b"more".to_vec()).await.is_err());
        assert!(writer.flush().await.is_err());
        assert!(writer.shutdown().await.is_err());
        writer.abort().await?;
        drop(writer);

        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_complete_failure_aborts_once_and_preserves_primary_error() -> anyhow::Result<()> {
        let (store, counters) = make_store(UploadBehavior {
            fail_complete: true,
            fail_abort: true,
            ..Default::default()
        });

        let mut writer = ObjectStoreWrite::new(store, &Path::from("complete-fail.vortex")).await?;
        writer.write_all(b"payload".to_vec()).await?;

        let err = writer
            .shutdown()
            .await
            .expect_err("complete failure must propagate");
        assert!(
            err.to_string().contains("synthetic complete failure"),
            "primary complete error must not be masked by abort failure: {err}"
        );
        assert!(writer.put_result().is_none());
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);

        // Repeated abort and Drop must not trigger a second abort.
        writer.abort().await?;
        drop(writer);
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_explicit_abort_and_completed_drop_behavior() -> anyhow::Result<()> {
        let (store1, counters1) = make_store(UploadBehavior::default());
        let mut writer1 = ObjectStoreWrite::new(store1, &Path::from("abort.vortex")).await?;
        writer1.write_all(b"partial".to_vec()).await?;
        writer1.abort().await?;
        writer1.abort().await?;
        assert!(writer1.write_all(b"after".to_vec()).await.is_err());
        assert!(writer1.shutdown().await.is_err());
        drop(writer1);
        assert_eq!(counters1.abort_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters1.complete_calls.load(Ordering::SeqCst), 0);

        // Completed uploads must never be aborted on explicit abort or Drop.
        let (store2, counters2) = make_store(UploadBehavior::default());
        let mut writer2 = ObjectStoreWrite::new(store2, &Path::from("ok.vortex")).await?;
        writer2.write_all(b"complete".to_vec()).await?;
        writer2.shutdown().await?;
        assert!(writer2.put_result().is_some());
        writer2.abort().await?;
        drop(writer2);
        assert_eq!(counters2.complete_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters2.abort_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_drop_without_shutdown_aborts_multipart_upload() -> anyhow::Result<()> {
        let (store, counters) = make_store(UploadBehavior::default());
        {
            let mut writer = ObjectStoreWrite::new(store, &Path::from("dropped.vortex")).await?;
            writer.write_all(b"staged".to_vec()).await?;
        }
        wait_for_abort(&counters).await;
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_cancelled_flush_and_shutdown_abort_and_reject_reuse() -> anyhow::Result<()> {
        // Case 1: cancelling an in-flight `flush()` during `put_part`.
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let (store, counters) = make_store(UploadBehavior {
            gate_part: Some((Arc::clone(&entered), Arc::clone(&release))),
            ..Default::default()
        });
        let mut writer = ObjectStoreWrite::new(store, &Path::from("cancel-flush.vortex")).await?;
        writer.write_all(vec![1u8; CHUNK_SIZE + 1]).await?;
        {
            let mut flush_fut = Box::pin(writer.flush());
            let poll = poll_fn(|cx| Poll::Ready(flush_fut.as_mut().poll(cx))).await;
            assert!(poll.is_pending());
            timeout(Duration::from_secs(2), entered.notified()).await?;
        }
        wait_for_abort(&counters).await;
        assert!(writer.shutdown().await.is_err());
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 0);

        // Case 2: cancelling an in-flight `shutdown()` during `complete()`.
        let entered2 = Arc::new(Notify::new());
        let release2 = Arc::new(Notify::new());
        let (store2, counters2) = make_store(UploadBehavior {
            gate_complete: Some((Arc::clone(&entered2), Arc::clone(&release2))),
            ..Default::default()
        });
        let mut writer2 =
            ObjectStoreWrite::new(store2, &Path::from("cancel-shutdown.vortex")).await?;
        writer2.write_all(b"payload".to_vec()).await?;
        {
            let mut shutdown_fut = Box::pin(writer2.shutdown());
            let poll = poll_fn(|cx| Poll::Ready(shutdown_fut.as_mut().poll(cx))).await;
            assert!(poll.is_pending());
            timeout(Duration::from_secs(2), entered2.notified()).await?;
        }
        wait_for_abort(&counters2).await;
        assert!(writer2.shutdown().await.is_err());
        assert!(writer2.put_result().is_none());
        assert_eq!(counters2.abort_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum CleanupRuntimeSetup {
        StoppedTokioRuntime,
        IdleTokioRuntime,
        MultiThreadTokioRuntime,
        EnteredShutdownTokioHandle,
        EnteredIdleTokioHandle,
    }

    #[rstest]
    #[case(CleanupRuntimeSetup::StoppedTokioRuntime)]
    #[case(CleanupRuntimeSetup::IdleTokioRuntime)]
    #[case(CleanupRuntimeSetup::MultiThreadTokioRuntime)]
    #[case(CleanupRuntimeSetup::EnteredShutdownTokioHandle)]
    #[case(CleanupRuntimeSetup::EnteredIdleTokioHandle)]
    fn test_drop_and_abort_succeed_across_runtime_setups(
        #[case] setup: CleanupRuntimeSetup,
    ) -> anyhow::Result<()> {
        let (store_drop, counters_drop) = make_store(UploadBehavior {
            abort_uses_tokio_timer: true,
            ..Default::default()
        });
        let (store_abort, counters_abort) = make_store(UploadBehavior {
            abort_uses_tokio_timer: true,
            ..Default::default()
        });

        let tokio_rt = match setup {
            CleanupRuntimeSetup::MultiThreadTokioRuntime => {
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()?
            }
            _ => tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?,
        };
        let handle = tokio_rt.handle().clone();

        let (writer_drop, mut writer_abort) = tokio_rt.block_on(async {
            let mut w1 = ObjectStoreWrite::new(store_drop, &Path::from("rt-drop.vortex")).await?;
            w1.write_all(b"staged".to_vec()).await?;
            let mut w2 = ObjectStoreWrite::new(store_abort, &Path::from("rt-abort.vortex")).await?;
            w2.write_all(b"staged".to_vec()).await?;
            Ok::<_, anyhow::Error>((w1, w2))
        })?;

        let maybe_rt = match setup {
            CleanupRuntimeSetup::StoppedTokioRuntime
            | CleanupRuntimeSetup::EnteredShutdownTokioHandle => {
                drop(tokio_rt);
                None
            }
            CleanupRuntimeSetup::IdleTokioRuntime
            | CleanupRuntimeSetup::MultiThreadTokioRuntime
            | CleanupRuntimeSetup::EnteredIdleTokioHandle => Some(tokio_rt),
        };

        let _entered = match setup {
            CleanupRuntimeSetup::EnteredShutdownTokioHandle
            | CleanupRuntimeSetup::EnteredIdleTokioHandle => Some(handle.enter()),
            _ => None,
        };

        // Native cleanup runs on smol + Compat independently of the caller's Tokio runtime state.
        drop(writer_drop);
        CurrentThreadRuntime::new().block_on(async {
            writer_abort.abort().await?;
            Ok::<_, anyhow::Error>(())
        })?;
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while counters_drop.abort_calls.load(Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for background multipart abort"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(counters_drop.abort_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters_abort.abort_calls.load(Ordering::SeqCst), 1);
        drop(_entered);
        drop(maybe_rt);
        Ok(())
    }

    #[test]
    fn test_local_fs_staging_file_cleaned_up_on_drop_outside_tokio() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let store =
            Arc::new(LocalFileSystem::new_with_prefix(temp_dir.path())?) as Arc<dyn ObjectStore>;
        let path = Path::from("staged-cleanup.vortex");

        let tokio_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let writer = tokio_rt.block_on(async {
            let mut w = ObjectStoreWrite::new(Arc::clone(&store), &path).await?;
            w.write_all(vec![42u8; CHUNK_SIZE + 1]).await?;
            w.flush().await?;
            Ok::<_, anyhow::Error>(w)
        })?;

        // Drop the writer while the originating Tokio runtime stays idle.
        drop(writer);

        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            let entries: Vec<_> = std::fs::read_dir(temp_dir.path())?.collect::<Result<_, _>>()?;
            if entries.is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "staged local_fs file was not cleaned up on drop: {entries:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        drop(tokio_rt);
        Ok(())
    }

    #[tokio::test]
    async fn test_cancelled_shutdown_after_failed_complete_reports_abort_error_on_explicit_abort()
    -> anyhow::Result<()> {
        // Case 1: explicit `abort()` started while failed-complete cleanup is still in flight
        // must surface the cleanup error with only one abort attempt.
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let (store, counters) = make_store(UploadBehavior {
            fail_complete: true,
            fail_abort: true,
            gate_abort: Some((Arc::clone(&entered), Arc::clone(&release))),
            ..Default::default()
        });

        let mut writer =
            ObjectStoreWrite::new(store, &Path::from("failed-complete-abort.vortex")).await?;
        {
            let mut shutdown_fut = Box::pin(writer.shutdown());
            let first_poll = poll_fn(|cx| Poll::Ready(shutdown_fut.as_mut().poll(cx))).await;
            assert!(first_poll.is_pending());
            timeout(Duration::from_secs(2), entered.notified()).await?;
            drop(shutdown_fut);
        }

        let mut abort_fut = Box::pin(writer.abort());
        let abort_first_poll = poll_fn(|cx| Poll::Ready(abort_fut.as_mut().poll(cx))).await;
        assert!(abort_first_poll.is_pending());
        release.notify_one();

        let err = abort_fut
            .await
            .expect_err("explicit abort must surface the cleanup failure");
        assert!(
            err.to_string().contains("synthetic abort failure"),
            "expected abort error, got: {err}"
        );
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);

        // Case 2: backend rejects cleanup before explicit `abort()` is called; the writer must
        // retain that failure and report it from `abort()` without issuing a second abort.
        let entered_done = Arc::new(Notify::new());
        let release_done = Arc::new(Notify::new());
        let (store_done, counters_done) = make_store(UploadBehavior {
            fail_complete: true,
            fail_abort: true,
            gate_abort: Some((Arc::clone(&entered_done), Arc::clone(&release_done))),
            ..Default::default()
        });

        let mut writer_done =
            ObjectStoreWrite::new(store_done, &Path::from("failed-complete-settled.vortex"))
                .await?;
        {
            let mut shutdown_fut = Box::pin(writer_done.shutdown());
            let first_poll = poll_fn(|cx| Poll::Ready(shutdown_fut.as_mut().poll(cx))).await;
            assert!(first_poll.is_pending());
            timeout(Duration::from_secs(2), entered_done.notified()).await?;
            drop(shutdown_fut);
        }

        release_done.notify_one();
        timeout(Duration::from_secs(2), async {
            loop {
                let mut notified = std::pin::pin!(counters_done.abort_finished_notify.notified());
                notified.as_mut().enable();
                if counters_done.abort_finished.load(Ordering::SeqCst) > 0 {
                    break;
                }
                notified.await;
            }
        })
        .await?;

        let err_done = writer_done
            .abort()
            .await
            .expect_err("explicit abort must surface a cleanup failure that already settled");
        assert!(
            err_done.to_string().contains("synthetic abort failure"),
            "expected abort error, got: {err_done}"
        );
        assert_eq!(counters_done.abort_calls.load(Ordering::SeqCst), 1);

        // Case 3: a subsequent `shutdown()` on the same scenario reports the original completion
        // error with only one abort attempt.
        let entered2 = Arc::new(Notify::new());
        let release2 = Arc::new(Notify::new());
        let (store2, counters2) = make_store(UploadBehavior {
            fail_complete: true,
            fail_abort: true,
            gate_abort: Some((Arc::clone(&entered2), Arc::clone(&release2))),
            ..Default::default()
        });

        let mut writer2 =
            ObjectStoreWrite::new(store2, &Path::from("failed-complete-shutdown.vortex")).await?;
        {
            let mut shutdown_fut = Box::pin(writer2.shutdown());
            let first_poll = poll_fn(|cx| Poll::Ready(shutdown_fut.as_mut().poll(cx))).await;
            assert!(first_poll.is_pending());
            timeout(Duration::from_secs(2), entered2.notified()).await?;
            drop(shutdown_fut);
        }

        let mut shutdown_retry = Box::pin(writer2.shutdown());
        let retry_first_poll = poll_fn(|cx| Poll::Ready(shutdown_retry.as_mut().poll(cx))).await;
        assert!(retry_first_poll.is_pending());
        release2.notify_one();

        let err2 = shutdown_retry
            .await
            .expect_err("subsequent shutdown must surface the completion failure");
        assert!(
            err2.to_string().contains("synthetic complete failure"),
            "expected completion error, got: {err2}"
        );
        assert_eq!(counters2.abort_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_cancelled_shutdown_preserves_ready_complete_result_when_both_ready()
    -> anyhow::Result<()> {
        #[derive(Debug)]
        struct PostPublishGatedUpload {
            inner: Box<dyn MultipartUpload>,
            published: Arc<Notify>,
            deliver_rx: Mutex<Option<oneshot::Receiver<()>>>,
            abort_calls: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl MultipartUpload for PostPublishGatedUpload {
            fn put_part(&mut self, data: PutPayload) -> UploadPart {
                self.inner.put_part(data)
            }

            async fn complete(&mut self) -> object_store::Result<PutResult> {
                let res = self.inner.complete().await;
                self.published.notify_one();
                let rx = self.deliver_rx.lock().take();
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                res
            }

            async fn abort(&mut self) -> object_store::Result<()> {
                self.abort_calls.fetch_add(1, Ordering::SeqCst);
                self.inner.abort().await
            }
        }

        #[derive(Debug)]
        struct PostPublishGatedStore {
            inner: Arc<dyn ObjectStore>,
            published: Arc<Notify>,
            deliver_rx: Mutex<Option<oneshot::Receiver<()>>>,
            abort_calls: Arc<AtomicUsize>,
        }

        impl fmt::Display for PostPublishGatedStore {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "PostPublishGatedStore")
            }
        }

        #[async_trait]
        impl ObjectStore for PostPublishGatedStore {
            async fn put_opts(
                &self,
                location: &Path,
                payload: PutPayload,
                opts: PutOptions,
            ) -> object_store::Result<PutResult> {
                self.inner.put_opts(location, payload, opts).await
            }

            async fn put_multipart_opts(
                &self,
                location: &Path,
                opts: PutMultipartOptions,
            ) -> object_store::Result<Box<dyn MultipartUpload>> {
                let inner = self.inner.put_multipart_opts(location, opts).await?;
                Ok(Box::new(PostPublishGatedUpload {
                    inner,
                    published: Arc::clone(&self.published),
                    deliver_rx: Mutex::new(self.deliver_rx.lock().take()),
                    abort_calls: Arc::clone(&self.abort_calls),
                }))
            }

            async fn get_opts(
                &self,
                location: &Path,
                options: GetOptions,
            ) -> object_store::Result<GetResult> {
                self.inner.get_opts(location, options).await
            }

            fn delete_stream(
                &self,
                locations: BoxStream<'static, object_store::Result<Path>>,
            ) -> BoxStream<'static, object_store::Result<Path>> {
                self.inner.delete_stream(locations)
            }

            fn list(
                &self,
                prefix: Option<&Path>,
            ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
                self.inner.list(prefix)
            }

            async fn list_with_delimiter(
                &self,
                prefix: Option<&Path>,
            ) -> object_store::Result<ListResult> {
                self.inner.list_with_delimiter(prefix).await
            }

            async fn copy_opts(
                &self,
                from: &Path,
                to: &Path,
                options: object_store::CopyOptions,
            ) -> object_store::Result<()> {
                self.inner.copy_opts(from, to, options).await
            }
        }

        let dir = tempdir()?;
        let local: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
        let target = Path::from("published.vortex");
        let published = Arc::new(Notify::new());
        let (deliver_tx, deliver_rx) = oneshot::channel();
        let abort_calls = Arc::new(AtomicUsize::new(0));
        let store: Arc<dyn ObjectStore> = Arc::new(PostPublishGatedStore {
            inner: Arc::clone(&local),
            published: Arc::clone(&published),
            deliver_rx: Mutex::new(Some(deliver_rx)),
            abort_calls: Arc::clone(&abort_calls),
        });

        let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &target).await?;
        let mut shutdown_fut = Box::pin(writer.shutdown());
        let first_poll = poll_fn(|cx| Poll::Ready(shutdown_fut.as_mut().poll(cx))).await;
        assert!(first_poll.is_pending());

        // Wait until `LocalFileSystem::complete()` has published the file and parked before
        // returning `PutResult`.
        timeout(Duration::from_secs(2), published.notified()).await?;
        assert!(local.get(&target).await.is_ok());

        // Hold all global `smol` executor worker threads so both `complete_fut` and `cancel_rx`
        // become ready before the completion task is polled again.
        let worker_count = std::env::var("SMOL_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        let (entered_tx, entered_rx) = mpsc::channel();
        let mut release_txs = Vec::with_capacity(worker_count);
        let mut blockers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let entered_tx = entered_tx.clone();
            let (release_tx, release_rx) = mpsc::channel();
            release_txs.push(release_tx);
            blockers.push(smol::spawn(async move {
                let _ = entered_tx.send(());
                let _ = release_rx.recv();
            }));
        }
        for _ in 0..worker_count {
            entered_rx.recv_timeout(Duration::from_secs(2))?;
        }

        // Make the `PutResult` ready, then cancel `shutdown()` and start `abort()`.
        let _ = deliver_tx.send(());
        drop(shutdown_fut);

        let mut abort_fut = Box::pin(writer.abort());
        let abort_first_poll = poll_fn(|cx| Poll::Ready(abort_fut.as_mut().poll(cx))).await;
        assert!(abort_first_poll.is_pending());

        for release_tx in release_txs {
            let _ = release_tx.send(());
        }
        for blocker in blockers {
            blocker.await;
        }

        assert!(
            abort_fut.await.is_err(),
            "abort() must report that publication won when PutResult was already ready"
        );
        assert_eq!(abort_calls.load(Ordering::SeqCst), 0);
        assert!(
            writer
                .put_result()
                .and_then(|res| res.e_tag.as_deref())
                .is_some(),
            "retained PutResult must preserve the published ETag"
        );
        writer.shutdown().await?;
        Ok(())
    }
}
