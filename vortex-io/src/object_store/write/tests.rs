// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::future::Future;
use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
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
use object_store::path::Path;
use tokio::sync::Notify;
use tokio::time::timeout;

use super::*;

#[allow(clippy::unimplemented)]
mod multipart_lifecycle_tests {
    use super::*;

    #[derive(Default)]
    struct UploadCounters {
        part_calls: AtomicUsize,
        complete_calls: AtomicUsize,
        abort_calls: AtomicUsize,
        abort_notify: Notify,
    }

    #[derive(Default)]
    struct UploadBehavior {
        fail_part_on_call: Option<usize>,
        fail_complete: bool,
        fail_abort: bool,
        gate_part: Option<(Arc<Notify>, Arc<Notify>)>,
        gate_complete: Option<(Arc<Notify>, Arc<Notify>)>,
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
                extensions: Default::default(),
            })
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            if self.behavior.abort_uses_tokio_timer {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            self.counters.abort_calls.fetch_add(1, Ordering::SeqCst);
            self.counters.abort_notify.notify_one();
            if self.behavior.fail_abort {
                return Err(object_store::Error::Generic {
                    store: "test",
                    source: "synthetic abort failure".into(),
                });
            }
            Ok(())
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
        wait_for_abort(&counters).await;
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);

        // Subsequent operations fail immediately without calling abort or complete again.
        assert!(writer.write_all(b"more".to_vec()).await.is_err());
        assert!(writer.flush().await.is_err());
        assert!(writer.shutdown().await.is_err());
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
        wait_for_abort(&counters).await;
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);

        // Dropping the failed writer must not trigger a second abort.
        drop(writer);
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_completed_drop_does_not_abort() -> anyhow::Result<()> {
        let (store, counters) = make_store(UploadBehavior::default());
        let mut writer = ObjectStoreWrite::new(store, &Path::from("ok.vortex")).await?;
        writer.write_all(b"complete".to_vec()).await?;
        writer.shutdown().await?;
        assert!(writer.put_result().is_some());
        drop(writer);
        assert_eq!(counters.complete_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 0);
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
        // Cancellation rejects reuse immediately, but cleanup must wait for submitted work.
        assert!(writer.shutdown().await.is_err());
        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 0);
        release.notify_one();
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

    #[test]
    fn test_drop_aborts_after_caller_tokio_runtime_stops() -> anyhow::Result<()> {
        let (store, counters) = make_store(UploadBehavior {
            abort_uses_tokio_timer: true,
            ..Default::default()
        });

        let tokio_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let writer = tokio_rt.block_on(async {
            let mut w = ObjectStoreWrite::new(store, &Path::from("rt-drop.vortex")).await?;
            w.write_all(b"staged".to_vec()).await?;
            Ok::<_, anyhow::Error>(w)
        })?;

        drop(tokio_rt);

        // Native cleanup runs on smol + Compat even after the caller's Tokio runtime stops.
        drop(writer);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while counters.abort_calls.load(Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for background multipart abort"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(counters.abort_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
mod bounded_tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use parking_lot::Mutex;
    use rstest::rstest;
    use tempfile::TempDir;
    use tempfile::tempdir;
    use tokio::sync::Semaphore;

    use super::*;

    #[tokio::test]
    #[rstest]
    #[case(0)]
    #[case(CHUNK_SIZE / 2)]
    #[case(CHUNK_SIZE)]
    #[case(CHUNK_SIZE + CHUNK_SIZE / 2)]
    async fn test_object_store_writer_multiple_flushes(
        #[case] chunk_size: usize,
    ) -> VortexResult<()> {
        let temp_dir = tempdir()?;
        let local_store =
            Arc::new(LocalFileSystem::new_with_prefix(temp_dir.path())?) as Arc<dyn ObjectStore>;
        let memory_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let location = Path::from("test.bin");

        for test_store in [memory_store, local_store] {
            let mut writer = ObjectStoreWrite::new(Arc::clone(&test_store), &location).await?;
            let data = (0u8..3).map(|i| vec![i; chunk_size]).collect::<Vec<_>>();
            for data in &data {
                let buffer = data.clone();
                let pointer = buffer.as_ptr();
                let returned = writer.write_all(buffer).await?;
                assert_eq!(returned.as_ptr(), pointer);
                assert_eq!(&returned, data);
                writer.flush().await?;
                assert!(test_store.head(&location).await.is_err());
                assert!(writer.put_result().is_none());
            }

            writer.shutdown().await?;
            writer.shutdown().await?;
            assert!(writer.put_result().is_some());
            assert!(writer.write_all(vec![1]).await.is_err());
            assert!(writer.abort().await.is_err());
            let bytes = test_store.get(&location).await?.bytes().await?;
            assert_eq!(bytes, itertools::concat(data.into_iter()));
        }
        Ok(())
    }

    #[derive(Debug)]
    struct Probe {
        // Retain the real handle beyond the writer's lifetime: local Drop cleanup must not hide
        // a missing explicit abort, as cloud stores cannot rely on Drop for cleanup.
        upload: tokio::sync::Mutex<Box<dyn MultipartUpload>>,
        sizes: Mutex<Vec<usize>>,
        settled: AtomicUsize,
        aborts: AtomicUsize,
        completes: AtomicUsize,
        permits: Semaphore,
        fail_part: bool,
        fail_complete: bool,
        fail_abort: AtomicBool,
    }

    #[derive(Debug)]
    struct ProbedUpload(Arc<Probe>);

    fn upload_error(message: &'static str) -> object_store::Error {
        object_store::Error::Generic {
            store: "test",
            source: message.into(),
        }
    }

    #[async_trait::async_trait]
    impl MultipartUpload for ProbedUpload {
        fn put_part(&mut self, payload: PutPayload) -> UploadPart {
            self.0.sizes.lock().push(payload.content_length());
            let part = self.0.upload.try_lock().unwrap().put_part(payload);
            let probe = Arc::clone(&self.0);
            async move {
                probe.permits.acquire().await.unwrap().forget();
                let result = part.await;
                probe.settled.fetch_add(1, Ordering::SeqCst);
                if probe.fail_part {
                    return Err(upload_error("injected part failure"));
                }
                result
            }
            .boxed()
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            assert_eq!(
                self.0.settled.load(Ordering::SeqCst),
                self.0.sizes.lock().len()
            );
            self.0.completes.fetch_add(1, Ordering::SeqCst);
            if self.0.fail_complete {
                return Err(upload_error("injected complete failure"));
            }
            self.0.upload.lock().await.complete().await
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            assert_eq!(
                self.0.settled.load(Ordering::SeqCst),
                self.0.sizes.lock().len()
            );
            self.0.aborts.fetch_add(1, Ordering::SeqCst);
            if self.0.fail_abort.load(Ordering::SeqCst) {
                return Err(upload_error("injected abort failure"));
            }
            self.0.upload.lock().await.abort().await
        }
    }

    async fn probed_writer(
        directory: &TempDir,
        fail_part: bool,
        fail_complete: bool,
        fail_abort: bool,
    ) -> VortexResult<(ObjectStoreWrite, Arc<Probe>, Arc<dyn ObjectStore>)> {
        let store =
            Arc::new(LocalFileSystem::new_with_prefix(directory.path())?) as Arc<dyn ObjectStore>;
        let mut writer = ObjectStoreWrite::new(Arc::clone(&store), &Path::from("test.bin")).await?;
        let UploadState::Active(mut active) =
            std::mem::replace(&mut writer.state, UploadState::Aborted)
        else {
            panic!("new writer must be active");
        };
        let probe = Arc::new(Probe {
            upload: tokio::sync::Mutex::new(active.upload),
            sizes: Mutex::new(Vec::new()),
            settled: AtomicUsize::new(0),
            aborts: AtomicUsize::new(0),
            completes: AtomicUsize::new(0),
            permits: Semaphore::new(0),
            fail_part,
            fail_complete,
            fail_abort: AtomicBool::new(fail_abort),
        });
        active.upload = Box::new(ProbedUpload(Arc::clone(&probe)));
        writer.state = UploadState::Active(active);
        Ok((writer, probe, store))
    }

    #[rstest]
    #[case::exact_parts(0)]
    #[case::partial_tail(31)]
    #[tokio::test]
    async fn test_oversized_write_backpressure_and_order(
        #[case] tail: usize,
        #[values(1, 4)] concurrency: usize,
    ) -> VortexResult<()> {
        let directory = tempdir()?;
        let (writer, probe, store) = probed_writer(&directory, false, false, false).await?;
        let mut writer = writer.with_concurrency(concurrency);
        writer.write_all(*b"prefix!").await?;
        writer.flush().await?;
        assert!(probe.sizes.lock().is_empty());

        let data = (0u8..=250)
            .cycle()
            .take((concurrency + 1) * CHUNK_SIZE + tail - 7)
            .collect::<Vec<_>>();
        let expected = [b"prefix!".as_slice(), data.as_slice()].concat();
        let pointer = data.as_ptr();
        {
            let write = writer.write_all(data);
            tokio::pin!(write);
            assert!(futures::poll!(&mut write).is_pending());
            assert_eq!(*probe.sizes.lock(), vec![CHUNK_SIZE; concurrency]);
            assert_eq!(probe.completes.load(Ordering::SeqCst), 0);
            probe.permits.add_permits(concurrency + 2);
            let returned = write.await?;
            assert_eq!(returned.as_ptr(), pointer);
        }
        writer.flush().await?;
        assert_eq!(*probe.sizes.lock(), vec![CHUNK_SIZE; concurrency + 1]);
        assert!(store.head(&Path::from("test.bin")).await.is_err());
        writer.shutdown().await?;
        let mut sizes = vec![CHUNK_SIZE; concurrency + 1];
        if tail != 0 {
            sizes.push(tail);
        }
        assert_eq!(*probe.sizes.lock(), sizes);
        let bytes = store.get(&Path::from("test.bin")).await?.bytes().await?;
        assert_eq!(bytes, expected);
        Ok(())
    }

    #[rstest]
    #[case::default(None)]
    #[case::explicit_zero(Some(0))]
    #[tokio::test]
    async fn test_unlimited_concurrency(#[case] concurrency: Option<usize>) -> VortexResult<()> {
        let directory = tempdir()?;
        let (mut writer, probe, store) = probed_writer(&directory, false, false, false).await?;
        if let Some(concurrency) = concurrency {
            writer = writer.with_concurrency(concurrency);
        }
        let data = (0u8..=250)
            .cycle()
            .take(9 * CHUNK_SIZE + 31)
            .collect::<Vec<_>>();
        let pointer = data.as_ptr();
        // No part can finish, but unlimited writes must not wait for upload capacity.
        let returned = writer.write_all(data).now_or_never().unwrap()?;
        assert_eq!(returned.as_ptr(), pointer);
        assert_eq!(*probe.sizes.lock(), vec![CHUNK_SIZE; 9]);
        {
            let flush = writer.flush();
            tokio::pin!(flush);
            assert!(futures::poll!(&mut flush).is_pending());
            probe.permits.add_permits(10);
            flush.await?;
        }
        assert!(store.head(&Path::from("test.bin")).await.is_err());
        writer.shutdown().await?;
        let mut sizes = vec![CHUNK_SIZE; 9];
        sizes.push(31);
        assert_eq!(*probe.sizes.lock(), sizes);
        let bytes = store.get(&Path::from("test.bin")).await?.bytes().await?;
        assert_eq!(bytes, returned);
        Ok(())
    }

    #[rstest]
    #[case::part_during_write(5 * CHUNK_SIZE, true, false, false, false)]
    #[case::part_during_flush(CHUNK_SIZE, true, false, false, true)]
    #[case::part_during_shutdown(101, true, false, false, false)]
    #[case::complete(101, false, true, false, false)]
    #[case::part_and_abort(101, true, false, true, false)]
    #[case::complete_and_abort(101, false, true, true, false)]
    #[tokio::test]
    async fn test_upload_errors_cleanup_and_precedence(
        #[case] size: usize,
        #[case] fail_part: bool,
        #[case] fail_complete: bool,
        #[case] fail_abort: bool,
        #[case] flush: bool,
    ) -> VortexResult<()> {
        let directory = tempdir()?;
        let (writer, probe, _) =
            probed_writer(&directory, fail_part, fail_complete, fail_abort).await?;
        let mut writer = writer.with_concurrency(4);
        probe.permits.add_permits(8);
        let write = writer.write_all(vec![3; size]).await;
        let result = if size == 5 * CHUNK_SIZE {
            write.map(|_| ())
        } else {
            write?;
            if flush {
                writer.flush().await
            } else {
                writer.shutdown().await
            }
        };
        let expected = if fail_part {
            "injected part failure"
        } else {
            "injected complete failure"
        };
        assert!(result.unwrap_err().to_string().contains(expected));
        assert_eq!(probe.aborts.load(Ordering::SeqCst), 1);
        assert!(writer.put_result().is_none());
        assert!(writer.write_all(vec![4]).await.is_err());
        assert!(writer.shutdown().await.is_err());
        // Cleanup can be retried even after complete and abort both failed.
        probe.fail_abort.store(false, Ordering::SeqCst);
        writer.abort().await?;
        drop(writer);
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 0);
        assert_eq!(
            probe.completes.load(Ordering::SeqCst),
            usize::from(fail_complete)
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_cancelled_abort_can_be_awaited_again() -> VortexResult<()> {
        let directory = tempdir()?;
        let (mut writer, probe, _) = probed_writer(&directory, false, false, false).await?;
        writer.write_all(vec![7; CHUNK_SIZE]).await?;
        {
            let abort = writer.abort();
            tokio::pin!(abort);
            assert!(futures::poll!(&mut abort).is_pending());
        }
        assert!(writer.shutdown().await.is_err());
        assert_eq!(probe.aborts.load(Ordering::SeqCst), 0);
        probe.permits.add_permits(1);
        writer.abort().await?;
        writer.abort().await?;
        drop(writer);
        assert_eq!(probe.settled.load(Ordering::SeqCst), 1);
        assert_eq!(probe.aborts.load(Ordering::SeqCst), 1);
        assert_eq!(probe.completes.load(Ordering::SeqCst), 0);
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 0);
        Ok(())
    }

    #[rstest]
    #[case::empty(0, false)]
    #[case::empty_abort_error(0, true)]
    #[case::external_failure_after_progress(4 * CHUNK_SIZE + 53, false)]
    #[case::cancelled_oversized_write(5 * CHUNK_SIZE + 53, false)]
    #[tokio::test]
    async fn test_explicit_abort_settles_parts(
        #[case] size: usize,
        #[case] fail_abort: bool,
    ) -> VortexResult<()> {
        let directory = tempdir()?;
        let (writer, probe, _) = probed_writer(&directory, false, false, fail_abort).await?;
        let mut writer = writer.with_concurrency(4);
        if size > 5 * CHUNK_SIZE {
            let write = writer.write_all(vec![7; size]);
            tokio::pin!(write);
            assert!(futures::poll!(&mut write).is_pending());
        } else {
            writer.write_all(vec![7; size]).await?;
        }
        let pending = probe.sizes.lock().len();
        {
            let abort = writer.abort();
            tokio::pin!(abort);
            let polled = futures::poll!(&mut abort);
            if pending != 0 {
                assert!(polled.is_pending());
                assert_eq!(probe.aborts.load(Ordering::SeqCst), 0);
                probe.permits.add_permits(pending);
            }
            let result = match polled {
                Poll::Ready(result) => result,
                Poll::Pending => abort.await,
            };
            if fail_abort {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("injected abort failure")
                );
            } else {
                result?;
            }
        }
        probe.fail_abort.store(false, Ordering::SeqCst);
        writer.abort().await?;
        writer.abort().await?;
        assert!(writer.shutdown().await.is_err());
        assert_eq!(probe.completes.load(Ordering::SeqCst), 0);
        drop(writer);
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 0);
        Ok(())
    }
}
