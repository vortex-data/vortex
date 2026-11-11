// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::BytesMut;
use futures::StreamExt;
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

/// Adapter type to write data through a [`ObjectStore`] instance.
///
/// Buffers at most the configured number of in-flight 16 MiB parts and one 16 MiB staging part.
/// [`Self::with_concurrency`] defaults to eight, matching [`object_store::buffered::BufWriter`].
/// This bound excludes caller-owned buffers and allocations internal to the object store.
///
/// [`VortexWrite::flush`] waits for submitted parts without publishing the object or uploading a
/// partial tail. Only [`VortexWrite::shutdown`] publishes it, so callers can validate the completed
/// Vortex file's footer and statistics before committing the upload.
///
/// Upload errors trigger best-effort cleanup without replacing the original error. Call
/// [`Self::abort`] for external errors, empty output, or after cancelling a write or shutdown.
/// Dropping this writer does not guarantee cleanup. A cancelled operation must not be resumed.
pub struct ObjectStoreWrite {
    upload: Box<dyn MultipartUpload>,
    buffer: BytesMut,
    parts: FuturesUnordered<UploadPart>,
    concurrency: usize,
    put_result: Option<PutResult>,
    state: State,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;
const DEFAULT_CONCURRENCY: usize = 8;

enum State {
    Open,
    Failed,
    Aborted,
    Completed,
}

impl ObjectStoreWrite {
    /// Start a multipart upload. The object is not published until shutdown succeeds.
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            upload,
            buffer: BytesMut::with_capacity(CHUNK_SIZE),
            parts: FuturesUnordered::new(),
            concurrency: DEFAULT_CONCURRENCY,
            put_result: None,
            state: State::Open,
        })
    }

    /// Set the maximum number of in-flight parts, defaulting to eight.
    ///
    /// Lower concurrency reduces adapter-owned buffering; higher concurrency may improve upload
    /// throughput. Zero is treated as one. Configure this before writing: parts already submitted
    /// are not cancelled when lowering the limit.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// The result of a successful shutdown, or `None` before publication.
    pub fn put_result(&self) -> Option<&PutResult> {
        self.put_result.as_ref()
    }

    /// Discard staged bytes, settle all submitted parts, and abort this upload.
    ///
    /// May be retried after cleanup fails, including after a completion error. Repeated calls
    /// after a successful abort are harmless. A successfully published object cannot be aborted.
    /// When handling an external write or validation error, preserve that error if abort also
    /// fails. For otherwise successful empty output, propagate the abort error instead.
    pub async fn abort(&mut self) -> io::Result<()> {
        match self.state {
            State::Aborted => return Ok(()),
            State::Completed => return Err(io::Error::other("upload already completed")),
            State::Open | State::Failed => self.state = State::Failed,
        }
        self.buffer = BytesMut::new();
        // Some stores spawn work internally. Dropping part futures need not stop that work;
        // settle every part before aborting so a late part cannot race with cleanup.
        while self.parts.next().await.is_some() {}
        self.upload.abort().await?;
        self.state = State::Aborted;
        Ok(())
    }

    fn ensure_open(&self) -> io::Result<()> {
        if !matches!(self.state, State::Open) {
            return Err(io::Error::other("upload is no longer writable"));
        }
        Ok(())
    }

    async fn upload_buffer(&mut self) -> io::Result<()> {
        while self.parts.len() >= self.concurrency {
            self.parts.next().await.transpose()?;
        }
        // Do not split a growing allocation: each submitted part and the new staging buffer
        // must have independent, fixed-size allocations.
        let payload = std::mem::replace(&mut self.buffer, BytesMut::with_capacity(CHUNK_SIZE));
        self.parts.push(
            self.upload
                .put_part(PutPayload::from_bytes(payload.freeze())),
        );
        Ok(())
    }

    async fn cleanup_error(&mut self, error: io::Error) -> io::Error {
        if let Err(cleanup) = self.abort().await {
            tracing::warn!(%cleanup, "Failed to abort multipart upload");
        }
        error
    }
}

impl VortexWrite for ObjectStoreWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.ensure_open()?;
        let mut remaining = buffer.as_slice();
        while !remaining.is_empty() {
            let len = remaining.len().min(CHUNK_SIZE - self.buffer.len());
            self.buffer.extend_from_slice(&remaining[..len]);
            remaining = &remaining[len..];
            if self.buffer.len() == CHUNK_SIZE
                && let Err(error) = self.upload_buffer().await
            {
                return Err(self.cleanup_error(error).await);
            }
        }
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        if matches!(self.state, State::Completed) {
            return Ok(());
        }
        self.ensure_open()?;
        while let Some(result) = self.parts.next().await {
            if let Err(error) = result {
                return Err(self.cleanup_error(error.into()).await);
            }
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        if matches!(self.state, State::Completed) {
            return Ok(());
        }
        self.ensure_open()?;
        if !self.buffer.is_empty()
            && let Err(error) = self.upload_buffer().await
        {
            return Err(self.cleanup_error(error).await);
        }
        self.flush().await?;
        // Cancellation or failure of complete must not allow another publication attempt.
        self.state = State::Failed;
        match self.upload.complete().await {
            Ok(result) => {
                self.put_result = Some(result);
                self.state = State::Completed;
                self.buffer = BytesMut::new();
            }
            Err(error) => return Err(self.cleanup_error(error.into()).await),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
    #[case(100)]
    #[case(8 * 1024 * 1024)]
    #[case(CHUNK_SIZE)]
    #[case(25 * 1024 * 1024)]
    #[case(26 * 1024 * 1024)]
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
        let probe = Arc::new(Probe {
            upload: tokio::sync::Mutex::new(store.put_multipart(&Path::from("test.bin")).await?),
            sizes: Mutex::new(Vec::new()),
            settled: AtomicUsize::new(0),
            aborts: AtomicUsize::new(0),
            completes: AtomicUsize::new(0),
            permits: Semaphore::new(0),
            fail_part,
            fail_complete,
            fail_abort: AtomicBool::new(fail_abort),
        });
        let writer = ObjectStoreWrite {
            upload: Box::new(ProbedUpload(Arc::clone(&probe))),
            buffer: BytesMut::with_capacity(CHUNK_SIZE),
            parts: FuturesUnordered::new(),
            concurrency: DEFAULT_CONCURRENCY,
            put_result: None,
            state: State::Open,
        }
        .with_concurrency(4);
        Ok((writer, probe, store))
    }

    #[rstest]
    #[case::exact_parts(0)]
    #[case::partial_tail(31)]
    #[tokio::test]
    async fn test_oversized_write_backpressure_and_order(
        #[case] tail: usize,
        #[values(0, 1, 4, 8)] concurrency: usize,
    ) -> VortexResult<()> {
        let directory = tempdir()?;
        let (writer, probe, store) = probed_writer(&directory, false, false, false).await?;
        let mut writer = writer.with_concurrency(concurrency);
        let concurrency = concurrency.max(1);
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
        let (mut writer, probe, _) =
            probed_writer(&directory, fail_part, fail_complete, fail_abort).await?;
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
        let (mut writer, probe, _) = probed_writer(&directory, false, false, fail_abort).await?;
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
                std::task::Poll::Ready(result) => result,
                std::task::Poll::Pending => abort.await,
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
