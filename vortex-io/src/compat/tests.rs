// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::CoalesceConfig;
use crate::IoBuf;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::VortexWrite;
use crate::compat::Compat;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::test_harness::ReadFault;
use crate::test_harness::SimulatedExecutor;
use crate::test_harness::SimulatedReadAt;

fn assert_tokio_context(op: &str) {
    assert!(
        tokio::runtime::Handle::try_current().is_ok(),
        "Compat must enter a Tokio runtime context during {op}"
    );
}

#[test]
fn test_compat_future_enters_tokio_outside_and_reuses_current_inside() {
    assert!(tokio::runtime::Handle::try_current().is_err());

    let exec = SimulatedExecutor::new();
    let fallback_id = exec.drive(Compat::new(async {
        tokio::runtime::Handle::try_current()
            .expect("Compat should provide a fallback Tokio runtime")
            .id()
    }));

    let dedicated_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build current-thread tokio runtime");
    let expected_id = dedicated_rt.handle().id();
    let observed_id = dedicated_rt.block_on(Compat::new(async {
        tokio::runtime::Handle::current().id()
    }));

    assert_eq!(observed_id, expected_id);
    assert_ne!(fallback_id, expected_id);
}

#[test]
fn test_compat_pinned_drop_enters_tokio_context() {
    struct DropRequiresTokio(Arc<AtomicBool>);

    impl Drop for DropRequiresTokio {
        fn drop(&mut self) {
            self.0.store(
                tokio::runtime::Handle::try_current().is_ok(),
                Ordering::SeqCst,
            );
        }
    }

    assert!(tokio::runtime::Handle::try_current().is_err());
    let saw_tokio_on_drop = Arc::new(AtomicBool::new(false));
    let compat = Compat::new(DropRequiresTokio(Arc::clone(&saw_tokio_on_drop)));
    drop(compat);

    assert!(
        saw_tokio_on_drop.load(Ordering::SeqCst),
        "Compat::drop must enter a Tokio context when dropping inner"
    );
}

#[test]
fn test_compat_stream_enters_tokio_forwards_size_hint_and_errors() {
    struct CheckedStream {
        items: std::vec::IntoIter<VortexResult<u32>>,
    }

    impl Stream for CheckedStream {
        type Item = VortexResult<u32>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            assert_tokio_context("Stream::poll_next");
            Poll::Ready(self.items.next())
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.items.size_hint()
        }
    }

    assert!(tokio::runtime::Handle::try_current().is_err());
    let exec = SimulatedExecutor::new();
    let mut compat = Compat::new(CheckedStream {
        items: vec![Ok(10), Err(vortex_error::vortex_err!("stream boom"))].into_iter(),
    });
    assert_eq!(compat.size_hint(), (2, Some(2)));

    let first = exec
        .drive(compat.next())
        .expect("first item present")
        .expect("first item ok");
    assert_eq!(first, 10);
    assert_eq!(compat.size_hint(), (1, Some(1)));

    let second_err = exec
        .drive(compat.next())
        .expect("second item present")
        .expect_err("second item must propagate error");
    assert!(
        second_err.to_string().contains("stream boom"),
        "{second_err}"
    );
    assert!(exec.drive(compat.next()).is_none());
}

struct TokioRequiringReadAt {
    inner: SimulatedReadAt,
}

impl VortexReadAt for TokioRequiringReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        self.inner.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let fut = self.inner.size();
        async move {
            assert_tokio_context("VortexReadAt::size");
            fut.await
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let fut = self.inner.read_at(offset, length, alignment);
        async move {
            assert_tokio_context("VortexReadAt::read_at");
            fut.await
        }
        .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        let inner_stream = self.inner.read_ranges(requests);
        inner_stream
            .map(|item| {
                assert_tokio_context("VortexReadAt::read_ranges");
                item
            })
            .boxed()
    }
}

#[test]
fn test_compat_read_at_forwards_metadata_reads_ranges_and_errors() -> VortexResult<()> {
    assert!(tokio::runtime::Handle::try_current().is_err());
    let exec = SimulatedExecutor::new();
    let sim = SimulatedReadAt::new(ByteBuffer::from(b"hello vortex".to_vec()))
        .with_uri("sim://compat-test")
        .with_concurrency(3)
        .with_coalesce_config(CoalesceConfig::new(64, 1024));
    let compat = Compat::new(TokioRequiringReadAt { inner: sim.clone() });

    assert_eq!(compat.uri().map(Arc::as_ref), Some("sim://compat-test"));
    assert_eq!(compat.concurrency(), 3);
    assert_eq!(compat.coalesce_config().map(|c| c.max_size), Some(1024));
    assert_eq!(exec.drive(compat.size())?, 12);

    let slice = exec.drive(compat.read_at(6, 6, Alignment::none()))?;
    assert_eq!(exec.drive(slice.to_host()).as_ref(), b"vortex");

    let oob_err = exec
        .drive(compat.read_at(10, 5, Alignment::none()))
        .expect_err("out-of-bounds read must fail");
    assert!(oob_err.to_string().contains("out of bounds"), "{oob_err}");

    sim.inject_fault_for_offset(
        0,
        ReadFault::TransientError {
            remaining: 1,
            message: "injected compat read error".to_string(),
        },
    );
    let requests = Arc::from([
        ReadAtRequest::new(0, 5, Alignment::none()),
        ReadAtRequest::new(6, 6, Alignment::none()),
    ]);
    let results = exec.drive(compat.read_ranges(requests).collect::<Vec<_>>());
    assert_eq!(results.len(), 2);

    let mut saw_ok = false;
    let mut saw_err = false;
    for (req, res) in results {
        match req.offset {
            0 => {
                let err = res.expect_err("offset 0 fault should propagate");
                assert!(
                    err.to_string().contains("injected compat read error"),
                    "{err}"
                );
                saw_err = true;
            }
            6 => {
                assert_eq!(exec.drive(res?.to_host()).as_ref(), b"vortex");
                saw_ok = true;
            }
            other => panic!("unexpected offset {other}"),
        }
    }
    assert!(saw_ok && saw_err);
    Ok(())
}

#[test]
fn test_compat_write_executes_in_tokio_and_propagates_errors() -> io::Result<()> {
    #[derive(Default)]
    struct TokioRequiringWrite {
        written: Vec<u8>,
        flushed: bool,
        shut_down: bool,
        fail_write: Option<io::ErrorKind>,
        fail_flush: bool,
        fail_shutdown: bool,
    }

    impl VortexWrite for TokioRequiringWrite {
        fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> + Send {
            let fail = self.fail_write;
            let bytes = buffer.as_slice().to_vec();
            if fail.is_none() {
                self.written.extend_from_slice(&bytes);
            }
            async move {
                assert_tokio_context("VortexWrite::write_all");
                if let Some(kind) = fail {
                    return Err(io::Error::new(kind, "write_all injected failure"));
                }
                Ok(buffer)
            }
        }

        fn flush(&mut self) -> impl Future<Output = io::Result<()>> + Send {
            let fail = self.fail_flush;
            if !fail {
                self.flushed = true;
            }
            async move {
                assert_tokio_context("VortexWrite::flush");
                if fail {
                    return Err(io::Error::other("flush injected failure"));
                }
                Ok(())
            }
        }

        fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> + Send {
            let fail = self.fail_shutdown;
            if !fail {
                self.shut_down = true;
            }
            async move {
                assert_tokio_context("VortexWrite::shutdown");
                if fail {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "shutdown injected failure",
                    ));
                }
                Ok(())
            }
        }
    }

    assert!(tokio::runtime::Handle::try_current().is_err());
    let exec = SimulatedExecutor::new();

    let mut writer = Compat::new(TokioRequiringWrite::default());
    let returned = exec.drive(writer.write_all(b"abc".to_vec()))?;
    assert_eq!(returned, b"abc");
    exec.drive(writer.write_all(b"def".to_vec()))?;
    exec.drive(writer.flush())?;
    exec.drive(writer.shutdown())?;
    assert_eq!(writer.inner().written, b"abcdef");
    assert!(writer.inner().flushed);
    assert!(writer.inner().shut_down);

    let mut failing = Compat::new(TokioRequiringWrite {
        fail_write: Some(io::ErrorKind::WriteZero),
        fail_flush: true,
        fail_shutdown: true,
        ..Default::default()
    });
    let write_err = exec
        .drive(failing.write_all(b"x".to_vec()))
        .expect_err("write error must propagate");
    assert_eq!(write_err.kind(), io::ErrorKind::WriteZero);
    assert!(write_err.to_string().contains("write_all injected failure"));

    let flush_err = exec
        .drive(failing.flush())
        .expect_err("flush error must propagate");
    assert!(flush_err.to_string().contains("flush injected failure"));

    let shutdown_err = exec
        .drive(failing.shutdown())
        .expect_err("shutdown error must propagate");
    assert_eq!(shutdown_err.kind(), io::ErrorKind::BrokenPipe);
    assert!(
        shutdown_err
            .to_string()
            .contains("shutdown injected failure")
    );
    Ok(())
}

#[test]
fn test_compat_filesystem_wraps_ops_and_opened_reader() -> VortexResult<()> {
    #[derive(Debug)]
    struct TokioRequiringFs {
        files: Mutex<std::collections::BTreeMap<String, ByteBuffer>>,
    }

    #[async_trait]
    impl FileSystem for TokioRequiringFs {
        fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
            let prefix = prefix.to_owned();
            let items: Vec<FileListing> = self
                .files
                .lock()
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(k, v)| FileListing {
                    path: k.clone(),
                    size: Some(v.len() as u64),
                })
                .collect();
            stream::iter(items)
                .map(|item| {
                    assert_tokio_context("FileSystem::list");
                    Ok(item)
                })
                .boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            assert_tokio_context("FileSystem::head");
            Ok(self.files.lock().get(path).map(|b| FileListing {
                path: path.to_owned(),
                size: Some(b.len() as u64),
            }))
        }

        async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            assert_tokio_context("FileSystem::open_read");
            let Some(bytes) = self.files.lock().get(path).cloned() else {
                vortex_bail!("missing file: {path}");
            };
            Ok(Arc::new(TokioRequiringReadAt {
                inner: SimulatedReadAt::new(bytes),
            }))
        }

        async fn delete(&self, path: &str) -> VortexResult<()> {
            assert_tokio_context("FileSystem::delete");
            if self.files.lock().remove(path).is_none() {
                vortex_bail!("cannot delete missing file: {path}");
            }
            Ok(())
        }
    }

    assert!(tokio::runtime::Handle::try_current().is_err());
    let exec = SimulatedExecutor::new();
    let mut initial = std::collections::BTreeMap::new();
    initial.insert(
        "dir/a.vortex".to_owned(),
        ByteBuffer::from(b"payload".to_vec()),
    );
    let fs = Compat::new(TokioRequiringFs {
        files: Mutex::new(initial),
    });

    let listed = exec.drive(fs.list("dir/").try_collect::<Vec<_>>())?;
    assert_eq!(
        listed,
        vec![FileListing {
            path: "dir/a.vortex".to_owned(),
            size: Some(7),
        }]
    );

    assert_eq!(
        exec.drive(fs.head("dir/a.vortex"))?,
        Some(FileListing {
            path: "dir/a.vortex".to_owned(),
            size: Some(7),
        })
    );
    assert_eq!(exec.drive(fs.head("dir/missing.vortex"))?, None);

    // `open_read` must wrap the returned reader in `Compat` too so reads outside Tokio work.
    let reader = exec.drive(fs.open_read("dir/a.vortex"))?;
    assert_eq!(exec.drive(reader.size())?, 7);
    let buf = exec.drive(reader.read_at(0, 7, Alignment::none()))?;
    assert_eq!(exec.drive(buf.to_host()).as_ref(), b"payload");

    let open_err = exec
        .drive(fs.open_read("dir/missing.vortex"))
        .err()
        .expect("missing open_read must fail");
    assert!(
        open_err
            .to_string()
            .contains("missing file: dir/missing.vortex"),
        "{open_err}"
    );

    exec.drive(fs.delete("dir/a.vortex"))?;
    assert_eq!(exec.drive(fs.head("dir/a.vortex"))?, None);

    let del_err = exec
        .drive(fs.delete("dir/a.vortex"))
        .expect_err("deleting already-deleted file must fail");
    assert!(
        del_err
            .to_string()
            .contains("cannot delete missing file: dir/a.vortex"),
        "{del_err}"
    );
    Ok(())
}

#[cfg(feature = "object_store")]
#[expect(clippy::unimplemented)]
mod obj_store_tests {
    use std::fmt::Display;
    use std::fmt::Formatter;
    use std::ops::Range;

    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::StreamExt;
    use futures::TryStreamExt;
    use futures::stream::BoxStream;
    use object_store::CopyOptions;
    use object_store::GetOptions;
    use object_store::GetResult;
    use object_store::GetResultPayload;
    use object_store::ListResult;
    use object_store::MultipartUpload;
    use object_store::ObjectMeta;
    use object_store::ObjectStore;
    use object_store::ObjectStoreExt;
    use object_store::PutMultipartOptions;
    use object_store::PutOptions;
    use object_store::PutPayload;
    use object_store::PutResult;
    use object_store::UploadPart;
    use object_store::memory::InMemory;
    use object_store::path::Path;

    use super::assert_tokio_context;
    use crate::compat::Compat;
    use crate::test_harness::SimulatedExecutor;

    #[derive(Debug)]
    struct TokioCheckedStore {
        inner: InMemory,
    }

    impl Display for TokioCheckedStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "TokioCheckedStore")
        }
    }

    #[derive(Debug)]
    struct TokioCheckedUpload {
        inner: Box<dyn MultipartUpload>,
    }

    #[async_trait]
    impl MultipartUpload for TokioCheckedUpload {
        fn put_part(&mut self, data: PutPayload) -> UploadPart {
            let fut = self.inner.put_part(data);
            Box::pin(async move {
                assert_tokio_context("MultipartUpload::put_part");
                fut.await
            })
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            assert_tokio_context("MultipartUpload::complete");
            self.inner.complete().await
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            assert_tokio_context("MultipartUpload::abort");
            self.inner.abort().await
        }
    }

    #[async_trait]
    impl ObjectStore for TokioCheckedStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            assert_tokio_context("ObjectStore::put_opts");
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            assert_tokio_context("ObjectStore::put_multipart_opts");
            let upload = self.inner.put_multipart_opts(location, opts).await?;
            Ok(Box::new(TokioCheckedUpload { inner: upload }))
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            assert_tokio_context("ObjectStore::get_opts");
            let mut result = self.inner.get_opts(location, options).await?;
            result.payload = match result.payload {
                GetResultPayload::Stream(s) => GetResultPayload::Stream(
                    s.map(|item| {
                        assert_tokio_context("GetResultPayload::Stream");
                        item
                    })
                    .boxed(),
                ),
                #[cfg(not(target_arch = "wasm32"))]
                GetResultPayload::File(file, path) => GetResultPayload::File(file, path),
            };
            Ok(result)
        }

        async fn get_ranges(
            &self,
            location: &Path,
            ranges: &[Range<u64>],
        ) -> object_store::Result<Vec<Bytes>> {
            assert_tokio_context("ObjectStore::get_ranges");
            self.inner.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            unimplemented!()
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner
                .list(prefix)
                .map(|item| {
                    assert_tokio_context("ObjectStore::list");
                    item
                })
                .boxed()
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            assert_tokio_context("ObjectStore::list_with_delimiter");
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            _from: &Path,
            _to: &Path,
            _options: CopyOptions,
        ) -> object_store::Result<()> {
            unimplemented!()
        }
    }

    #[test]
    fn test_compat_object_store_and_multipart_outside_tokio() -> object_store::Result<()> {
        assert!(tokio::runtime::Handle::try_current().is_err());
        let exec = SimulatedExecutor::new();
        let store = Compat::new(TokioCheckedStore {
            inner: InMemory::new(),
        });

        let path_a = Path::from("dir/a.bin");
        exec.drive(store.put_opts(
            &path_a,
            PutPayload::from("0123456789"),
            PutOptions::default(),
        ))?;

        // `get_opts` must also wrap `GetResultPayload::Stream` in `Compat`.
        let get_res = exec.drive(store.get_opts(&path_a, GetOptions::default()))?;
        assert_eq!(exec.drive(get_res.bytes())?.as_ref(), b"0123456789");

        let ranges = exec.drive(store.get_ranges(&path_a, &[0..3, 7..10]))?;
        assert_eq!(ranges[0].as_ref(), b"012");
        assert_eq!(ranges[1].as_ref(), b"789");

        let listed: Vec<_> = exec.drive(store.list(Some(&Path::from("dir"))).try_collect())?;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].location, path_a);

        let delimited = exec.drive(store.list_with_delimiter(Some(&Path::from("dir"))))?;
        assert_eq!(delimited.objects.len(), 1);

        // MultipartUpload complete and abort must both run inside Tokio.
        let path_mp = Path::from("dir/mp.bin");
        let mut upload =
            exec.drive(store.put_multipart_opts(&path_mp, PutMultipartOptions::default()))?;
        exec.drive(upload.put_part(PutPayload::from("part1-")))?;
        exec.drive(upload.put_part(PutPayload::from("part2")))?;
        exec.drive(upload.complete())?;
        assert_eq!(
            exec.drive(exec.drive(store.get(&path_mp))?.bytes())?
                .as_ref(),
            b"part1-part2"
        );

        let path_aborted = Path::from("dir/aborted.bin");
        let mut aborted_upload =
            exec.drive(store.put_multipart_opts(&path_aborted, PutMultipartOptions::default()))?;
        exec.drive(aborted_upload.put_part(PutPayload::from("discarded")))?;
        exec.drive(aborted_upload.abort())?;
        let missing_err = exec
            .drive(store.get(&path_aborted))
            .expect_err("aborted multipart upload must not exist");
        assert!(matches!(missing_err, object_store::Error::NotFound { .. }));
        Ok(())
    }
}
