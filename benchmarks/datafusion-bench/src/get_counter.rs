// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Counts the GETs a run sends to the benchmark's object store, with `--opt get-log=<path>`: how
//! many, their bytes, and how long each took, from its request until its body was all in or
//! dropped. A process appends one JSON line to the path for each run, and one for each stretch
//! between runs (opening the tables, a cache's leftovers); a GET counts where it finished.

use std::fmt;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::CopyOptions;
use object_store::GetOptions;
use object_store::GetResult;
use object_store::GetResultPayload;
use object_store::ListResult;
use object_store::MultipartUpload;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::PutMultipartOptions;
use object_store::PutOptions;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::RenameOptions;
use object_store::path::Path;

/// Where a [`GetCounter::log`] line's GETs finished.
#[derive(Debug, Clone, Copy)]
pub enum GetSegment {
    Run,
    BetweenRuns,
}

/// The GETs counted since the last [`Self::log`], over every store a [`CountingStore`] wraps
/// with it.
#[derive(Debug)]
pub struct GetCounter {
    log: PathBuf,
    gets: AtomicU64,
    bytes: AtomicU64,
    heads: AtomicU64,
    slowest_us: AtomicU64,
    over_1s: AtomicU64,
    over_2s: AtomicU64,
}

impl GetCounter {
    /// A counter whose [`Self::log`] appends to the file at `log`.
    pub fn logging_to(log: impl Into<PathBuf>) -> Self {
        Self {
            log: log.into(),
            gets: AtomicU64::default(),
            bytes: AtomicU64::default(),
            heads: AtomicU64::default(),
            slowest_us: AtomicU64::default(),
            over_1s: AtomicU64::default(),
            over_2s: AtomicU64::default(),
        }
    }

    /// `store`, its GETs counted here.
    pub fn wrap(self: &Arc<Self>, store: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(CountingStore {
            inner: store,
            counter: Arc::clone(self),
        })
    }

    /// Appends the counts since the last call, as one JSON line, the GETs of `segment`.
    pub fn log(&self, segment: GetSegment) -> anyhow::Result<()> {
        let segment = match segment {
            GetSegment::Run => "run",
            GetSegment::BetweenRuns => "between",
        };
        let line = format!(
            "{{\"segment\":\"{segment}\",\"gets\":{},\"bytes\":{},\"heads\":{},\"slowest_get_ms\":{:.3},\"gets_over_1s\":{},\"gets_over_2s\":{}}}\n",
            self.gets.swap(0, Ordering::Relaxed),
            self.bytes.swap(0, Ordering::Relaxed),
            self.heads.swap(0, Ordering::Relaxed),
            self.slowest_us.swap(0, Ordering::Relaxed) as f64 / 1000.0,
            self.over_1s.swap(0, Ordering::Relaxed),
            self.over_2s.swap(0, Ordering::Relaxed),
        );
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log)?
            .write_all(line.as_bytes())?;
        Ok(())
    }

    fn finished(&self, took: Duration, bytes: u64) {
        let us = u64::try_from(took.as_micros()).unwrap_or(u64::MAX);
        self.gets.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.slowest_us.fetch_max(us, Ordering::Relaxed);
        if took > Duration::from_secs(1) {
            self.over_1s.fetch_add(1, Ordering::Relaxed);
        }
        if took > Duration::from_secs(2) {
            self.over_2s.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A GET's body as it comes: counted once, when it is all in or dropped.
struct BodyTimer {
    counter: Arc<GetCounter>,
    sent: Instant,
    bytes: u64,
}

impl BodyTimer {
    /// `piece`, the body's next bytes, came. A method, so that the body's closure captures the
    /// whole timer, which drops with the body, not a copy of its count.
    fn received(&mut self, piece: &[u8]) {
        self.bytes += piece.len() as u64;
    }
}

impl Drop for BodyTimer {
    fn drop(&mut self) {
        self.counter.finished(self.sent.elapsed(), self.bytes);
    }
}

#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn ObjectStore>,
    counter: Arc<GetCounter>,
}

impl fmt::Display for CountingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GETs counted over {}", self.inner)
    }
}

#[async_trait]
impl ObjectStore for CountingStore {
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
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if options.head {
            self.counter.heads.fetch_add(1, Ordering::Relaxed);
            return self.inner.get_opts(location, options).await;
        }
        let mut timer = BodyTimer {
            counter: Arc::clone(&self.counter),
            sent: Instant::now(),
            bytes: 0,
        };
        let answer = self.inner.get_opts(location, options).await?;
        let GetResultPayload::Stream(body) = answer.payload else {
            return Ok(answer);
        };
        let body = body
            .map(move |piece| {
                if let Ok(piece) = &piece {
                    timer.received(piece);
                }
                piece
            })
            .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(body),
            ..answer
        })
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}
