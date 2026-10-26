// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use object_store::Attributes;
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
use object_store::Result;
use object_store::path::Path;
use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Model {
    pub ttfb_ms: f64,
    pub request_mib_per_sec: f64,
    pub max_inflight: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct Counts {
    pub get: u64,
    pub head: u64,
    pub list: u64,
    pub put: u64,
    pub bytes: u64,
    pub errors: u64,
    pub peak_inflight: usize,
    pub request_seconds: f64,
    #[serde(skip)]
    inflight: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct Request {
    operation: &'static str,
    path: String,
    requested_range: String,
    bytes: u64,
    first_byte_ms: Option<f64>,
    duration_ms: f64,
    inflight_at_start: usize,
}

#[derive(Debug, Default)]
struct Metrics {
    counts: Counts,
    requests: Vec<Request>,
}

#[derive(Debug)]
struct Guard {
    metrics: Arc<Mutex<Metrics>>,
    start: Instant,
    request: Request,
    // Hold admission until the response stream completes or is cancelled.
    _permit: Option<OwnedSemaphorePermit>,
}

impl Guard {
    fn bytes(&mut self, len: usize) {
        self.request.bytes += len as u64;
        self.request
            .first_byte_ms
            .get_or_insert_with(|| self.start.elapsed().as_secs_f64() * 1000.0);
        self.metrics.lock().counts.bytes += len as u64;
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        self.request.duration_ms = elapsed * 1000.0;
        let mut metrics = self.metrics.lock();
        metrics.counts.inflight -= 1;
        metrics.counts.request_seconds += elapsed;
        metrics.requests.push(Request {
            operation: self.request.operation,
            path: std::mem::take(&mut self.request.path),
            requested_range: std::mem::take(&mut self.request.requested_range),
            bytes: self.request.bytes,
            first_byte_ms: self.request.first_byte_ms,
            duration_ms: self.request.duration_ms,
            inflight_at_start: self.request.inflight_at_start,
        });
    }
}

#[derive(Debug)]
pub(super) struct Object {
    meta: ObjectMeta,
    list_meta: ObjectMeta,
    attributes: Attributes,
    data: Bytes,
}

pub(super) type Snapshot = Arc<BTreeMap<Path, Object>>;

/// Copy the source bytes and metadata once, outside every measurement window.
pub(super) async fn snapshot(store: &dyn ObjectStore, prefix: &Path) -> anyhow::Result<Snapshot> {
    let objects: Vec<ObjectMeta> = store.list(Some(prefix)).try_collect().await?;
    anyhow::ensure!(!objects.is_empty(), "No objects under {prefix}");
    anyhow::ensure!(
        objects.len() < 1000,
        "Validation requires single-page flat listings"
    );
    let size: u64 = objects.iter().map(|m| m.size).sum();
    let system = sysinfo::System::new_all();
    let available = system.available_memory();
    // Leave room for the download buffer and at least 60% of available RAM for queries.
    anyhow::ensure!(
        size < available * 2 / 5,
        "Dataset is {size} bytes; only {available} bytes RAM available"
    );
    eprintln!(
        "Preloading {} objects ({size} bytes), available RAM {available}",
        objects.len()
    );
    let mut result = BTreeMap::new();
    for meta in objects {
        let response = store.get(&meta.location).await?;
        let attributes = response.attributes.clone();
        let actual_meta = response.meta.clone();
        let data = response.bytes().await?;
        anyhow::ensure!(
            actual_meta.e_tag == meta.e_tag && data.len() as u64 == meta.size,
            "Object changed during preload: {}",
            meta.location
        );
        result.insert(
            meta.location.clone(),
            Object {
                list_meta: meta,
                meta: actual_meta,
                attributes,
                data,
            },
        );
    }
    Ok(Arc::new(result))
}

#[derive(Debug)]
pub(super) struct Store {
    source: Arc<dyn ObjectStore>,
    snapshot: Option<Snapshot>,
    model: Option<Model>,
    admission: Option<Arc<Semaphore>>,
    metrics: Arc<Mutex<Metrics>>,
}

impl Store {
    pub fn new(
        source: Arc<dyn ObjectStore>,
        snapshot: Option<Snapshot>,
        model: Option<Model>,
    ) -> Self {
        Self {
            source,
            snapshot,
            model,
            admission: model
                .filter(|m| m.max_inflight > 0)
                .map(|m| Arc::new(Semaphore::new(m.max_inflight))),
            metrics: Arc::default(),
        }
    }

    pub fn reset(&self) -> anyhow::Result<()> {
        let mut metrics = self.metrics.lock();
        anyhow::ensure!(
            metrics.counts.inflight == 0,
            "Requests still in flight at measurement boundary"
        );
        *metrics = Metrics::default();
        Ok(())
    }

    pub fn take(&self) -> anyhow::Result<(Counts, Vec<Request>)> {
        let mut metrics = self.metrics.lock();
        anyhow::ensure!(
            metrics.counts.inflight == 0,
            "Requests still in flight at measurement boundary"
        );
        Ok((
            metrics.counts.clone(),
            std::mem::take(&mut metrics.requests),
        ))
    }

    async fn begin(&self, operation: &'static str, path: String, range: String) -> Result<Guard> {
        let permit = match &self.admission {
            Some(semaphore) => Some(Arc::clone(semaphore).acquire_owned().await.map_err(|e| {
                object_store::Error::Generic {
                    store: "s3-model",
                    source: Box::new(e),
                }
            })?),
            None => None,
        };
        let guard = {
            let mut metrics = self.metrics.lock();
            match operation {
                "GET" => metrics.counts.get += 1,
                "HEAD" => metrics.counts.head += 1,
                "LIST" => metrics.counts.list += 1,
                "PUT" => metrics.counts.put += 1,
                _ => {}
            }
            metrics.counts.inflight += 1;
            metrics.counts.peak_inflight =
                metrics.counts.peak_inflight.max(metrics.counts.inflight);
            Guard {
                metrics: Arc::clone(&self.metrics),
                start: Instant::now(),
                request: Request {
                    operation,
                    path,
                    requested_range: range,
                    bytes: 0,
                    first_byte_ms: None,
                    duration_ms: 0.0,
                    inflight_at_start: metrics.counts.inflight,
                },
                _permit: permit,
            }
        };
        if let Some(model) = self.model.filter(|m| m.ttfb_ms > 0.0) {
            tokio::time::sleep(Duration::from_secs_f64(model.ttfb_ms / 1000.0)).await;
        }
        Ok(guard)
    }

    fn ram_get(
        &self,
        location: &Path,
        options: &GetOptions,
        objects: &Snapshot,
        request_start: Instant,
    ) -> Result<GetResult> {
        let object = objects
            .get(location)
            .ok_or_else(|| object_store::Error::NotFound {
                path: location.to_string(),
                source: "Object missing from immutable snapshot".into(),
            })?;
        options.check_preconditions(&object.meta)?;
        if options.version.is_some() && options.version != object.meta.version {
            return Err(object_store::Error::NotFound {
                path: location.to_string(),
                source: "Version missing from snapshot".into(),
            });
        }
        let range = options
            .range
            .as_ref()
            .map(|r| r.as_range(object.meta.size))
            .transpose()
            .map_err(|e| object_store::Error::Generic {
                store: "s3-model",
                source: Box::new(e),
            })?
            .unwrap_or(0..object.meta.size);
        let data = if options.head {
            Bytes::new()
        } else {
            let start = usize::try_from(range.start).map_err(|e| object_store::Error::Generic {
                store: "s3-model",
                source: Box::new(e),
            })?;
            let end = usize::try_from(range.end).map_err(|e| object_store::Error::Generic {
                store: "s3-model",
                source: Box::new(e),
            })?;
            object.data.slice(start..end)
        };
        let model = self.model;
        let start = tokio::time::Instant::from_std(request_start);
        let payload = stream::try_unfold((data, 0usize), move |(data, offset)| async move {
            if offset == data.len() {
                return Ok(None);
            }
            let end = (offset + 256 * 1024).min(data.len());
            if let Some(model) = model {
                // Absolute deadlines avoid adding a timer tick for each response chunk.
                let deadline = start
                    + Duration::from_secs_f64(
                        model.ttfb_ms / 1000.0
                            + end as f64 / (model.request_mib_per_sec * 1048576.0),
                    );
                tokio::time::sleep_until(deadline).await;
            }
            Ok(Some((data.slice(offset..end), (data, end))))
        })
        .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(payload),
            meta: object.meta.clone(),
            range,
            attributes: object.attributes.clone(),
        })
    }
}

impl fmt::Display for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S3ValidationStore")
    }
}

fn read_only(operation: &str) -> object_store::Error {
    object_store::Error::NotSupported {
        source: format!("Read-only validation store: {operation}").into(),
    }
}

#[async_trait]
impl ObjectStore for Store {
    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let head = options.head;
        let guard = self
            .begin(
                if head { "HEAD" } else { "GET" },
                location.to_string(),
                format!("{:?}", options.range),
            )
            .await?;
        let result = match &self.snapshot {
            Some(objects) => self.ram_get(location, &options, objects, guard.start),
            None => self.source.get_opts(location, options).await,
        };
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                self.metrics.lock().counts.errors += 1;
                return Err(error);
            }
        };
        if head {
            drop(guard);
            return Ok(response);
        }
        let meta = response.meta.clone();
        let range = response.range.clone();
        let attributes = response.attributes.clone();
        let payload = stream::try_unfold(
            (response.into_stream(), guard),
            |(mut stream, mut guard)| async move {
                match stream.try_next().await {
                    Ok(Some(bytes)) => {
                        guard.bytes(bytes.len());
                        Ok(Some((bytes, (stream, guard))))
                    }
                    Ok(None) => Ok(None),
                    Err(error) => {
                        guard.metrics.lock().counts.errors += 1;
                        Err(error)
                    }
                }
            },
        )
        .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(payload),
            meta,
            range,
            attributes,
        })
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        let prefix = prefix.cloned();
        let metrics = Arc::clone(&self.metrics);
        let model = self.model;
        let source = Arc::clone(&self.source);
        let objects = self.snapshot.clone();
        let admission = self.admission.clone();
        stream::once(async move {
            let permit = match admission {
                Some(semaphore) => Some(semaphore.acquire_owned().await.map_err(|e| {
                    object_store::Error::Generic {
                        store: "s3-model",
                        source: Box::new(e),
                    }
                })),
                None => None,
            };
            let permit = match permit.transpose() {
                Ok(permit) => permit,
                Err(error) => return stream::once(async { Err(error) }).boxed(),
            };
            let guard = {
                let mut stats = metrics.lock();
                stats.counts.list += 1;
                stats.counts.inflight += 1;
                stats.counts.peak_inflight = stats.counts.peak_inflight.max(stats.counts.inflight);
                Guard {
                    metrics: Arc::clone(&metrics),
                    start: Instant::now(),
                    _permit: permit,
                    request: Request {
                        operation: "LIST",
                        path: prefix.as_ref().map(ToString::to_string).unwrap_or_default(),
                        requested_range: String::new(),
                        bytes: 0,
                        first_byte_ms: None,
                        duration_ms: 0.0,
                        inflight_at_start: stats.counts.inflight,
                    },
                }
            };
            if let Some(model) = model.filter(|m| m.ttfb_ms > 0.0) {
                tokio::time::sleep(Duration::from_secs_f64(model.ttfb_ms / 1000.0)).await;
            }
            let inner = match objects {
                Some(objects) => stream::iter(
                    objects
                        .values()
                        .filter(|o| {
                            prefix
                                .as_ref()
                                .is_none_or(|p| o.meta.location.prefix_matches(p))
                        })
                        .map(|o| Ok(o.list_meta.clone()))
                        .collect::<Vec<_>>(),
                )
                .boxed(),
                None => source.list(prefix.as_ref()),
            };
            stream::unfold((inner, guard), |(mut inner, guard)| async move {
                inner.next().await.map(|item| {
                    if item.is_err() {
                        guard.metrics.lock().counts.errors += 1;
                    }
                    (item, (inner, guard))
                })
            })
            .boxed()
        })
        .flatten()
        .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        // These benchmarks use flat table directories. Fail if that assumption stops holding.
        let objects: Vec<ObjectMeta> = self.list(prefix).try_collect().await?;
        let base = prefix
            .map(|p| p.as_ref().trim_end_matches('/'))
            .unwrap_or("");
        if objects.iter().any(|o| {
            o.location
                .as_ref()
                .trim_start_matches(base)
                .trim_start_matches('/')
                .contains('/')
        }) {
            return Err(read_only("nested list_with_delimiter"));
        }
        Ok(ListResult {
            common_prefixes: Vec::new(),
            objects,
        })
    }

    async fn put_opts(&self, _: &Path, _: PutPayload, _: PutOptions) -> Result<PutResult> {
        self.metrics.lock().counts.put += 1;
        Err(read_only("PUT"))
    }
    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        Err(read_only("multipart PUT"))
    }
    fn delete_stream(
        &self,
        _: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        stream::once(async { Err(read_only("DELETE")) }).boxed()
    }
    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> Result<()> {
        Err(read_only("COPY"))
    }
}
