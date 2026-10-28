// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::future::Shared;
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
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use vortex::error::VortexExpect;

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Model {
    pub ttfb_ms: f64,
    pub request_mib_per_sec: f64,
    pub max_inflight: usize,
    #[serde(default)]
    pub latency_cdf_ms: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cdf_probabilities: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub latency_samples: Vec<LatencySample>,
    #[serde(default)]
    pub latency_bins: Vec<LatencyBin>,
    #[serde(default)]
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct LatencyBin {
    pub max_bytes: u64,
    pub latency_cdf_ms: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cdf_probabilities: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub latency_samples: Vec<LatencySample>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct LatencySample {
    pub header_ms: f64,
    pub body_ms: f64,
    pub cdf_probability: f64,
}

impl Model {
    fn timing_ms(
        &self,
        operation: &str,
        path: &str,
        range: &str,
        sample: u64,
        ordinal: u64,
        bytes: u64,
    ) -> (f64, f64) {
        let (cdf, probabilities, samples) = self
            .latency_bins
            .iter()
            .find(|bin| {
                bytes <= bin.max_bytes
                    && (!bin.latency_cdf_ms.is_empty() || !bin.latency_samples.is_empty())
            })
            .map(|bin| {
                (
                    &bin.latency_cdf_ms,
                    &bin.cdf_probabilities,
                    &bin.latency_samples,
                )
            })
            .unwrap_or((
                &self.latency_cdf_ms,
                &self.cdf_probabilities,
                &self.latency_samples,
            ));
        if operation != "GET" || (cdf.is_empty() && samples.is_empty()) {
            return (self.ttfb_ms, 0.0);
        }
        // Stable request keys make draws independent of the order concurrent tasks are polled.
        let mut hash = 0xcbf29ce484222325u64;
        for byte in operation
            .bytes()
            .chain([0])
            .chain(path.bytes())
            .chain([0])
            .chain(range.bytes())
            .chain(self.seed.to_le_bytes())
            .chain(sample.to_le_bytes())
            .chain(ordinal.to_le_bytes())
        {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        // Mix the FNV result so nearby keys do not select nearby latency quantiles.
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
        hash ^= hash >> 31;
        let uniform = (hash >> 11) as f64 / (1u64 << 53) as f64;
        if !samples.is_empty() {
            let index = samples.partition_point(|sample| sample.cdf_probability <= uniform);
            return (samples[index].header_ms, samples[index].body_ms);
        }
        let index = if probabilities.is_empty() {
            usize::try_from(hash % cdf.len() as u64)
                .vortex_expect("quantile index is bounded by the vector length")
        } else {
            probabilities.partition_point(|probability| *probability <= uniform)
        };
        (cdf[index], 0.0)
    }

    #[cfg(test)]
    fn delay_ms(
        &self,
        operation: &str,
        path: &str,
        range: &str,
        sample: u64,
        ordinal: u64,
        bytes: u64,
    ) -> f64 {
        self.timing_ms(operation, path, range, sample, ordinal, bytes)
            .0
    }
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
    pub get_calls: u64,
    pub tail_cache_hits: u64,
    #[serde(skip)]
    inflight: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct Request {
    operation: &'static str,
    path: String,
    requested_range: String,
    bytes: u64,
    start_ms: f64,
    response_ms: Option<f64>,
    first_byte_ms: Option<f64>,
    duration_ms: f64,
    inflight_at_start: usize,
    model_delay_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_body_delay_ms: Option<f64>,
}

#[derive(Debug, Default)]
struct Metrics {
    counts: Counts,
    requests: Vec<Request>,
    sample: u64,
    ordinals: BTreeMap<(String, String, String), u64>,
    started: Option<Instant>,
}

#[derive(Debug)]
struct Guard {
    metrics: Arc<Mutex<Metrics>>,
    start: Instant,
    request: Request,
    delay_ms: f64,
    body_delay_ms: f64,
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
            start_ms: self.request.start_ms,
            response_ms: self.request.response_ms,
            first_byte_ms: self.request.first_byte_ms,
            duration_ms: self.request.duration_ms,
            inflight_at_start: self.request.inflight_at_start,
            model_delay_ms: self.request.model_delay_ms,
            model_body_delay_ms: self.request.model_body_delay_ms,
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

#[derive(Serialize)]
pub(super) struct ObjectIdentity {
    path: String,
    size: u64,
    e_tag: Option<String>,
    version: Option<String>,
}

pub(super) fn manifest(objects: &Snapshot) -> Vec<ObjectIdentity> {
    objects
        .iter()
        .map(|(path, object)| ObjectIdentity {
            path: path.to_string(),
            size: object.meta.size,
            e_tag: object.meta.e_tag.clone(),
            version: object.meta.version.clone(),
        })
        .collect()
}

#[derive(Serialize)]
pub(super) struct SourceWarmup {
    requests: u64,
    bytes: u64,
    seconds: f64,
    concurrency: usize,
}

pub(super) async fn warm_source(
    source: &dyn ObjectStore,
    objects: &Snapshot,
    seconds: f64,
) -> anyhow::Result<SourceWarmup> {
    const CONCURRENCY: usize = 32;
    let start = Instant::now();
    let deadline = start + Duration::from_secs_f64(seconds);
    let keys = objects
        .iter()
        .map(|(path, object)| (path, object.meta.size))
        .collect::<Vec<_>>();
    anyhow::ensure!(!keys.is_empty(), "No source objects to warm");
    let counts = stream::iter(0..CONCURRENCY)
        .map(|worker| {
            let keys = &keys;
            async move {
                let (mut requests, mut bytes) = (0u64, 0u64);
                let mut cursor = worker;
                while Instant::now() < deadline {
                    let (path, size) = keys[cursor % keys.len()];
                    let length = size.min(65536);
                    let offset = (cursor as u64).wrapping_mul(131071) % (size - length + 1);
                    bytes += source.get_range(path, offset..offset + length).await?.len() as u64;
                    requests += 1;
                    cursor += CONCURRENCY;
                    tokio::task::yield_now().await;
                }
                Ok::<_, object_store::Error>((requests, bytes))
            }
        })
        .buffer_unordered(CONCURRENCY)
        .try_collect::<Vec<_>>()
        .await?;
    Ok(SourceWarmup {
        requests: counts.iter().map(|(requests, _)| requests).sum(),
        bytes: counts.iter().map(|(_, bytes)| bytes).sum(),
        seconds: start.elapsed().as_secs_f64(),
        concurrency: CONCURRENCY,
    })
}

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

#[derive(Clone)]
pub(super) struct Store {
    source: Arc<dyn ObjectStore>,
    snapshot: Option<Snapshot>,
    model: Option<Arc<Model>>,
    admission: Option<Arc<Semaphore>>,
    metrics: Arc<Mutex<Metrics>>,
    cache_tail: bool,
    tail_reads: TailCache,
    sampling_prefix: Arc<str>,
}

pub(super) const TAIL_CACHE_BYTES: u64 = 4 * 1024 * 1024;
type TailKey = (Path, u64, u64);
type TailCache = Arc<Mutex<BTreeMap<TailKey, TailRead>>>;
type TailRead =
    Shared<BoxFuture<'static, std::result::Result<CachedGet, Arc<object_store::Error>>>>;

#[derive(Clone)]
struct CachedGet {
    meta: ObjectMeta,
    attributes: Attributes,
    range: Range<u64>,
    bytes: Bytes,
}

impl CachedGet {
    fn response(self) -> GetResult {
        GetResult {
            meta: self.meta,
            attributes: self.attributes,
            range: self.range,
            payload: GetResultPayload::Stream(stream::once(async move { Ok(self.bytes) }).boxed()),
        }
    }
}

impl fmt::Debug for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Store")
            .field("model", &self.model)
            .field("cache_tail", &self.cache_tail)
            .finish_non_exhaustive()
    }
}

impl Store {
    pub fn new(
        source: Arc<dyn ObjectStore>,
        snapshot: Option<Snapshot>,
        model: Option<Model>,
        cache_tail: bool,
    ) -> Self {
        let admission = model
            .as_ref()
            .filter(|m| m.max_inflight > 0)
            .map(|m| Arc::new(Semaphore::new(m.max_inflight)));
        let sampling_prefix = snapshot
            .as_ref()
            .and_then(|objects| objects.keys().next())
            .and_then(|path| path.as_ref().rsplit_once('/'))
            .map(|(prefix, _)| format!("{prefix}/"))
            .unwrap_or_default();
        Self {
            source,
            snapshot,
            model: model.map(Arc::new),
            admission,
            metrics: Arc::default(),
            cache_tail,
            tail_reads: Arc::default(),
            sampling_prefix: sampling_prefix.into(),
        }
    }

    pub fn reset(&self) -> anyhow::Result<()> {
        self.tail_reads.lock().clear();
        let mut metrics = self.metrics.lock();
        anyhow::ensure!(
            metrics.counts.inflight == 0,
            "Requests still in flight at measurement boundary"
        );
        let sample = metrics.sample + 1;
        *metrics = Metrics {
            sample,
            started: Some(Instant::now()),
            ..Metrics::default()
        };
        Ok(())
    }

    pub fn take(&self) -> anyhow::Result<(Counts, Vec<Request>)> {
        self.tail_reads.lock().clear();
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

    async fn begin(
        &self,
        operation: &'static str,
        path: String,
        range: String,
        bytes: u64,
    ) -> Result<Guard> {
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
            let start_ms = metrics
                .started
                .get_or_insert_with(Instant::now)
                .elapsed()
                .as_secs_f64()
                * 1000.0;
            let timing = self.model.as_ref().map(|model| {
                let sample = metrics.sample;
                let ordinal = metrics
                    .ordinals
                    .entry((operation.to_string(), path.clone(), range.clone()))
                    .or_default();
                let logical_path = path
                    .strip_prefix(self.sampling_prefix.as_ref())
                    .unwrap_or(&path);
                let delay =
                    model.timing_ms(operation, logical_path, &range, sample, *ordinal, bytes);
                *ordinal += 1;
                delay
            });
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
                delay_ms: timing.map_or(0.0, |(header, _)| header),
                body_delay_ms: timing.map_or(0.0, |(_, body)| body),
                request: Request {
                    operation,
                    path,
                    requested_range: range,
                    bytes: 0,
                    start_ms,
                    response_ms: None,
                    first_byte_ms: None,
                    duration_ms: 0.0,
                    inflight_at_start: metrics.counts.inflight,
                    model_delay_ms: timing.map(|(header, _)| header),
                    model_body_delay_ms: timing.map(|(_, body)| body),
                },
                _permit: permit,
            }
        };
        if guard.delay_ms > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(guard.delay_ms / 1000.0)).await;
        }
        Ok(guard)
    }

    fn ram_get(
        &self,
        location: &Path,
        options: &GetOptions,
        objects: &Snapshot,
        request_start: Instant,
        delay_ms: f64,
        body_delay_ms: f64,
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
        let rate = self.model.as_ref().map(|m| m.request_mib_per_sec);
        let start = tokio::time::Instant::from_std(request_start);
        let payload = stream::try_unfold((data, 0usize), move |(data, offset)| async move {
            if offset == data.len() {
                return Ok(None);
            }
            let end = (offset + 256 * 1024).min(data.len());
            if let Some(rate) = rate {
                // Absolute deadlines avoid adding a timer tick for each response chunk.
                let deadline = start
                    + Duration::from_secs_f64(
                        (delay_ms + body_delay_ms) / 1000.0 + end as f64 / (rate * 1048576.0),
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

    fn tail_key(&self, location: &Path, options: &GetOptions) -> Option<TailKey> {
        if !self.cache_tail
            || options.head
            || options.version.is_some()
            || options.if_match.is_some()
            || options.if_none_match.is_some()
            || options.if_modified_since.is_some()
            || options.if_unmodified_since.is_some()
        {
            return None;
        }
        let size = self.snapshot.as_ref()?.get(location)?.meta.size;
        let range = options.range.as_ref()?.as_range(size).ok()?;
        // Keep this cache bounded to the immutable file tail, including large Vortex footers.
        (range.start >= size.saturating_sub(TAIL_CACHE_BYTES) || range.end == size)
            .then(|| (location.clone(), range.start, range.end))
    }

    async fn fetch(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let head = options.head;
        let bytes = self
            .snapshot
            .as_ref()
            .and_then(|objects| objects.get(location))
            .and_then(|object| {
                options
                    .range
                    .as_ref()
                    .map_or(Some(object.meta.size), |range| {
                        range
                            .as_range(object.meta.size)
                            .ok()
                            .map(|range| range.end - range.start)
                    })
            })
            .unwrap_or(0);
        let mut guard = self
            .begin(
                if head { "HEAD" } else { "GET" },
                location.to_string(),
                format!("{:?}", options.range),
                bytes,
            )
            .await?;
        let result = match self.snapshot.as_ref().filter(|_| self.model.is_some()) {
            Some(objects) => self.ram_get(
                location,
                &options,
                objects,
                guard.start,
                guard.delay_ms,
                guard.body_delay_ms,
            ),
            None => self.source.get_opts(location, options).await,
        };
        guard.request.response_ms = Some(guard.start.elapsed().as_secs_f64() * 1000.0);
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
        if !options.head {
            self.metrics.lock().counts.get_calls += 1;
        }
        let key = self.tail_key(location, &options);
        let Some(key) = key else {
            return self.fetch(location, options).await;
        };
        let (future, hit) = {
            let mut reads = self.tail_reads.lock();
            if let Some(future) = reads.get(&key) {
                (future.clone(), true)
            } else {
                let store = self.clone();
                let location = location.clone();
                let future = async move {
                    let response = store.fetch(&location, options).await.map_err(Arc::new)?;
                    let meta = response.meta.clone();
                    let attributes = response.attributes.clone();
                    let range = response.range.clone();
                    let bytes = response.bytes().await.map_err(Arc::new)?;
                    Ok(CachedGet {
                        meta,
                        attributes,
                        range,
                        bytes,
                    })
                }
                .boxed()
                .shared();
                reads.insert(key, future.clone());
                (future, false)
            }
        };
        if hit {
            self.metrics.lock().counts.tail_cache_hits += 1;
        }
        future
            .await
            .map(CachedGet::response)
            .map_err(|error| object_store::Error::Generic {
                store: "s3-validation-tail-cache",
                source: error.to_string().into(),
            })
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        let prefix = prefix.cloned();
        let store = self.clone();
        let source = Arc::clone(&self.source);
        let objects = self.snapshot.clone().filter(|_| self.model.is_some());
        stream::once(async move {
            let guard = match store
                .begin(
                    "LIST",
                    prefix.as_ref().map(ToString::to_string).unwrap_or_default(),
                    String::new(),
                    0,
                )
                .await
            {
                Ok(guard) => guard,
                Err(error) => return stream::once(async { Err(error) }).boxed(),
            };
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
