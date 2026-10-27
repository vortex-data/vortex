// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An object store with the latency characteristics of S3, for running workloads in memory.
//!
//! [`LatencyStore`] wraps any store (usually `object_store`'s `InMemory`) and delays requests
//! according to a latency model fitted by `scripts/s3-latency-analyze.py`
//! (`vortex-s3-latency-model/1`): each GET waits a first-byte latency sampled from the measured
//! distribution, then streams its body at `min(per_connection, nic / active_transfers)`: no faster
//! than one connection, and sharing the instance's network bandwidth evenly with every other
//! transfer in flight, the same rule the analysis simulator validates against measured scans. Writes, listings and deletes
//! pass straight through.

use std::fmt;
use std::ops::Range;
use std::path::Path as FsPath;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use object_store::CopyOptions;
use object_store::GetOptions;
use object_store::GetRange;
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
use object_store::Result;
use object_store::path::Path;
use object_store::throttle::ThrottleConfig;
use parking_lot::Mutex;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;
use tokio::time::Instant;
use tokio::time::sleep_until;

/// Bodies are delivered in chunks of this size, so the bandwidth share is re-evaluated as
/// transfers start and finish.
const CHUNK: usize = 64 << 10;

/// An inverse CDF, as `[quantile, seconds]` pairs with increasing quantiles.
#[derive(Clone, Debug, Deserialize)]
pub struct InverseCdf {
    inverse_cdf: Vec<(f64, f64)>,
}

impl InverseCdf {
    fn sample(&self, u: f64) -> f64 {
        for pair in self.inverse_cdf.windows(2) {
            let ((q0, v0), (q1, v1)) = (pair[0], pair[1]);
            if u <= q1 {
                let width = if q1 > q0 { q1 - q0 } else { 1.0 };
                return v0 + (v1 - v0) * (u - q0) / width;
            }
        }
        self.inverse_cdf.last().map_or(0.0, |(_, v)| *v)
    }

    fn quantile(&self, q: f64) -> f64 {
        self.sample(q)
    }
}

/// The `request_model` section of a latency model file.
#[derive(Clone, Debug, Deserialize)]
pub struct LatencyModel {
    first_byte_s: InverseCdf,
    head_s: InverseCdf,
    per_connection_bytes_per_s: f64,
    nic_bytes_per_s: f64,
    #[serde(default)]
    small_read_extra_first_byte_s: std::collections::BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct ModelFile {
    schema: String,
    request_model: LatencyModel,
}

impl LatencyModel {
    pub fn from_file(path: &FsPath) -> anyhow::Result<Self> {
        let file: ModelFile = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(
            file.schema == "vortex-s3-latency-model/1",
            "unsupported latency model schema {}",
            file.schema
        );
        Ok(file.request_model)
    }

    fn extra_first_byte(&self, len: u64) -> f64 {
        self.small_read_extra_first_byte_s
            .get(&len.to_string())
            .copied()
            .unwrap_or(0.0)
    }

    /// The closest fixed-delay configuration of `object_store`'s `ThrottledStore`: the median
    /// first byte per call and one connection's bandwidth per byte.
    pub fn throttle_config(&self) -> ThrottleConfig {
        ThrottleConfig {
            wait_get_per_call: Duration::from_secs_f64(self.first_byte_s.quantile(0.5)),
            wait_get_per_byte: Duration::from_secs_f64(1.0 / self.per_connection_bytes_per_s),
            ..ThrottleConfig::default()
        }
    }
}

struct Shared {
    model: LatencyModel,
    rng: Mutex<StdRng>,
    /// Bodies currently transferring, which share the network bandwidth.
    active: AtomicUsize,
}

impl Shared {
    fn sample(&self, cdf: &InverseCdf) -> Duration {
        let u = self.rng.lock().random::<f64>();
        Duration::from_secs_f64(cdf.sample(u).max(0.0))
    }

    /// When a chunk of `len` bytes arrives on a connection whose previous chunk arrived at
    /// `conn_free_at`.
    ///
    /// Each connection keeps its own timeline rather than restarting from the time it is polled:
    /// the network delivers at line rate whether or not the reader is awake, and timers wake late,
    /// so starting from "now" would add the timer slack to every chunk.
    fn chunk_ready(&self, len: usize, conn_free_at: Instant) -> Instant {
        let active = self.active.load(Ordering::Relaxed).max(1);
        let rate = self
            .model
            .per_connection_bytes_per_s
            .min(self.model.nic_bytes_per_s / active as f64);
        conn_free_at + Duration::from_secs_f64(len as f64 / rate)
    }
}

/// An object store that adds modelled S3 latency to an inner store.
pub struct LatencyStore {
    inner: Arc<dyn ObjectStore>,
    shared: Arc<Shared>,
}

impl LatencyStore {
    pub fn new(inner: Arc<dyn ObjectStore>, model: LatencyModel, seed: u64) -> Self {
        Self {
            inner,
            shared: Arc::new(Shared {
                model,
                rng: Mutex::new(StdRng::seed_from_u64(seed)),
                active: AtomicUsize::new(0),
            }),
        }
    }
}

impl fmt::Debug for LatencyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LatencyStore")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for LatencyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatencyStore({})", self.inner)
    }
}

/// Counts a body as transferring for as long as it is alive.
struct Transfer(Arc<Shared>);

impl Transfer {
    fn new(shared: Arc<Shared>) -> Self {
        shared.active.fetch_add(1, Ordering::Relaxed);
        Self(shared)
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Re-deliver `body` in [`CHUNK`]-sized pieces, each no earlier than its bandwidth share allows.
fn paced(
    body: BoxStream<'static, Result<Bytes>>,
    shared: Arc<Shared>,
) -> BoxStream<'static, Result<Bytes>> {
    let chunks = body.flat_map(|item| {
        let pieces: Vec<Result<Bytes>> = match item {
            Ok(bytes) => (0..bytes.len())
                .step_by(CHUNK)
                .map(|start| Ok(bytes.slice(start..(start + CHUNK).min(bytes.len()))))
                .collect(),
            Err(e) => vec![Err(e)],
        };
        stream::iter(pieces)
    });
    let transfer = Transfer::new(shared);
    stream::unfold(
        (chunks, transfer, Instant::now()),
        |(mut chunks, transfer, conn_free_at)| async move {
            let item = chunks.next().await?;
            let mut free_at = conn_free_at;
            if let Ok(bytes) = &item {
                free_at = transfer.0.chunk_ready(bytes.len(), conn_free_at);
                sleep_until(free_at).await;
            }
            Some((item, (chunks, transfer, free_at)))
        },
    )
    .boxed()
}

#[async_trait]
impl ObjectStore for LatencyStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let start = Instant::now();
        if options.head {
            sleep_until(start + self.shared.sample(&self.shared.model.head_s)).await;
            return self.inner.get_opts(location, options).await;
        }
        let requested = match &options.range {
            Some(GetRange::Bounded(r)) => Some(r.end - r.start),
            Some(GetRange::Suffix(n)) => Some(*n),
            Some(GetRange::Offset(_)) | None => None,
        };
        let extra = requested.map_or(0.0, |n| self.shared.model.extra_first_byte(n));
        let first_byte =
            self.shared.sample(&self.shared.model.first_byte_s) + Duration::from_secs_f64(extra);
        let mut result = self.inner.get_opts(location, options).await?;
        sleep_until(start + first_byte).await;
        let body = match result.payload {
            GetResultPayload::Stream(s) => s,
            GetResultPayload::File(..) => {
                // Only in-memory and remote stores are meant to be wrapped.
                return Err(object_store::Error::NotImplemented {
                    operation: "LatencyStore over a file-backed store".into(),
                    implementer: "LatencyStore".into(),
                });
            }
        };
        result.payload = GetResultPayload::Stream(paced(body, Arc::clone(&self.shared)));
        Ok(result)
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let mut out = Vec::with_capacity(ranges.len());
        for range in ranges {
            let options = GetOptions {
                range: Some(GetRange::Bounded(range.clone())),
                ..Default::default()
            };
            out.push(self.get_opts(location, options).await?.bytes().await?);
        }
        Ok(out)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;

    use super::*;

    fn model(first_byte_s: f64, bytes_per_s: f64, nic_bytes_per_s: f64) -> LatencyModel {
        let fixed = InverseCdf {
            inverse_cdf: vec![(0.0, first_byte_s), (1.0, first_byte_s)],
        };
        LatencyModel {
            first_byte_s: fixed.clone(),
            head_s: fixed,
            per_connection_bytes_per_s: bytes_per_s,
            nic_bytes_per_s,
            small_read_extra_first_byte_s: Default::default(),
        }
    }

    async fn store(model: LatencyModel) -> anyhow::Result<(LatencyStore, Path)> {
        let inner = Arc::new(InMemory::new());
        let path = Path::from("f");
        inner
            .put(&path, PutPayload::from(vec![0u8; 8 << 20]))
            .await?;
        Ok((LatencyStore::new(inner, model, 0), path))
    }

    #[test]
    fn inverse_cdf_interpolates() {
        let cdf = InverseCdf {
            inverse_cdf: vec![(0.0, 1.0), (0.5, 2.0), (1.0, 4.0)],
        };
        assert!((cdf.sample(0.25) - 1.5).abs() < 1e-9);
        assert!((cdf.sample(0.75) - 3.0).abs() < 1e-9);
        assert!((cdf.sample(1.0) - 4.0).abs() < 1e-9);
    }

    #[tokio::test(start_paused = true)]
    async fn get_waits_first_byte_then_bandwidth() -> anyhow::Result<()> {
        let mib = f64::from(1 << 20);
        let (store, path) = store(model(0.025, 100.0 * mib, 1000.0 * mib)).await?;
        let start = Instant::now();
        let bytes = store.get_range(&path, 0..(4 << 20)).await?;
        assert_eq!(bytes.len(), 4 << 20);
        // 25 ms to first byte plus 4 MiB at 100 MiB/s.
        let elapsed = start.elapsed().as_secs_f64();
        assert!((elapsed - 0.065).abs() < 0.002, "{elapsed}");
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_gets_share_the_network() -> anyhow::Result<()> {
        let mib = f64::from(1 << 20);
        // Each connection could do 100 MiB/s, but the link only carries 100 MiB/s in total.
        let (store, path) = store(model(0.0, 100.0 * mib, 100.0 * mib)).await?;
        let start = Instant::now();
        let reads = (0..4).map(|_| store.get_range(&path, 0..(1 << 20)));
        futures::future::try_join_all(reads).await?;
        let elapsed = start.elapsed().as_secs_f64();
        assert!((elapsed - 0.04).abs() < 0.002, "{elapsed}");
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_gets_below_the_cap_run_at_connection_speed() -> anyhow::Result<()> {
        let mib = f64::from(1 << 20);
        // 8 connections at 100 MiB/s each stay well under the 1000 MiB/s link.
        let (store, path) = store(model(0.0, 100.0 * mib, 1000.0 * mib)).await?;
        let store = Arc::new(store);
        let start = Instant::now();
        let reads = (0..8).map(|_| {
            let store = Arc::clone(&store);
            let path = path.clone();
            tokio::spawn(async move { store.get_range(&path, 0..(8 << 20)).await })
        });
        for read in futures::future::join_all(reads).await {
            read??;
        }
        let elapsed = start.elapsed().as_secs_f64();
        assert!((elapsed - 0.08).abs() < 0.002, "{elapsed}");
        Ok(())
    }

    #[test]
    fn throttle_config_uses_median_and_connection_bandwidth() {
        let config = model(0.03, 1e8, 1e9).throttle_config();
        assert_eq!(config.wait_get_per_call, Duration::from_millis(30));
        assert_eq!(config.wait_get_per_byte, Duration::from_nanos(10));
    }
}
