use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::execution::object_store::{DefaultObjectStoreRegistry, ObjectStoreRegistry};
use futures::stream::BoxStream;
use governor::{DefaultDirectRateLimiter, Quota};
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOpts,
    PutOptions, PutPayload, PutResult, Result as OSResult,
};
use rand::distr::Distribution;
use rand::rng;
use rand_distr::LogNormal;
use reqwest::Url;
use vortex::error::VortexUnwrap;

#[derive(Debug)]
pub struct SlowObjectStore {
    inner: Arc<dyn ObjectStore>,
    distribution: LogNormal<f32>,
    rate_limiter: Arc<DefaultDirectRateLimiter>,
}

#[derive(Debug)]
pub struct SlowObjectStoreRegistry {
    pub inner: Arc<dyn ObjectStoreRegistry>,
}

impl Default for SlowObjectStoreRegistry {
    fn default() -> Self {
        Self {
            inner: Arc::new(DefaultObjectStoreRegistry::default()),
        }
    }
}

impl ObjectStoreRegistry for SlowObjectStoreRegistry {
    fn register_store(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        self.inner
            .register_store(url, Arc::new(SlowObjectStore::new(store)))
    }

    fn get_store(&self, url: &Url) -> datafusion_common::Result<Arc<dyn ObjectStore>> {
        self.inner.get_store(url)
    }
}

impl SlowObjectStore {
    pub fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self {
            inner: object_store,
            distribution: LogNormal::new(4.7, 0.5).unwrap(), //p50 ~ 100, p95 ~ 250 and p100 ~ 600
            rate_limiter: Arc::new(DefaultDirectRateLimiter::direct(Quota::per_second(
                (2_u32 << 30).try_into().unwrap(), // 1GB/s
            ))),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn wait_time(&self) -> Duration {
        let duration = (self.distribution.sample(&mut rng()) as u64).clamp(30, 1_000);
        Duration::from_millis(duration)
    }

    /// Injects an artificial sleep of somewhere between 30ms to a full second.
    // wait times will p50 ~ 100ms, p95 ~ 250ms and p100 ~ 600ms.
    ///
    /// We always wait at least 30ms, which seems to be the rough baseline for object store access.
    async fn wait(&self) {
        tokio::time::sleep(self.wait_time()).await;
    }

    /// Same as `Self::wait`, but with additional wait time according to the size of the response.
    async fn wait_with_size(&self, size: usize) {
        let base_wait_time = self.wait_time();
        let additional_ms = size.div_ceil(65536) as u64; // 64KB, roughly median throughput on S3
        let total_time = base_wait_time + Duration::from_millis(additional_ms);
        tokio::time::sleep(total_time).await;
    }
}

impl std::fmt::Display for SlowObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlowObjectStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for SlowObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.wait().await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOpts,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.wait().await;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        // Ideally, we would tune `wait` here for the actual if it exists in options.range
        let r = self.inner.get_opts(location, options).await?;

        self.wait_with_size(r.meta.size).await;
        self.rate_limiter
            .until_n_ready(
                u32::try_from(r.meta.size)
                    .vortex_unwrap()
                    .try_into()
                    .vortex_unwrap(),
            )
            .await
            .unwrap();

        Ok(r)
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.wait().await;
        self.inner.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'_, OSResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        // This just makes listing super slow and its not really the part we're interested in
        // self.wait().await;
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.wait().await;
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.wait().await;
        self.copy_if_not_exists(from, to).await
    }
}
