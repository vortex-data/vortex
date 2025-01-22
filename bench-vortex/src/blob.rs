use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOpts,
    PutOptions, PutPayload, PutResult, Result as OSResult,
};
use rand::prelude::Distribution as _;
use rand::thread_rng;
use zipf::ZipfDistribution;

#[derive(Debug)]
pub struct SlowObjectStore {
    inner: Arc<dyn ObjectStore>,
    zipf: ZipfDistribution,
}

impl SlowObjectStore {
    pub fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self {
            inner: object_store,
            zipf: ZipfDistribution::new(1000, 1.4).unwrap(),
        }
    }

    /// Injects an artificial wait of somewhere between 20ms to a full second.
    /// max wait is 1sec, but p95 is around 200ms (roughly the same as AnyBlob paper).
    ///
    /// We always wait at least 30ms, which seems to be the rough baseline for object store access.
    async fn wait(&self) {
        let duration = self.zipf.sample(&mut thread_rng()) + 30;
        tokio::time::sleep(Duration::from_millis(duration as u64)).await;
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
        self.wait().await;
        self.inner.get_opts(location, options).await
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.wait().await;
        self.inner.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'_, OSResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.wait().await;
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
