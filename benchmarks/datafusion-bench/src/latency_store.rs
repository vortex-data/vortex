// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An object store that delays every GET, to emulate a remote store over local files.

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::CopyOptions;
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
use object_store::Result;
use object_store::path::Path;

/// Delays every GET by a fixed latency before passing it to `inner`.
///
/// Unlike `object_store::throttle::ThrottledStore`, this keeps a local store's file payloads, so
/// readers still read local files directly once the latency has passed. Its `Display` does not
/// name the inner store, so readers treat it as a remote store.
#[derive(Debug)]
pub struct LatencyStore<T> {
    inner: T,
    latency: Duration,
}

impl<T> LatencyStore<T> {
    pub fn new(inner: T, latency: Duration) -> Self {
        Self { inner, latency }
    }
}

impl<T> fmt::Display for LatencyStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatencyStore({:?})", self.latency)
    }
}

#[async_trait]
impl<T: ObjectStore> ObjectStore for LatencyStore<T> {
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
        tokio::time::sleep(self.latency).await;
        self.inner.get_opts(location, options).await
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
