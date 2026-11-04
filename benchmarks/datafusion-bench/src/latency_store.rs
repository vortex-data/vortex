// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! EXPERIMENT: an [`ObjectStore`] wrapper that adds a fixed delay to every GET, to emulate object
//! storage round trips on local files. `ThrottledStore` cannot wrap `LocalFileSystem` because it
//! does not support file-backed GET payloads.

use std::fmt;
use std::ops::Range;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
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

#[derive(Debug)]
pub struct LatencyStore<T> {
    inner: T,
    get_latency: Duration,
}

impl<T> LatencyStore<T> {
    pub fn new(inner: T, get_latency: Duration) -> Self {
        Self { inner, get_latency }
    }
}

impl<T: fmt::Display> fmt::Display for LatencyStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LatencyStore({}, {}ms)",
            self.inner,
            self.get_latency.as_millis()
        )
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
        tokio::time::sleep(self.get_latency).await;
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        tokio::time::sleep(self.get_latency).await;
        self.inner.get_ranges(location, ranges).await
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
