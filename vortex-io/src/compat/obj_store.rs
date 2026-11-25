// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
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
use object_store::Result;
use object_store::UploadPart;
use object_store::path::Path;
use smol::future::FutureExt;
use smol::stream::StreamExt;

use crate::compat::Compat;

impl<T: ObjectStore> Display for Compat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Compat<{}>", self.inner())
    }
}

#[async_trait]
impl<T: ObjectStore> ObjectStore for Compat<T> {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        let fut = self.enter(|inner| inner.put_opts(location, payload, opts));
        Compat::new(fut).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        let fut = self.enter(|inner| inner.put_multipart_opts(location, opts));
        Ok(Box::new(Compat::new(Compat::new(fut).await?)))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let fut = self.enter(|inner| inner.get_opts(location, options));
        let mut result = Compat::new(fut).await?;
        result.payload = match result.payload {
            GetResultPayload::Stream(stream) => {
                GetResultPayload::Stream(Compat::new(stream).boxed())
            }
            #[cfg(not(target_arch = "wasm32"))]
            GetResultPayload::File(file, path) => GetResultPayload::File(file, path),
        };
        Ok(result)
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let fut = self.enter(|inner| inner.get_ranges(location, ranges));
        Compat::new(fut).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        Compat::new(self.enter(|inner| inner.delete_stream(locations))).boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        Compat::new(self.enter(|inner| inner.list(prefix))).boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        Compat::new(self.enter(|inner| inner.list_with_offset(prefix, offset))).boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        let fut = self.enter(|inner| inner.list_with_delimiter(prefix));
        Compat::new(fut).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        let fut = self.enter(|inner| inner.copy_opts(from, to, options));
        Compat::new(fut).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
        let fut = self.enter(|inner| inner.rename_opts(from, to, options));
        Compat::new(fut).await
    }
}

#[async_trait]
impl<T: MultipartUpload> MultipartUpload for Compat<T> {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        Compat::new(self.enter_mut(|inner| inner.put_part(data))).boxed()
    }

    async fn complete(&mut self) -> Result<PutResult> {
        let fut = self.enter_mut(|inner| inner.complete());
        Compat::new(fut).await
    }

    async fn abort(&mut self) -> Result<()> {
        let fut = self.enter_mut(|inner| inner.abort());
        Compat::new(fut).await
    }
}
