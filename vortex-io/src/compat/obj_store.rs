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
        Compat::new(self.inner().put_opts(location, payload, opts)).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        Ok(Box::new(Compat::new(
            Compat::new(self.inner().put_multipart_opts(location, opts)).await?,
        )))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        Compat::new(self.inner().get_opts(location, options)).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        Compat::new(self.inner().get_ranges(location, ranges)).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        Compat::new(self.inner().delete_stream(locations)).boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        Compat::new(self.inner().list(prefix)).boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        Compat::new(self.inner().list_with_offset(prefix, offset)).boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        Compat::new(self.inner().list_with_delimiter(prefix)).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        Compat::new(self.inner().copy_opts(from, to, options)).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
        Compat::new(self.inner().rename_opts(from, to, options)).await
    }
}

#[async_trait]
impl<T: MultipartUpload> MultipartUpload for Compat<T> {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        Compat::new(self.inner_mut().put_part(data)).boxed()
    }

    async fn complete(&mut self) -> Result<PutResult> {
        Compat::new(self.inner_mut().complete()).await
    }

    async fn abort(&mut self) -> Result<()> {
        Compat::new(self.inner_mut().abort()).await
    }
}
