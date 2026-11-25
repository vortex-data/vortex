// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::compat::Compat;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;

/// Compatibility adapter for `FileSystem` implementations that are based on Tokio.
#[deny(clippy::missing_trait_methods)]
#[async_trait]
impl<F: FileSystem> FileSystem for Compat<F> {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        Compat::new(self.enter(|inner| inner.list(prefix))).boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        let fut = self.enter(|inner| inner.head(path));
        Compat::new(fut).await
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let fut = self.enter(|inner| inner.open_read(path));
        let read_at = Compat::new(fut).await?;
        Ok(Arc::new(Compat::new(read_at)))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        let fut = self.enter(|inner| inner.delete(path));
        Compat::new(fut).await
    }
}
