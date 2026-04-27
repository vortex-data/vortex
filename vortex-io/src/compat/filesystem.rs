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
        Compat::new(self.inner().list(prefix)).boxed()
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let read_at = Compat::new(self.inner().open_read(path)).await?;
        Ok(Arc::new(Compat::new(read_at)))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        Compat::new(self.inner().delete(path)).await
    }
}
