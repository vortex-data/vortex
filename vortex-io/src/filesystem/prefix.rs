// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::filesystem::FileSystemRef;

/// A [`FileSystem`] decorator that roots all operations under a given prefix.
///
/// Paths returned from [`list`](FileSystem::list) are relative to the prefix, and paths
/// passed to [`open_read`](FileSystem::open_read) are automatically prefixed.
#[derive(Debug)]
pub struct PrefixFileSystem {
    inner: FileSystemRef,
    prefix: String,
}

impl PrefixFileSystem {
    pub fn new(inner: FileSystemRef, prefix: String) -> Self {
        // Normalize to always have a trailing slash for clean concatenation.
        let prefix = format!("{}/", prefix.trim_matches('/'));
        Self { inner, prefix }
    }
}

#[async_trait]
impl FileSystem for PrefixFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let full_prefix = format!("{}{}", self.prefix, prefix.trim_start_matches('/'));

        let strip_prefix = self.prefix.clone();
        self.inner
            .list(&full_prefix)
            .map(move |result| {
                result.map(|mut listing| {
                    listing.path = listing
                        .path
                        .strip_prefix(&strip_prefix)
                        .unwrap_or(&listing.path)
                        .to_string();
                    listing
                })
            })
            .boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        let full_path = format!("{}{}", self.prefix, path.trim_start_matches('/'));
        Ok(self.inner.head(&full_path).await?.map(|mut listing| {
            listing.path = listing
                .path
                .strip_prefix(&self.prefix)
                .unwrap_or(&listing.path)
                .to_string();
            listing
        }))
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        self.inner
            .open_read(&format!("{}{}", self.prefix, path.trim_start_matches('/')))
            .await
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        self.inner
            .delete(&format!("{}{}", self.prefix, path.trim_start_matches('/')))
            .await
    }
}

impl dyn FileSystem + 'static {
    /// Create a new filesystem that applies the given prefix to all operations on this filesystem.
    pub fn with_prefix(self: Arc<Self>, prefix: String) -> FileSystemRef {
        Arc::new(PrefixFileSystem::new(self, prefix))
    }
}
