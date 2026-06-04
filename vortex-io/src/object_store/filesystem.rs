// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FileSystem`] implementation backed by an [`ObjectStore`].

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::VortexReadAt;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::object_store::ObjectStoreReadAt;
use crate::runtime::Handle;

/// A [`FileSystem`] backed by an [`ObjectStore`].
///
// TODO(ngates): we could consider spawning a driver task inside this file system such that we can
//  apply concurrency limits to the overall object store, rather than on a per-file basis.
pub struct ObjectStoreFileSystem {
    store: Arc<dyn ObjectStore>,
    handle: Handle,
}

impl Debug for ObjectStoreFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStoreFileSystem")
            .field("store", &self.store)
            .finish()
    }
}

impl ObjectStoreFileSystem {
    /// Create a new filesystem backed by the given object store and runtime handle.
    pub fn new(store: Arc<dyn ObjectStore>, handle: Handle) -> Self {
        Self { store, handle }
    }

    /// Create a new filesystem backed by a local file system object store and the given runtime
    /// handle.
    pub fn local(handle: Handle) -> Self {
        Self::new(
            Arc::new(object_store::local::LocalFileSystem::new()),
            handle,
        )
    }
}

#[async_trait]
impl FileSystem for ObjectStoreFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let path = if prefix.is_empty() {
            None
        } else {
            Some(Path::from(prefix))
        };
        self.store
            .list(path.as_ref())
            .map(|result| {
                result
                    .map(|meta| FileListing {
                        path: meta.location.to_string(),
                        size: Some(meta.size),
                    })
                    .map_err(Into::into)
            })
            .boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        // `head` issues a single metadata lookup (e.g. an S3 HEAD) for the exact key, unlike
        // `list`, which enumerates by path-segment prefix and never returns the key itself.
        match self.store.head(&Path::from(path)).await {
            Ok(meta) => Ok(Some(FileListing {
                path: meta.location.to_string(),
                size: Some(meta.size),
            })),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(ObjectStoreReadAt::new(
            Arc::clone(&self.store),
            path.into(),
            self.handle.clone(),
        )))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        self.store
            .delete(
                &Path::from_url_path(path)
                    .map_err(|_| vortex_err!("invalid path for url {path}"))?,
            )
            .await?;
        Ok(())
    }
}

// Exercises the fix against a real object store, whose `list` excludes the exact-path match.
// `Handle::find` only resolves a runtime under the `tokio` feature, so gate these tests on it.
#[cfg(test)]
#[cfg(feature = "tokio")]
mod tests {
    use futures::TryStreamExt;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;

    use super::*;
    use crate::filesystem::FileSystem;
    use crate::runtime::Handle;

    /// Build an [`ObjectStoreFileSystem`] over an in-memory store seeded with `(path, size)` files.
    async fn memory_fs(files: &[(&str, usize)]) -> VortexResult<ObjectStoreFileSystem> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        for &(path, size) in files {
            store.put(&Path::from(path), vec![0u8; size].into()).await?;
        }
        let handle = Handle::find().expect("tokio runtime available within #[tokio::test]");
        Ok(ObjectStoreFileSystem::new(store, handle))
    }

    /// Regression test for #6599: globbing an exact path that exists must return that one file.
    /// `ObjectStore::list` never yields the prefix itself, so this would return nothing if the
    /// exact-path branch used `list`.
    #[tokio::test]
    async fn test_glob_exact_existing_path() -> VortexResult<()> {
        let fs = memory_fs(&[("data/file.vortex", 1024)]).await?;
        let fs_dyn: &dyn FileSystem = &fs;
        let results: Vec<FileListing> = fs_dyn.glob("data/file.vortex")?.try_collect().await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "data/file.vortex");
        assert_eq!(results[0].size, Some(1024));
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_exact_missing_path_is_empty() -> VortexResult<()> {
        let fs = memory_fs(&[("data/other.vortex", 1)]).await?;
        let fs_dyn: &dyn FileSystem = &fs;
        let results: Vec<FileListing> = fs_dyn.glob("data/missing.vortex")?.try_collect().await?;
        assert!(results.is_empty());
        Ok(())
    }

    /// `list("foo.vortex")` would surface the prefix-sibling `foo.vortex.backup`; `head` does not.
    #[tokio::test]
    async fn test_glob_exact_path_ignores_prefix_siblings() -> VortexResult<()> {
        let fs = memory_fs(&[("foo.vortex", 10), ("foo.vortex.backup", 20)]).await?;
        let fs_dyn: &dyn FileSystem = &fs;
        let results: Vec<FileListing> = fs_dyn.glob("foo.vortex")?.try_collect().await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "foo.vortex");
        assert_eq!(results[0].size, Some(10));
        Ok(())
    }

    #[tokio::test]
    async fn test_head_existing_and_missing() -> VortexResult<()> {
        let fs = memory_fs(&[("a/b.vortex", 7)]).await?;
        assert_eq!(
            fs.head("a/b.vortex").await?,
            Some(FileListing {
                path: "a/b.vortex".to_string(),
                size: Some(7),
            })
        );
        assert_eq!(fs.head("a/missing.vortex").await?, None);
        Ok(())
    }
}
