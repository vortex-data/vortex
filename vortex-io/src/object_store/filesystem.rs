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

/// Convert a literal filesystem path into an object-store [`Path`] (key).
///
/// Object stores key their objects *literally*: a file named `a~b.vortex` has the key
/// `a~b.vortex`, and `LocalFileSystem` likewise surfaces real filenames verbatim. [`Path::parse`]
/// preserves those characters, whereas [`Path::from`] percent-encodes `~`, `%`, `[`, `]`, `#`,
/// etc. — turning `a~b.vortex` into the key `a%7Eb.vortex`, which no real object has. Using
/// `parse` keeps inputs and the keys returned by [`list`](FileSystem::list) on the same literal
/// representation, so a path from `list`/`head` round-trips back through `open_read` unchanged.
///
/// `parse` rejects empty, `.`, and `..` segments; for those we fall back to [`Path::from`], which
/// normalizes them (this never applies to a key `list` produced, so it cannot break a round-trip).
fn to_object_path(path: &str) -> Path {
    Path::parse(path).unwrap_or_else(|_| Path::from(path))
}

fn listing_from_meta(location: &Path, size: u64) -> FileListing {
    FileListing {
        path: location.to_string(),
        size: Some(size),
    }
}

#[async_trait]
impl FileSystem for ObjectStoreFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let path = if prefix.is_empty() {
            None
        } else {
            Some(to_object_path(prefix))
        };
        self.store
            .list(path.as_ref())
            .map(|result| {
                result
                    .map(|meta| listing_from_meta(&meta.location, meta.size))
                    .map_err(Into::into)
            })
            .boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        // `head` issues a single metadata lookup (e.g. an S3 HEAD) for the exact key, unlike
        // `list`, which enumerates by path-segment prefix and never returns the key itself.
        match self.store.head(&to_object_path(path)).await {
            Ok(meta) => Ok(Some(listing_from_meta(&meta.location, meta.size))),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(ObjectStoreReadAt::new(
            Arc::clone(&self.store),
            to_object_path(path),
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
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use rstest::rstest;

    use super::*;
    use crate::filesystem::FileSystem;
    use crate::runtime::Handle;

    /// Build an [`ObjectStoreFileSystem`] over an in-memory store seeded with `(path, size)` files.
    ///
    /// Keys are written with [`to_object_path`] so the store holds the same literal keys a real
    /// backend would (e.g. `a~b.vortex`, not the percent-encoded `a%7Eb.vortex`).
    async fn memory_fs(files: &[(&str, usize)]) -> VortexResult<ObjectStoreFileSystem> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        for &(path, size) in files {
            store
                .put(&to_object_path(path), vec![0u8; size].into())
                .await?;
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

    /// Paths containing characters that `object_store` percent-encodes (`~ % [ ] # { } ^`, …)
    /// must round-trip on a single literal-path convention: `head` returns the literal key, and
    /// that exact string must reopen the file via `open_read` (what multi-file scan does).
    /// Previously these were converted with `Path::from`, which encoded them into a key no real
    /// object has, so the file was silently lost.
    #[tokio::test]
    #[rstest]
    #[case::tilde("dir/a~b.vortex")]
    #[case::percent("dir/a%20b.vortex")]
    #[case::brackets("dir/a[1].vortex")]
    #[case::hash("dir/a#b.vortex")]
    #[case::braces("dir/a{x}.vortex")]
    #[case::caret("dir/a^b.vortex")]
    #[case::backslash_tilde("dir/a\\~b.vortex")]
    #[case::space("dir/a b.vortex")]
    async fn test_head_open_read_round_trip_special_chars(#[case] path: &str) -> VortexResult<()> {
        let fs = memory_fs(&[(path, 5)]).await?;
        // `head` returns the literal key, matching the caller's input.
        assert_eq!(
            fs.head(path).await?,
            Some(FileListing {
                path: path.to_string(),
                size: Some(5),
            })
        );
        // and that literal path reopens the file (`size()` issues a `head` under the hood).
        assert_eq!(fs.open_read(path).await?.size().await?, 5);
        Ok(())
    }

    /// Glob (both branches) over paths with encoded—but not glob-metacharacter—characters returns
    /// the literal path, which must reopen the file. (`*`, `?`, `[` are excluded here: they make
    /// the input a pattern rather than an exact path.)
    #[tokio::test]
    #[rstest]
    #[case::tilde("dir/a~b.vortex")]
    #[case::percent("dir/a%20b.vortex")]
    #[case::hash("dir/a#b.vortex")]
    #[case::backslash_tilde("dir/a\\~b.vortex")]
    #[case::space("dir/a b.vortex")]
    #[case::plain("dir/plain.vortex")]
    async fn test_glob_round_trip_special_chars(#[case] path: &str) -> VortexResult<()> {
        let fs = memory_fs(&[(path, 5)]).await?;
        let fs_dyn: &dyn FileSystem = &fs;
        let expected = FileListing {
            path: path.to_string(),
            size: Some(5),
        };

        // Exact-path glob returns the literal path.
        let exact: Vec<FileListing> = fs_dyn.glob(path)?.try_collect().await?;
        assert_eq!(exact, vec![expected.clone()]);

        // Wildcard glob over the directory lists the same literal path (not an encoded one).
        let wild: Vec<FileListing> = fs_dyn.glob("dir/*.vortex")?.try_collect().await?;
        assert!(
            wild.contains(&expected),
            "wildcard glob should list {path:?}, got {wild:?}"
        );

        // The returned path reopens the file.
        assert_eq!(fs.open_read(path).await?.size().await?, 5);
        Ok(())
    }

    /// The same round-trip against the real local-filesystem backend: a `~` in an on-disk filename
    /// must be addressed literally (converting it with `Path::from` would percent-encode it and
    /// miss the file). Confirms the literal-path convention holds across backends, not just for the
    /// in-memory store.
    #[tokio::test]
    async fn test_local_filesystem_special_char_round_trip() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("a~b.vortex"), [0u8; 5])?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?) as Arc<dyn ObjectStore>;
        let fs = ObjectStoreFileSystem::new(store, Handle::find().expect("tokio runtime"));
        let fs_dyn: &dyn FileSystem = &fs;
        let expected = FileListing {
            path: "a~b.vortex".to_string(),
            size: Some(5),
        };

        let wild: Vec<FileListing> = fs_dyn.glob("*.vortex")?.try_collect().await?;
        assert!(wild.contains(&expected), "wildcard glob got {wild:?}");

        let exact: Vec<FileListing> = fs_dyn.glob("a~b.vortex")?.try_collect().await?;
        assert_eq!(exact, vec![expected]);

        assert_eq!(fs.open_read("a~b.vortex").await?.size().await?, 5);
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
