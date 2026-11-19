// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FileSystem`] implementation backed by an [`ObjectStore`].

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::filesystem::FileVersion;
use crate::object_store::ObjectStoreReadAt;
use crate::object_store::object_path_from_literal;
use crate::runtime::Handle;

/// A [`FileSystem`] backed by an [`ObjectStore`].
// TODO(ngates): we could consider spawning a driver task inside this file system such that we can
//  apply concurrency limits to the overall object store, rather than on a per-file basis.
pub struct ObjectStoreFileSystem {
    store: Arc<dyn ObjectStore>,
    handle: Handle,
    cache_namespace: u64,
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
        static NEXT_CACHE_NAMESPACE: AtomicU64 = AtomicU64::new(0);

        Self {
            store,
            handle,
            cache_namespace: NEXT_CACHE_NAMESPACE.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Create a new filesystem backed by a local file system object store and the given runtime
    /// handle.
    #[deprecated = "use StdFileSystem::new"]
    pub fn local(handle: Handle) -> Self {
        Self::new(
            Arc::new(object_store::local::LocalFileSystem::new()),
            handle,
        )
    }
}

fn listing_from_meta(meta: &ObjectMeta) -> FileListing {
    FileListing {
        path: meta.location.to_string(),
        size: Some(meta.size),
        // S3 uses the literal "null" for the mutable current version when versioning is suspended.
        version: meta
            .version
            .as_deref()
            .filter(|version| *version != "null")
            .map(FileVersion::new),
    }
}

#[async_trait]
impl FileSystem for ObjectStoreFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let path = if prefix.is_empty() {
            None
        } else {
            Some(object_path_from_literal(prefix))
        };
        self.store
            .list(path.as_ref())
            .map(|result| {
                result
                    .map(|meta| listing_from_meta(&meta))
                    .map_err(Into::into)
            })
            .boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        // `head` issues a single metadata lookup (e.g. an S3 HEAD) for the exact key, unlike
        // `list`, which enumerates by path-segment prefix and never returns the key itself.
        match self.store.head(&object_path_from_literal(path)).await {
            Ok(meta) => Ok(Some(listing_from_meta(&meta))),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn open_read(&self, file: &FileListing) -> VortexResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(
            ObjectStoreReadAt::new(
                Arc::clone(&self.store),
                object_path_from_literal(&file.path),
                self.handle.clone(),
            )
            .with_listing_metadata(file),
        ))
    }

    fn cache_key(&self, file: &FileListing) -> Option<String> {
        Some(format!(
            "object-store:{}:{}",
            self.cache_namespace, file.path
        ))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        self.store.delete(&object_path_from_literal(path)).await?;
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
    use crate::filesystem::FileSystemRef;
    use crate::runtime::Handle;

    /// Build an [`ObjectStoreFileSystem`] over an in-memory store seeded with `(path, size)` files.
    ///
    /// Keys are written with [`object_path_from_literal`] so the store holds the same literal keys
    /// a real backend would (e.g. `a~b.vortex`, not the percent-encoded `a%7Eb.vortex`).
    async fn memory_fs(files: &[(&str, usize)]) -> VortexResult<ObjectStoreFileSystem> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        for &(path, size) in files {
            store
                .put(&object_path_from_literal(path), vec![0u8; size].into())
                .await?;
        }
        let handle = Handle::find().expect("tokio runtime available within #[tokio::test]");
        Ok(ObjectStoreFileSystem::new(store, handle))
    }

    #[tokio::test]
    async fn test_listing_requires_a_real_object_version() -> VortexResult<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = object_path_from_literal("versioned.vortex");
        store.put(&path, vec![0u8].into()).await?;
        let meta = store.head(&path).await?;

        let before = listing_from_meta(&ObjectMeta {
            version: Some("version-1".to_string()),
            ..meta.clone()
        });
        let after = listing_from_meta(&ObjectMeta {
            version: Some("version-2".to_string()),
            ..meta.clone()
        });
        assert_eq!(before.size, after.size);
        assert_ne!(before.version, after.version);

        let suspended = listing_from_meta(&ObjectMeta {
            version: Some("null".to_string()),
            e_tag: Some("changed-etag".to_string()),
            ..meta
        });
        assert_eq!(suspended.version, None);
        Ok(())
    }

    #[tokio::test]
    async fn test_cache_key_scoped_to_filesystem_and_prefix() -> VortexResult<()> {
        let fs_a = memory_fs(&[("a/shared.vortex", 1), ("b/shared.vortex", 1)]).await?;
        let fs_b = memory_fs(&[("a/shared.vortex", 1)]).await?;
        let listing = fs_a.head("a/shared.vortex").await?.expect("file exists");
        let key_a = fs_a.cache_key(&listing);
        assert_eq!(key_a, fs_a.cache_key(&listing));
        assert_ne!(key_a, fs_b.cache_key(&listing));

        let root: FileSystemRef = Arc::new(fs_a);
        let prefix_a = Arc::clone(&root).with_prefix("a".to_string());
        let prefix_b = root.with_prefix("b".to_string());
        let listing_a = prefix_a.head("shared.vortex").await?.expect("file exists");
        let listing_b = prefix_b.head("shared.vortex").await?.expect("file exists");
        assert_eq!(prefix_a.cache_key(&listing_a), key_a);
        assert_ne!(
            prefix_a.cache_key(&listing_a),
            prefix_b.cache_key(&listing_b)
        );
        Ok(())
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
        let listing = fs.head(path).await?.expect("file exists");
        assert_eq!(listing.path, path);
        assert_eq!(listing.size, Some(5));
        assert_eq!(listing.version, None);
        // and that literal path reopens the file.
        assert_eq!(fs.open_read(&listing).await?.size().await?, 5);
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
        let expected = fs.head(path).await?.expect("file exists");

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
        assert_eq!(fs.open_read(&expected).await?.size().await?, 5);
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
        let expected = fs.head("a~b.vortex").await?.expect("file exists");
        // Local ETags use metadata that can remain unchanged across a same-size rewrite.
        assert_eq!(expected.version, None);

        let wild: Vec<FileListing> = fs_dyn.glob("*.vortex")?.try_collect().await?;
        assert!(wild.contains(&expected), "wildcard glob got {wild:?}");

        let exact: Vec<FileListing> = fs_dyn.glob("a~b.vortex")?.try_collect().await?;
        assert_eq!(exact, vec![expected.clone()]);

        assert_eq!(fs.open_read(&expected).await?.size().await?, 5);
        Ok(())
    }

    /// `delete` must address the same literal key as the rest of the trait. It used
    /// `from_url_path`, which percent-*decodes*: given the real object `a%20b.vortex` it resolved
    /// to `a b.vortex` and deleted a *different* object that happened to exist.
    #[tokio::test]
    async fn test_delete_addresses_the_literal_key() -> VortexResult<()> {
        let fs = memory_fs(&[("dir/a%20b.vortex", 1), ("dir/a b.vortex", 2)]).await?;

        let remaining = fs.head("dir/a b.vortex").await?;
        fs.delete("dir/a%20b.vortex").await?;

        assert_eq!(fs.head("dir/a%20b.vortex").await?, None);
        assert_eq!(
            fs.head("dir/a b.vortex").await?,
            remaining,
            "deleting the percent-literal key must not touch its decoded neighbour"
        );
        Ok(())
    }

    /// The keys `head`/`open_read` accept must also be the keys `delete` accepts.
    #[tokio::test]
    #[rstest]
    #[case::tilde("dir/a~b.vortex")]
    #[case::brackets("dir/a[1].vortex")]
    #[case::hash("dir/a#b.vortex")]
    #[case::space("dir/a b.vortex")]
    #[case::plain("dir/plain.vortex")]
    async fn test_delete_round_trip_special_chars(#[case] path: &str) -> VortexResult<()> {
        let fs = memory_fs(&[(path, 5)]).await?;
        assert!(fs.head(path).await?.is_some());
        fs.delete(path).await?;
        assert_eq!(fs.head(path).await?, None);
        Ok(())
    }

    #[tokio::test]
    async fn test_head_existing_and_missing() -> VortexResult<()> {
        let fs = memory_fs(&[("a/b.vortex", 7)]).await?;
        let listing = fs.head("a/b.vortex").await?.expect("file exists");
        assert_eq!(listing.path, "a/b.vortex");
        assert_eq!(listing.size, Some(7));
        assert_eq!(listing.version, None);
        assert_eq!(fs.head("a/missing.vortex").await?, None);
        Ok(())
    }
}
