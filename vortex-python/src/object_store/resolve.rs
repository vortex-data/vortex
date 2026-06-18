// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use object_store::ObjectStore;
use object_store::path::Path;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::compat::Compat;

use crate::object_store::registry::Registry;

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// Resolve a path to either a local file system path, or a registered object store.
///
/// An explicit `ObjectStore` can be provided optionally, in which case the path is resolved
/// against the store's prefix.
///
/// If the store is provided, it is carried along, otherwise we look up an appropriate store
/// in the default registry.
pub(crate) fn resolve_store(
    url_or_path: &str,
    store: Option<Arc<dyn ObjectStore>>,
) -> VortexResult<ResolvedStore> {
    match store {
        // If explicit store is provided use that
        Some(store) => Ok(ResolvedStore::object_store(store, Path::from(url_or_path))),
        None => {
            // If the URL does not parse
            match Url::parse(url_or_path) {
                Ok(url) if url.scheme() == "file" => {
                    let path = url
                        .to_file_path()
                        .map_err(|_| vortex_err!("invalid file URL: {url_or_path}"))?;
                    Ok(ResolvedStore::Path(path))
                }
                Ok(url) => {
                    let (store, path) = REGISTRY.resolve(&url)?;
                    Ok(ResolvedStore::object_store(store, path))
                }
                Err(_) => {
                    // Treat the input string as a local file system path, which may be
                    Ok(ResolvedStore::Path(PathBuf::from(url_or_path)))
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum ResolvedStore {
    ObjectStore(Arc<dyn ObjectStore>, Path),
    Path(PathBuf),
}

impl ResolvedStore {
    /// Build an [`ObjectStore`](ResolvedStore::ObjectStore) variant, wrapping `store` in
    /// [`Compat`].
    fn object_store(store: Arc<dyn ObjectStore>, path: Path) -> Self {
        ResolvedStore::ObjectStore(Arc::new(Compat::new(store)), path)
    }

    #[cfg(test)]
    fn unwrap_store(self) -> (Arc<dyn ObjectStore>, Path) {
        match self {
            ResolvedStore::ObjectStore(store, path) => (store, path),
            ResolvedStore::Path(_) => {
                panic!("cannot unwrap ResolvedStore::Path as store")
            }
        }
    }

    #[cfg(test)]
    pub fn unwrap_path(self) -> PathBuf {
        match self {
            ResolvedStore::ObjectStore(..) => {
                panic!("cannot unwrap ResolvedStore::ObjectStore as path")
            }
            ResolvedStore::Path(path_buf) => path_buf,
        }
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;
    use std::sync::Arc;

    use object_store::local::LocalFileSystem;
    use object_store::path::Path;

    use crate::object_store::resolve::resolve_store;

    #[test]
    fn test_resolve() {
        assert_eq!(
            resolve_store("/my/absolute/path", None)
                .unwrap()
                .unwrap_path(),
            PathBuf::from("/my/absolute/path")
        );

        assert_eq!(
            resolve_store("file:///my/absolute/path", None)
                .unwrap()
                .unwrap_path(),
            PathBuf::from("/my/absolute/path")
        );

        let (_store, path) = resolve_store("s3://my-bucket/first/second/third/", None)
            .unwrap()
            .unwrap_store();

        assert_eq!(path, Path::from("first/second/third"));

        let local_store = Arc::new(LocalFileSystem::default());
        let (_store, path) = resolve_store("/root/test", Some(local_store))
            .unwrap()
            .unwrap_store();

        assert_eq!(path, Path::from("root/test"));
    }
}
