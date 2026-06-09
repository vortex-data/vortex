// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! URL-based resolution of cloud [`ObjectStore`]s via [`FileLocation`].
//!
//! [`FileLocation::resolve`] maps a URL or path string to either a local filesystem path or
//! a remote [`ObjectStore`] by delegating to [`parse_url_opts`] with case-insensitive
//! environment variables. No caching is performed here — callers that need process-level
//! store reuse should maintain their own registry.

use std::path::PathBuf;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::parse_url_opts;
use object_store::path::Path;
use url::Url;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// Where the bytes of a file live: on the local filesystem, or in an object store.
///
/// Produced by [`FileLocation::resolve`]. Local paths and `file://` URLs resolve to
/// [`FileLocation::Local`]; any other scheme resolves to [`FileLocation::Remote`].
#[derive(Debug)]
pub enum FileLocation {
    /// A local filesystem path.
    Local(PathBuf),
    /// An object store and the object's path within it.
    Remote {
        /// The object store to read from.
        store: Arc<dyn ObjectStore>,
        /// The object's path within `store`.
        path: Path,
    },
}

impl FileLocation {
    /// Resolve a URL or path string to a [`FileLocation`] using environment variables.
    ///
    /// Equivalent to `resolve_with_props(url, std::iter::empty())`.
    pub fn resolve(url_or_path: impl AsRef<str>) -> VortexResult<Self> {
        Self::resolve_with_props(url_or_path, std::iter::empty::<(String, String)>())
    }

    /// Resolve a URL or path string to a [`FileLocation`], merging `props` with the environment.
    ///
    /// - `file://` URLs and inputs that do not parse as a URL resolve to [`FileLocation::Local`].
    /// - All other schemes (`s3://`, `gs://`, `az://`, `http(s)://`, ...) are resolved via
    ///   [`parse_url_opts`] with case-insensitive environment variables merged with `props`.
    ///   `props` entries take precedence over same-named environment variables.
    ///
    /// No caching is performed. Callers that need process-level store reuse should maintain
    /// their own registry in their own crate.
    pub fn resolve_with_props<K, V>(
        url_or_path: impl AsRef<str>,
        props: impl IntoIterator<Item = (K, V)>,
    ) -> VortexResult<Self>
    where
        K: Into<String>,
        V: Into<String>,
    {
        let url_or_path = url_or_path.as_ref();
        match Url::parse(url_or_path) {
            Ok(url) if url.scheme() == "file" => {
                let path = url
                    .to_file_path()
                    .map_err(|_| vortex_err!("invalid file URL: {url_or_path}"))?;
                Ok(FileLocation::Local(path))
            }
            Ok(url) => {
                let env_opts = std::env::vars().map(|(k, v)| (k.to_ascii_lowercase(), v));
                let props_iter = props.into_iter().map(|(k, v)| (k.into(), v.into()));
                let (store, path) = parse_url_opts(&url, env_opts.chain(props_iter))?;
                Ok(FileLocation::Remote {
                    store: Arc::new(store),
                    path,
                })
            }
            // Not a URL: treat as a local filesystem path.
            Err(_) => Ok(FileLocation::Local(PathBuf::from(url_or_path))),
        }
    }

    /// Returns the local path if this is [`FileLocation::Local`], otherwise `None`.
    pub fn as_local(&self) -> Option<&std::path::Path> {
        match self {
            FileLocation::Local(path) => Some(path.as_path()),
            FileLocation::Remote { .. } => None,
        }
    }

    /// Returns a clone of the store and a reference to the object path if this is
    /// [`FileLocation::Remote`], otherwise `None`.
    pub fn as_remote(&self) -> Option<(Arc<dyn ObjectStore>, &Path)> {
        match self {
            FileLocation::Remote { store, path } => Some((Arc::clone(store), path)),
            FileLocation::Local(_) => None,
        }
    }

    /// Returns `true` if this is a local filesystem path.
    pub fn is_local(&self) -> bool {
        matches!(self, FileLocation::Local(_))
    }

    /// Returns `true` if this is a remote object store location.
    pub fn is_remote(&self) -> bool {
        matches!(self, FileLocation::Remote { .. })
    }

    /// Unwrap as a local filesystem path.
    ///
    /// Returns an error if this is a [`FileLocation::Remote`] location.
    pub fn into_local(self) -> VortexResult<PathBuf> {
        match self {
            FileLocation::Local(path) => Ok(path),
            FileLocation::Remote { path, .. } => {
                vortex_bail!("expected a local path, got remote object store path: {path}")
            }
        }
    }

    /// Unwrap as a remote object store, returning the store and object path.
    ///
    /// Returns an error if this is a [`FileLocation::Local`] path.
    pub fn into_remote(self) -> VortexResult<(Arc<dyn ObjectStore>, Path)> {
        match self {
            FileLocation::Remote { store, path } => Ok((store, path)),
            FileLocation::Local(path) => {
                vortex_bail!(
                    "expected a remote object store, got local path: {}",
                    path.display()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use object_store::path::Path;
    use vortex_error::VortexResult;

    use super::FileLocation;

    impl FileLocation {
        fn unwrap_local(self) -> PathBuf {
            match self {
                FileLocation::Local(path) => path,
                FileLocation::Remote { .. } => panic!("expected Local, got Remote"),
            }
        }
    }

    /// Run `func` with `key` set to `value`, restoring the previous value afterwards.
    fn with_var<F: FnOnce()>(key: &str, value: &str, func: F) {
        let old = std::env::var(key).ok();
        // SAFETY: these unit tests run single-threaded (one process per test under nextest).
        unsafe { std::env::set_var(key, value) };
        func();
        match old {
            None => unsafe { std::env::remove_var(key) },
            Some(val) => unsafe { std::env::set_var(key, val) },
        }
    }

    #[test]
    fn test_resolve() -> VortexResult<()> {
        assert_eq!(
            FileLocation::resolve("/my/absolute/path")?.unwrap_local(),
            PathBuf::from("/my/absolute/path")
        );

        assert_eq!(
            FileLocation::resolve("file:///my/absolute/path")?.unwrap_local(),
            PathBuf::from("/my/absolute/path")
        );

        let (_store, path) =
            FileLocation::resolve("s3://my-bucket/first/second/third/")?.into_remote()?;
        assert_eq!(path, Path::from("first/second/third"));

        Ok(())
    }

    #[test]
    fn test_accessors_local() -> VortexResult<()> {
        let local = FileLocation::resolve("/tmp/data.vortex")?;
        assert!(local.is_local());
        assert!(!local.is_remote());
        assert_eq!(
            local.as_local(),
            Some(std::path::Path::new("/tmp/data.vortex"))
        );
        assert!(local.as_remote().is_none());
        assert_eq!(local.into_local()?, PathBuf::from("/tmp/data.vortex"));

        Ok(())
    }

    #[test]
    fn test_accessors_remote() -> VortexResult<()> {
        let remote = FileLocation::resolve("s3://bucket/key")?;
        assert!(remote.is_remote());
        assert!(!remote.is_local());
        assert!(remote.as_local().is_none());

        let (_store, path) = remote.as_remote().expect("expected remote");
        assert_eq!(path, &Path::from("key"));

        Ok(())
    }

    #[test]
    fn test_into_local_on_remote_errors() -> VortexResult<()> {
        assert!(
            FileLocation::resolve("s3://bucket/key")?
                .into_local()
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn test_into_remote_on_local_errors() -> VortexResult<()> {
        assert!(FileLocation::resolve("/tmp/local")?.into_remote().is_err());
        Ok(())
    }

    #[test]
    #[expect(clippy::use_debug)]
    fn test_resolve_with_props_applies_config() -> VortexResult<()> {
        let (store, _) = FileLocation::resolve_with_props(
            "s3://my-bucket/key",
            [("region".to_string(), "us-west-2".to_string())],
        )?
        .into_remote()?;

        // object_store does not expose store config for inspection, so we assert via Debug,
        // matching the approach used elsewhere for object store configuration tests.
        assert!(format!("{store:?}").contains("us-west-2"));

        Ok(())
    }

    #[test]
    #[expect(clippy::use_debug)]
    fn test_resolve_with_props_overrides_env() -> VortexResult<()> {
        with_var("AWS_REGION", "eu-central-1", || {
            // No props: the environment region is used.
            let env_store = FileLocation::resolve("s3://my-bucket/key")
                .unwrap()
                .into_remote()
                .unwrap()
                .0;
            assert!(format!("{env_store:?}").contains("eu-central-1"));

            // Props take precedence over the same-named environment variable.
            let prop_store =
                FileLocation::resolve_with_props("s3://my-bucket/key", [("region", "ap-south-1")])
                    .unwrap()
                    .into_remote()
                    .unwrap()
                    .0;
            let debug = format!("{prop_store:?}");
            assert!(debug.contains("ap-south-1"));
            assert!(!debug.contains("eu-central-1"));
        });

        Ok(())
    }
}
