// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a [`MultiLayoutDataSource`] from multiple Vortex files.

mod session;
mod uri;

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
pub use session::MultiFileSession;
use session::MultiFileSessionExt;
use tracing::debug;
pub use uri::parse_uri_or_path;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::VortexReadAt;
use vortex_io::filesystem::FileListing;
use vortex_io::filesystem::FileSystemRef;
use vortex_layout::LayoutReaderRef;
use vortex_layout::scan::multi::LayoutReaderFactory;
use vortex_layout::scan::multi::MultiLayoutDataSource;
use vortex_scan::DataSource;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::VortexOpenOptions;

/// A builder that discovers multiple Vortex files from glob patterns and constructs a
/// [`MultiLayoutDataSource`] to scan them as a single data source.
///
/// The primary interface is [`Self::with_glob`], which accepts a glob pattern and an optional
/// filesystem. For non-local filesystems (S3, GCS, etc.), callers must provide a [`FileSystemRef`].
/// For local files, pass `None` and a local filesystem will be created automatically.
///
/// # Examples
///
/// ```ignore
/// // Local files — filesystem is auto-created:
/// let ds = MultiFileDataSource::new(session)
///     .with_glob("/data/warehouse/*.vortex", None)
///     .build()
///     .await?;
///
/// // S3 — caller provides the filesystem:
/// let ds = MultiFileDataSource::new(session)
///     .with_glob("prefix/*.vortex", Some(s3_fs))
///     .build()
///     .await?;
///
/// // Mixed filesystems — multiple globs with different filesystems:
/// let ds = MultiFileDataSource::new(session)
///     .with_glob("bucket-a/*.vortex", Some(s3_fs.clone()))
///     .with_glob("bucket-b/*.vortex", Some(s3_fs))
///     .with_glob("gcs-bucket/*.vortex", Some(gcs_fs))
///     .build()
///     .await?;
/// ```
pub struct MultiFileDataSource {
    session: VortexSession,
    /// List of (glob, optional filesystem) pairs to resolve.
    /// When the filesystem is None, a local filesystem will be created in build().
    glob_sources: Vec<(String, Option<FileSystemRef>)>,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
}

/// In-flight glob resolutions in [`MultiFileDataSource::build`]. Callers like the JNI data
/// source add one exact path per glob source, where each resolution is a single remote
/// metadata lookup; resolving them concurrently avoids one round trip of latency per file.
const GLOB_RESOLUTION_CONCURRENCY: usize = 16;

impl MultiFileDataSource {
    /// Create a new [`MultiFileDataSource`] builder.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            glob_sources: Vec::new(),
            open_options_fn: Arc::new(|opts| opts),
        }
    }

    /// Add a path glob for file discovery.
    ///
    /// The glob path should be relative to the filesystem's base URL. Pass `None` for the
    /// filesystem to use the local filesystem (auto-created in [`Self::build`]).
    ///
    /// Relative paths are resolved against the process working directory.
    ///
    /// With the `object_store_registry` feature enabled, an object-store URL (`s3://`, `gs://`,
    /// `az://`, `https://`, ...) passed with `None` is instead resolved through the
    /// `vortex-cloud` registry, configured from environment variables.
    pub fn with_glob(mut self, glob: impl Into<String>, fs: Option<FileSystemRef>) -> Self {
        let glob = glob.into();
        let glob = if object_store_url(&glob).is_some() {
            // Object-store URLs are resolved against the registry in
            // `build`; joining them onto the working directory or trimming
            // leading slashes would corrupt them.
            glob
        } else if fs.is_none() && std::path::Path::new(&glob).is_relative() {
            std::env::current_dir()
                .map(|cwd| cwd.join(&glob).to_string_lossy().into_owned())
                .unwrap_or(glob)
                .trim_start_matches('/')
                .to_string()
        } else {
            glob.trim_start_matches('/').to_string()
        };
        self.glob_sources.push((glob, fs));
        self
    }

    /// Customize [`VortexOpenOptions`] applied to each file.
    ///
    /// Use this to configure segment caches, metrics registries, or other per-file options.
    pub fn with_open_options(
        mut self,
        f: impl Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync + 'static,
    ) -> Self {
        self.open_options_fn = Arc::new(f);
        self
    }

    /// Build the [`DataSource`].
    ///
    /// Discovers files via glob, opens the first file eagerly to determine the schema,
    /// and creates lazy factories for the remaining files.
    pub async fn build(self) -> VortexResult<MultiLayoutDataSource> {
        if self.glob_sources.is_empty() {
            vortex_bail!("MultiFileDataSource requires at least one glob pattern");
        }

        // Create local filesystem lazily if needed (only if any glob lacks a filesystem).
        let local_fs: Option<FileSystemRef> = self
            .glob_sources
            .iter()
            .any(|(_, fs)| fs.is_none())
            .then(|| create_local_filesystem(&self.session))
            .transpose()?;

        let globs: Vec<String> = self.glob_sources.iter().map(|(g, _)| g.clone()).collect();

        // Resolve glob sources concurrently while preserving their order, since the order
        // determines partition indices and which file is opened eagerly for the schema.
        let resolved: Vec<Vec<(FileListing, FileSystemRef)>> =
            stream::iter(self.glob_sources.into_iter().map(|(glob, maybe_fs)| {
                // Use the provided filesystem, or fall back to the local filesystem.
                // We know local_fs is Some when maybe_fs is None (by construction above).
                let local_fs = local_fs.clone();
                let session = self.session.clone();
                async move {
                    // An object-store URL without an explicit filesystem is
                    // resolved through the shared registry; everything else
                    // falls back to the local filesystem as before.
                    let (glob, maybe_fs) = if maybe_fs.is_none() {
                        match object_store_url(&glob) {
                            Some(url) => {
                                let (store_path, fs) = registry_filesystem(&url, &session)?;
                                (store_path, Some(fs))
                            }
                            None => (glob, None),
                        }
                    } else {
                        (glob, maybe_fs)
                    };
                    let fs = maybe_fs.or(local_fs).unwrap_or_else(|| {
                        unreachable!("local_fs is set when any glob lacks a filesystem")
                    });
                    let files: Vec<FileListing> = fs.glob(&glob)?.try_collect().await?;
                    Ok::<_, VortexError>(
                        files
                            .into_iter()
                            .map(|file| (file, Arc::clone(&fs)))
                            .collect(),
                    )
                }
            }))
            .buffered(GLOB_RESOLUTION_CONCURRENCY)
            .try_collect()
            .await?;
        let all_files: Vec<(FileListing, FileSystemRef)> = resolved.into_iter().flatten().collect();

        if all_files.is_empty() {
            vortex_bail!("No files matched the glob pattern(s): {:?}", globs);
        }

        let file_count = all_files.len();
        debug!(file_count, glob = ?globs, "discovered files");

        // Open first file eagerly for dtype.
        let (first_file_listing, first_fs) = &all_files[0];
        let open_fn = self.open_options_fn.as_ref();
        let first_file = open_file(first_fs, first_file_listing, &self.session, open_fn).await?;
        let first_reader = first_file.layout_reader()?;

        let byte_sizes: Vec<Option<u64>> = all_files.iter().map(|(file, _)| file.size).collect();

        let factories: Vec<Arc<dyn LayoutReaderFactory>> = all_files[1..]
            .iter()
            .map(|(file, fs)| {
                Arc::new(VortexFileReaderFactory {
                    fs: Arc::clone(fs),
                    file: file.clone(),
                    session: self.session.clone(),
                    open_options_fn: Arc::clone(&self.open_options_fn),
                }) as Arc<dyn LayoutReaderFactory>
            })
            .collect();

        let inner = MultiLayoutDataSource::new_with_first(
            first_reader,
            factories,
            byte_sizes,
            &self.session,
        );

        debug!(file_count, dtype = %inner.dtype(), "built MultiFileDataSource");

        Ok(inner)
    }
}

/// Parse `glob` as an object-store URL, mirroring the resolution of the
/// Python binding's `resolve_store`: a parse failure or a `file` scheme is a
/// local path. Single-character schemes are also treated as local so Windows
/// drive paths (`C:/data.vortex`) never match.
#[cfg(feature = "object_store_registry")]
fn object_store_url(glob: &str) -> Option<url::Url> {
    let url = url::Url::parse(glob).ok()?;
    (url.scheme().len() > 1 && url.scheme() != "file").then_some(url)
}

#[cfg(not(feature = "object_store_registry"))]
fn object_store_url(_glob: &str) -> Option<url::Url> {
    None
}

/// Resolve `url` through the shared [`vortex_cloud::Registry`], returning the
/// store-relative path and a filesystem over the store. The store's I/O is
/// bridged onto a tokio-compatible thread via `Compat`, as in the Python
/// binding.
#[cfg(feature = "object_store_registry")]
fn registry_filesystem(
    url: &url::Url,
    session: &VortexSession,
) -> VortexResult<(String, FileSystemRef)> {
    use std::sync::LazyLock;

    use object_store::registry::ObjectStoreRegistry;
    use vortex_cloud::Registry;
    use vortex_io::compat::Compat;
    use vortex_io::object_store::ObjectStoreFileSystem;
    use vortex_io::session::RuntimeSessionExt;

    static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

    let (store, path) = REGISTRY.resolve(url)?;
    let store = Arc::new(Compat::new(store));
    let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(store, session.handle()));
    Ok((path.to_string(), fs))
}

#[cfg(not(feature = "object_store_registry"))]
fn registry_filesystem(
    _url: &url::Url,
    _session: &VortexSession,
) -> VortexResult<(String, FileSystemRef)> {
    unreachable!("object_store_url returns None without the object_store_registry feature")
}

/// Creates a local filesystem backed by `object_store::local::LocalFileSystem`.
// TODO(ngates): create a native file system without an object_store dependency.
//  Turns out it's not a trivial change because we have always used object_store with its own
//  coalescing and concurrency configs, so we need to re-tune for local disk.
#[cfg(feature = "object_store")]
fn create_local_filesystem(session: &VortexSession) -> VortexResult<FileSystemRef> {
    use vortex_io::object_store::ObjectStoreFileSystem;
    use vortex_io::session::RuntimeSessionExt;

    let store = Arc::new(object_store::local::LocalFileSystem::default());
    let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(store, session.handle()));
    Ok(fs)
}

#[cfg(not(feature = "object_store"))]
fn create_local_filesystem(_session: &VortexSession) -> VortexResult<FileSystemRef> {
    vortex_bail!(
        "The 'object_store' feature is required for automatic local filesystem creation. \
             Either enable the feature or provide a filesystem via .with_filesystem()."
    );
}

/// Open a single Vortex file, checking the session's footer cache.
async fn open_file(
    fs: &FileSystemRef,
    file: &FileListing,
    session: &VortexSession,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<VortexFile> {
    tracing::trace!(path = %file.path, "opening vortex file");

    let source = fs.open_read(&file.path).await?;
    open_cached(session, source, &file.path, file.size, open_options_fn).await
}

/// Open a single Vortex file through the session's footer cache, so that a later open of the
/// same file skips the footer read.
///
/// The cache is keyed by the source's [`uri`](vortex_io::VortexReadAt::uri) where it reports one,
/// since that includes the full path (with any filesystem prefix) and so stays unique even when
/// different filesystems strip paths to the same relative name. `fallback_key` identifies the file
/// for sources that report no URI, and must be stable and unique within the session — two
/// different files sharing a key would read each other's footer.
///
/// Caching the footer is independent of [`VortexOpenOptions::include_metadata`]: the footer holds
/// only metadata *locators*, and each open resolves the segments it was asked for.
pub async fn open_cached(
    session: &VortexSession,
    source: Arc<dyn VortexReadAt>,
    fallback_key: &str,
    file_size: Option<u64>,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<VortexFile> {
    let cache_key = source
        .uri()
        .map_or_else(|| fallback_key.to_owned(), |uri| uri.to_string());

    // Build open options. The cache guard from multi_file() must not live across an await,
    // so we scope the cache lookup in a block.
    let options = {
        let mut options = open_options_fn(session.open_options());
        if let Some(size) = file_size {
            options = options.with_file_size(size);
        }
        if let Some(footer) = session.multi_file().get_footer(&cache_key) {
            options = options.with_footer(footer);
        }
        options
    };

    let vortex_file = options.open(source).await?;

    // Store footer in cache (scoped to avoid holding the guard across subsequent code).
    session
        .multi_file()
        .put_footer(&cache_key, vortex_file.footer().clone());
    Ok(vortex_file)
}

/// A [`LayoutReaderFactory`] that lazily opens a single Vortex file and returns its layout reader.
struct VortexFileReaderFactory {
    fs: FileSystemRef,
    file: FileListing,
    session: VortexSession,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
}

#[async_trait]
impl LayoutReaderFactory for VortexFileReaderFactory {
    async fn open(&self) -> VortexResult<Option<LayoutReaderRef>> {
        let file = open_file(
            &self.fs,
            &self.file,
            &self.session,
            self.open_options_fn.as_ref(),
        )
        .await?;

        Ok(Some(file.layout_reader()?))
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_io::VortexReadAt;
    use vortex_io::filesystem::FileSystem;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;

    use super::*;
    use crate::WriteOptionsSessionExt;

    struct MetadataFileSystem {
        bytes: ByteBuffer,
    }

    impl fmt::Debug for MetadataFileSystem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MetadataFileSystem").finish()
        }
    }

    #[async_trait]
    impl FileSystem for MetadataFileSystem {
        fn list(&self, _prefix: &str) -> stream::BoxStream<'_, VortexResult<FileListing>> {
            stream::empty().boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            Ok((path == "metadata.vortex").then_some(FileListing {
                path: path.to_string(),
                size: Some(self.bytes.len() as u64),
            }))
        }

        async fn open_read(&self, _path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            Ok(Arc::new(self.bytes.clone()))
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            Ok(())
        }
    }

    async fn assert_cached_footer_metadata_order(default_first: bool) -> VortexResult<()> {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with::<MultiFileSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);

        let expected = ByteBuffer::copy_from(b"cached metadata");
        let mut output = ByteBufferMut::empty();
        session
            .write_options()
            .with_metadata_segment("key", expected.clone())
            .write(&mut output, buffer![1u32].into_array().to_array_stream())
            .await?;
        let fs: FileSystemRef = Arc::new(MetadataFileSystem {
            bytes: ByteBuffer::from(output),
        });
        let listing = FileListing {
            path: "metadata.vortex".to_string(),
            size: None,
        };
        let default_options = |options: VortexOpenOptions| options;
        let metadata_options = |options: VortexOpenOptions| options.include_metadata();

        if default_first {
            let default = open_file(&fs, &listing, &session, &default_options).await?;
            assert!(default.metadata_segment("key").is_none());
            let included = open_file(&fs, &listing, &session, &metadata_options).await?;
            assert_eq!(
                included.metadata_segment("key").map(ByteBuffer::as_slice),
                Some(expected.as_slice())
            );
        } else {
            let included = open_file(&fs, &listing, &session, &metadata_options).await?;
            assert_eq!(
                included.metadata_segment("key").map(ByteBuffer::as_slice),
                Some(expected.as_slice())
            );
            let default = open_file(&fs, &listing, &session, &default_options).await?;
            assert!(default.metadata_segment("key").is_none());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cached_footer_default_then_metadata_is_order_independent() -> VortexResult<()> {
        assert_cached_footer_metadata_order(true).await
    }

    #[tokio::test]
    async fn test_cached_footer_metadata_then_default_is_order_independent() -> VortexResult<()> {
        assert_cached_footer_metadata_order(false).await
    }

    struct TwoFileSystem {
        a: ByteBuffer,
        b: ByteBuffer,
    }

    impl fmt::Debug for TwoFileSystem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("TwoFileSystem").finish()
        }
    }

    #[async_trait]
    impl FileSystem for TwoFileSystem {
        fn list(&self, _prefix: &str) -> stream::BoxStream<'_, VortexResult<FileListing>> {
            stream::empty().boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            let size = match path {
                "a.vortex" => self.a.len(),
                "b.vortex" => self.b.len(),
                _ => return Ok(None),
            };
            Ok(Some(FileListing {
                path: path.to_string(),
                size: Some(size as u64),
            }))
        }

        async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            let bytes = match path {
                "b.vortex" => self.b.clone(),
                _ => self.a.clone(),
            };
            Ok(Arc::new(bytes))
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            Ok(())
        }
    }

    // Distinct files carry distinct metadata; opening both through the multi-file
    // cache must not let one file's metadata leak into another (URI isolation).
    #[tokio::test]
    async fn test_multi_file_metadata_isolated_per_uri() -> VortexResult<()> {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with::<MultiFileSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);

        let mut out_a = ByteBufferMut::empty();
        session
            .write_options()
            .with_metadata_segment("k", ByteBuffer::copy_from(b"file-a"))
            .write(&mut out_a, buffer![1u32].into_array().to_array_stream())
            .await?;
        let mut out_b = ByteBufferMut::empty();
        session
            .write_options()
            .with_metadata_segment("k", ByteBuffer::copy_from(b"file-b"))
            .write(&mut out_b, buffer![2u32].into_array().to_array_stream())
            .await?;

        let fs: FileSystemRef = Arc::new(TwoFileSystem {
            a: ByteBuffer::from(out_a),
            b: ByteBuffer::from(out_b),
        });
        let include = |options: VortexOpenOptions| options.include_metadata();

        let a_listing = FileListing {
            path: "a.vortex".to_string(),
            size: None,
        };
        let b_listing = FileListing {
            path: "b.vortex".to_string(),
            size: None,
        };
        let fa = open_file(&fs, &a_listing, &session, &include).await?;
        let fb = open_file(&fs, &b_listing, &session, &include).await?;

        assert_eq!(
            fa.metadata_segment("k").map(ByteBuffer::as_slice),
            Some(b"file-a".as_slice())
        );
        assert_eq!(
            fb.metadata_segment("k").map(ByteBuffer::as_slice),
            Some(b"file-b".as_slice())
        );
        Ok(())
    }

    #[cfg(feature = "object_store_registry")]
    #[tokio::test]
    async fn test_object_store_url_resolution() {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with::<MultiFileSession>();

        // A URL scheme resolves through the registry, not the local
        // filesystem. The Azure builder deterministically fails without
        // configuration, and that failure can only arise on the registry
        // path — local-path handling would report an unmatched glob.
        let message = match MultiFileDataSource::new(session)
            .with_glob("az://account/container/missing.vortex", None)
            .build()
            .await
        {
            Ok(_) => panic!("unconfigured az:// URL must not resolve"),
            Err(e) => e.to_string(),
        };
        assert!(
            !message.contains("glob"),
            "URL was resolved as a local path: {message}"
        );
    }
}
