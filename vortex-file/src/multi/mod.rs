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
use vortex_io::filesystem::FileVersion;
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
    pub fn with_glob(mut self, glob: impl Into<String>, fs: Option<FileSystemRef>) -> Self {
        let glob = glob.into();
        let glob = if fs.is_none() && std::path::Path::new(&glob).is_relative() {
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
                let fs = maybe_fs
                    .or_else(|| local_fs.as_ref().map(Arc::clone))
                    .unwrap_or_else(|| {
                        unreachable!("local_fs is set when any glob lacks a filesystem")
                    });
                async move {
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

    let source = fs.open_read(file).await?;
    let cache_key = fs.cache_key(file);
    let key = cache_key
        .as_deref()
        .or_else(|| file.version.is_none().then_some(file.path.as_str()));
    open_cached_with_version(
        session,
        key,
        source,
        file.versioned_size(),
        file.version.as_ref(),
        open_options_fn,
    )
    .await
}

/// Open a Vortex file without reusing or populating the session footer cache.
///
/// "key" is the optional cache key provided by user. If it's not found,
/// source.uri() is probed. If there's no uri(), this function errors.
///
/// Use [`open_cached_with_version`] when a strong file version is available. Without one, reusing a
/// path-keyed footer could silently read a previous version of a rewritten file.
pub async fn open_cached(
    session: &VortexSession,
    key: Option<&str>,
    source: Arc<dyn VortexReadAt>,
    file_size: Option<u64>,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<VortexFile> {
    open_cached_with_version(session, key, source, file_size, None, open_options_fn).await
}

/// Open a Vortex file, reusing its session-cached footer only for the same file version.
///
/// "key" is the optional cache key provided by user. A versioned open without an explicit key
/// bypasses both cache lookup and insertion, because `source.uri()` is only a debugging identifier
/// and need not include the backing store's identity. Explicit keys must include that identity.
/// A missing file version also disables both
/// cache operations; in that case the key handling is identical to [`open_cached`]. The source
/// must be opened from the listing that supplied the version, with its reads pinned to that version.
pub async fn open_cached_with_version(
    session: &VortexSession,
    mut key: Option<&str>,
    source: Arc<dyn VortexReadAt>,
    file_size: Option<u64>,
    file_version: Option<&FileVersion>,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<VortexFile> {
    let uri = source.uri().cloned();
    if file_version.is_none() {
        if key.is_none() {
            key = uri.as_deref();
        }
        if key.is_none() {
            vortex_bail!("Missing cache key");
        }
    }

    let mut options = open_options_fn(session.open_options());
    if let Some(size) = file_size {
        options = options.with_file_size(size);
    }

    if let Some(version) = file_version
        && let Some(key) = key
        && let Some(footer) = session.multi_file().get_footer(key, version)
    {
        options = options.with_footer(footer);
    }

    let file = options.open(source).await?;
    if let Some(version) = file_version
        && let Some(key) = key
    {
        session
            .multi_file()
            .put_footer(key, file.footer().clone(), version);
    }
    Ok(file)
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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::future::BoxFuture;
    use vortex_array::IntoArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_io::VortexReadAt;
    use vortex_io::filesystem::FileSystem;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

    use super::*;
    use crate::WriteOptionsSessionExt;

    #[derive(Clone)]
    struct CountingReadAt {
        bytes: ByteBuffer,
        reads: Arc<AtomicUsize>,
    }

    impl VortexReadAt for CountingReadAt {
        fn concurrency(&self) -> usize {
            1
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            self.bytes.size()
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.bytes.read_at(offset, length, alignment)
        }
    }

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
                version: Some(FileVersion::new("metadata-v1")),
            }))
        }

        async fn open_read(&self, _file: &FileListing) -> VortexResult<Arc<dyn VortexReadAt>> {
            Ok(Arc::new(self.bytes.clone()))
        }

        fn cache_key(&self, file: &FileListing) -> Option<String> {
            Some(format!("metadata:{}", file.path))
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            Ok(())
        }
    }

    async fn assert_cached_footer_metadata_order(default_first: bool) -> VortexResult<()> {
        let session = VortexSession::empty()
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
            version: Some(FileVersion::new("metadata-v1")),
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

    #[tokio::test]
    async fn test_open_cached_reuses_only_matching_file_version() -> VortexResult<()> {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with::<MultiFileSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);

        let mut before = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut before, buffer![1u32].into_array().to_array_stream())
            .await?;
        let before = ByteBuffer::from(before);
        let reads = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(CountingReadAt {
            bytes: before.clone(),
            reads: Arc::clone(&reads),
        }) as Arc<dyn VortexReadAt>;
        let version = FileVersion::new("version-1");
        let file = open_cached_with_version(
            &session,
            Some("versioned.vortex"),
            Arc::clone(&source),
            Some(before.len() as u64),
            Some(&version),
            &|options| options,
        )
        .await?;
        assert_eq!(file.row_count(), 1);
        let first_reads = reads.load(Ordering::SeqCst);
        assert!(first_reads > 0);

        let file = open_cached_with_version(
            &session,
            Some("versioned.vortex"),
            source,
            Some(before.len() as u64),
            Some(&version),
            &|options| options,
        )
        .await?;
        assert_eq!(file.row_count(), 1);
        assert_eq!(reads.load(Ordering::SeqCst), first_reads);

        let mut after = ByteBufferMut::empty();
        session
            .write_options()
            .write(
                &mut after,
                buffer![2u32, 3u32].into_array().to_array_stream(),
            )
            .await?;
        let after = ByteBuffer::from(after);
        let file = open_cached_with_version(
            &session,
            Some("versioned.vortex"),
            Arc::new(after.clone()),
            Some(after.len() as u64),
            Some(&FileVersion::new("version-2")),
            &|options| options,
        )
        .await?;
        assert_eq!(file.row_count(), 2);
        Ok(())
    }

    #[derive(Clone)]
    struct UriReadAt {
        bytes: ByteBuffer,
        uri: Arc<str>,
    }

    impl VortexReadAt for UriReadAt {
        fn concurrency(&self) -> usize {
            1
        }

        fn uri(&self) -> Option<&Arc<str>> {
            Some(&self.uri)
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            self.bytes.size()
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.bytes.read_at(offset, length, alignment)
        }
    }

    struct UnkeyedFileSystem {
        bytes: ByteBuffer,
        expose_uri: bool,
    }

    impl fmt::Debug for UnkeyedFileSystem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("UnkeyedFileSystem").finish()
        }
    }

    #[async_trait]
    impl FileSystem for UnkeyedFileSystem {
        fn list(&self, _prefix: &str) -> stream::BoxStream<'_, VortexResult<FileListing>> {
            stream::empty().boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            Ok((path == "shared.vortex").then_some(FileListing {
                path: path.to_string(),
                size: Some(self.bytes.len() as u64),
                version: Some(FileVersion::new("shared-version")),
            }))
        }

        async fn open_read(&self, file: &FileListing) -> VortexResult<Arc<dyn VortexReadAt>> {
            let source: Arc<dyn VortexReadAt> = if self.expose_uri {
                Arc::new(UriReadAt {
                    bytes: self.bytes.clone(),
                    uri: Arc::from(file.path.as_str()),
                })
            } else {
                Arc::new(self.bytes.clone())
            };
            Ok(source)
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_versioned_open_without_cache_key_never_uses_relative_uri() -> VortexResult<()> {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with::<MultiFileSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);

        let mut out_a = ByteBufferMut::empty();
        session
            .write_options()
            .with_metadata_segment("k", ByteBuffer::copy_from(b"bucket-a"))
            .write(&mut out_a, buffer![1u32].into_array().to_array_stream())
            .await?;
        let mut out_b = ByteBufferMut::empty();
        session
            .write_options()
            .with_metadata_segment("k", ByteBuffer::copy_from(b"bucket-b"))
            .write(&mut out_b, buffer![2u32].into_array().to_array_stream())
            .await?;
        let size = out_a.len() as u64;
        assert_eq!(size, out_b.len() as u64);

        let fs_a: FileSystemRef = Arc::new(UnkeyedFileSystem {
            bytes: ByteBuffer::from(out_a),
            expose_uri: true,
        });
        let fs_b: FileSystemRef = Arc::new(UnkeyedFileSystem {
            bytes: ByteBuffer::from(out_b),
            expose_uri: true,
        });
        let listing = FileListing {
            path: "shared.vortex".to_string(),
            size: Some(size),
            version: Some(FileVersion::new("shared-version")),
        };
        let include = |options: VortexOpenOptions| options.include_metadata();

        let file_a = open_file(&fs_a, &listing, &session, &include).await?;
        let file_b = open_file(&fs_b, &listing, &session, &include).await?;

        assert_eq!(
            file_a.metadata_segment("k").map(ByteBuffer::as_slice),
            Some(b"bucket-a".as_slice())
        );
        assert_eq!(
            file_b.metadata_segment("k").map(ByteBuffer::as_slice),
            Some(b"bucket-b".as_slice())
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_unversioned_open_without_uri_uses_listing_path() -> VortexResult<()> {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with::<MultiFileSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);

        let mut output = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut output, buffer![1u32].into_array().to_array_stream())
            .await?;
        let fs: FileSystemRef = Arc::new(UnkeyedFileSystem {
            bytes: ByteBuffer::from(output),
            expose_uri: false,
        });
        let listing = FileListing {
            path: "anonymous.vortex".to_string(),
            // This listing predates a different-size replacement at the same path.
            size: Some(1),
            version: None,
        };

        let file = open_file(&fs, &listing, &session, &|options| options).await?;
        assert_eq!(file.row_count(), 1);
        Ok(())
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
                version: Some(FileVersion::new(format!("{path}-v1"))),
            }))
        }

        async fn open_read(&self, file: &FileListing) -> VortexResult<Arc<dyn VortexReadAt>> {
            let bytes = match file.path.as_str() {
                "b.vortex" => self.b.clone(),
                _ => self.a.clone(),
            };
            Ok(Arc::new(bytes))
        }

        fn cache_key(&self, file: &FileListing) -> Option<String> {
            Some(format!("two-file:{}", file.path))
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            Ok(())
        }
    }

    // Distinct files carry distinct metadata; opening both through the multi-file
    // cache must not let one file's metadata leak into another (URI isolation).
    #[tokio::test]
    async fn test_multi_file_metadata_isolated_per_uri() -> VortexResult<()> {
        let session = VortexSession::empty()
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
            version: Some(FileVersion::new("a.vortex-v1")),
        };
        let b_listing = FileListing {
            path: "b.vortex".to_string(),
            size: None,
            version: Some(FileVersion::new("b.vortex-v1")),
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
}
