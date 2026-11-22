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
pub use session::FooterCacheKey;
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

    let source = fs.open_read(&file.path).await?;
    let key = source.uri().is_none().then_some(file.path.as_str());
    let namespace = fs.namespace();
    // FileListing carries only an unbound size from an earlier list/head call, so do not
    // pass it to the footer reader — if the file was replaced before open_read, the stale
    // size would seek to the wrong EOF.
    open_cached_in_namespace(
        session,
        namespace.as_deref(),
        key,
        source,
        None,
        open_options_fn,
    )
    .await
}

/// Open a Vortex file and reuse its footer from the session cache when both a filesystem
/// namespace and a handle-bound [`VortexReadAt::object_version`] are available.
///
/// When either the namespace or object version is absent, the footer is read directly from
/// `source` without reading from or writing to the session cache, preventing stale footer reuse
/// after a file is replaced or across distinct filesystems with identical relative paths.
pub async fn open_cached_in_namespace(
    session: &VortexSession,
    namespace: Option<&str>,
    key: Option<&str>,
    source: Arc<dyn VortexReadAt>,
    file_size: Option<u64>,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<VortexFile> {
    let uri = source.uri().cloned();
    let Some(path) = key.or(uri.as_deref()) else {
        vortex_bail!("Missing cache key");
    };

    let cache_key = namespace
        .zip(source.object_version())
        .map(|(ns, version)| FooterCacheKey::new(ns, path, Arc::clone(version)));

    let mut options = open_options_fn(session.open_options());
    if let Some(size) = file_size {
        options = options.with_file_size(size);
    }

    if let Some(cache_key) = &cache_key
        && let Some(footer) = session.multi_file().get_footer(cache_key)
    {
        options = options.with_footer(footer);
    }

    let file = options.open(source).await?;
    if let Some(cache_key) = cache_key {
        session
            .multi_file()
            .put_footer(cache_key, file.footer().clone());
    }
    Ok(file)
}

/// Open a Vortex file and cache its footer on the session when the source identifies its
/// immutable object version and namespace.
///
/// "key" is the optional cache key provided by the caller. If omitted, `source.uri()` is used.
/// If neither is present, `open_cached` returns an error.
pub async fn open_cached(
    session: &VortexSession,
    key: Option<&str>,
    source: Arc<dyn VortexReadAt>,
    file_size: Option<u64>,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<VortexFile> {
    open_cached_in_namespace(session, None, key, source, file_size, open_options_fn).await
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

    use parking_lot::Mutex;
    use vortex_array::IntoArray;
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

    struct MetadataFileSystem {
        bytes: ByteBuffer,
        reads: Arc<AtomicUsize>,
    }

    impl fmt::Debug for MetadataFileSystem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MetadataFileSystem").finish()
        }
    }

    #[async_trait]
    impl FileSystem for MetadataFileSystem {
        fn namespace(&self) -> Option<Arc<str>> {
            Some(Arc::from("metadata-test"))
        }

        fn list(&self, _prefix: &str) -> stream::BoxStream<'_, VortexResult<FileListing>> {
            stream::empty().boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            Ok((path == "metadata.vortex").then_some(FileListing {
                path: path.to_string(),
                size: Some(self.bytes.len() as u64),
            }))
        }

        async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            Ok(Arc::new(VersionedReadAt {
                bytes: self.bytes.clone(),
                uri: Arc::from(path),
                version: Some(Arc::from("v1")),
                reads: Arc::clone(&self.reads),
            }))
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
        let reads = Arc::new(AtomicUsize::new(0));
        let fs: FileSystemRef = Arc::new(MetadataFileSystem {
            bytes: ByteBuffer::from(output),
            reads: Arc::clone(&reads),
        });
        let listing = FileListing {
            path: "metadata.vortex".to_string(),
            size: None,
        };
        let cache_key = FooterCacheKey::new("metadata-test", "metadata.vortex", Arc::from("v1"));
        let default_options = |options: VortexOpenOptions| options;
        let metadata_options = |options: VortexOpenOptions| options.include_metadata();

        if default_first {
            let default = open_file(&fs, &listing, &session, &default_options).await?;
            assert!(default.metadata_segment("key").is_none());
            assert!(session.multi_file().get_footer(&cache_key).is_some());
            let reads_after_first = reads.load(Ordering::SeqCst);
            assert!(reads_after_first > 0);

            let included = open_file(&fs, &listing, &session, &metadata_options).await?;
            assert_eq!(
                included.metadata_segment("key").map(ByteBuffer::as_slice),
                Some(expected.as_slice())
            );
            // Cache hit skips re-reading the footer and only fetches the metadata segment (1 read).
            assert_eq!(reads.load(Ordering::SeqCst), reads_after_first + 1);
        } else {
            let included = open_file(&fs, &listing, &session, &metadata_options).await?;
            assert_eq!(
                included.metadata_segment("key").map(ByteBuffer::as_slice),
                Some(expected.as_slice())
            );
            assert!(session.multi_file().get_footer(&cache_key).is_some());
            let reads_after_first = reads.load(Ordering::SeqCst);
            assert!(reads_after_first > 0);

            let default = open_file(&fs, &listing, &session, &default_options).await?;
            assert!(default.metadata_segment("key").is_none());
            assert_eq!(reads.load(Ordering::SeqCst), reads_after_first);
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

    struct VersionedReadAt {
        bytes: ByteBuffer,
        uri: Arc<str>,
        version: Option<Arc<str>>,
        reads: Arc<AtomicUsize>,
    }

    impl VortexReadAt for VersionedReadAt {
        fn uri(&self) -> Option<&Arc<str>> {
            Some(&self.uri)
        }

        fn object_version(&self) -> Option<&Arc<str>> {
            self.version.as_ref()
        }

        fn concurrency(&self) -> usize {
            1
        }

        fn size(&self) -> futures::future::BoxFuture<'static, VortexResult<u64>> {
            let len = self.bytes.len() as u64;
            Box::pin(async move { Ok(len) })
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: vortex_buffer::Alignment,
        ) -> futures::future::BoxFuture<'static, VortexResult<vortex_array::buffer::BufferHandle>>
        {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.bytes.read_at(offset, length, alignment)
        }
    }

    struct SwappableFileSystem {
        namespace: Option<Arc<str>>,
        current: Mutex<(ByteBuffer, Option<Arc<str>>)>,
        reads: Arc<AtomicUsize>,
    }

    impl fmt::Debug for SwappableFileSystem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("SwappableFileSystem")
                .field("namespace", &self.namespace)
                .finish()
        }
    }

    #[async_trait]
    impl FileSystem for SwappableFileSystem {
        fn namespace(&self) -> Option<Arc<str>> {
            self.namespace.clone()
        }

        fn list(&self, _prefix: &str) -> stream::BoxStream<'_, VortexResult<FileListing>> {
            stream::empty().boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            let len = self.current.lock().0.len() as u64;
            Ok(Some(FileListing {
                path: path.to_string(),
                size: Some(len),
            }))
        }

        async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            let (bytes, version) = self.current.lock().clone();
            Ok(Arc::new(VersionedReadAt {
                bytes,
                uri: Arc::from(path),
                version,
                reads: Arc::clone(&self.reads),
            }))
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            Ok(())
        }
    }

    fn test_session() -> VortexSession {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with::<MultiFileSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    }

    async fn write_vortex_bytes(
        session: &VortexSession,
        array: vortex_array::ArrayRef,
    ) -> VortexResult<ByteBuffer> {
        let mut output = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut output, array.to_array_stream())
            .await?;
        Ok(ByteBuffer::from(output))
    }

    #[tokio::test]
    async fn test_replaced_file_does_not_inherit_stale_footer() -> VortexResult<()> {
        let session = test_session();
        let fresh_session = test_session();

        let gen1 = write_vortex_bytes(&session, buffer![10i32].into_array()).await?;
        let gen2 = write_vortex_bytes(&session, buffer![1i32, 2, 3].into_array()).await?;

        let reads = Arc::new(AtomicUsize::new(0));
        let fs = Arc::new(SwappableFileSystem {
            namespace: Some(Arc::from("fs-replace")),
            current: Mutex::new((gen1, None)),
            reads: Arc::clone(&reads),
        });
        let fs_ref: FileSystemRef = Arc::clone(&fs) as FileSystemRef;
        let listing = FileListing {
            path: "same.vortex".to_string(),
            size: None,
        };
        let opts = |o: VortexOpenOptions| o;

        let opened1 = open_file(&fs_ref, &listing, &session, &opts).await?;
        assert_eq!(opened1.row_count(), 1);

        // Replace with a 3-row file at the same path.
        *fs.current.lock() = (gen2.clone(), None);

        let opened2 = open_file(&fs_ref, &listing, &session, &opts).await?;
        let fresh2 = fresh_session.open_options().open_buffer(gen2)?;

        assert_eq!(opened2.row_count(), fresh2.row_count());
        assert_eq!(opened2.row_count(), 3);
        assert_eq!(opened2.dtype(), fresh2.dtype());
        assert_eq!(
            format!("{:?}", opened2.file_stats().map(|s| s.aggregates())),
            format!("{:?}", fresh2.file_stats().map(|s| s.aggregates()))
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_equal_size_replacement_does_not_inherit_stale_footer() -> VortexResult<()> {
        let session = test_session();
        let fresh_session = test_session();

        // Two distinct generations with identical byte length and different statistics.
        let gen1 = write_vortex_bytes(&session, buffer![100i32, 200, 300].into_array()).await?;
        let gen2 = write_vortex_bytes(&session, buffer![101i32, 201, 301].into_array()).await?;
        assert_eq!(gen1.len(), gen2.len());

        let reads = Arc::new(AtomicUsize::new(0));
        let fs = Arc::new(SwappableFileSystem {
            namespace: Some(Arc::from("fs-equal-size")),
            current: Mutex::new((gen1.clone(), None)),
            reads,
        });
        let fs_ref: FileSystemRef = Arc::clone(&fs) as FileSystemRef;
        let listing = FileListing {
            path: "equal.vortex".to_string(),
            size: Some(gen2.len() as u64),
        };
        let opts = |o: VortexOpenOptions| o;

        let opened1 = open_file(&fs_ref, &listing, &session, &opts).await?;
        let fresh1 = fresh_session.open_options().open_buffer(gen1)?;
        assert_eq!(
            format!("{:?}", opened1.file_stats().map(|s| s.aggregates())),
            format!("{:?}", fresh1.file_stats().map(|s| s.aggregates()))
        );

        *fs.current.lock() = (gen2.clone(), None);

        let opened2 = open_file(&fs_ref, &listing, &session, &opts).await?;
        let fresh2 = fresh_session.open_options().open_buffer(gen2)?;

        assert_eq!(opened2.row_count(), fresh2.row_count());
        assert_eq!(opened2.dtype(), fresh2.dtype());
        assert_eq!(
            format!("{:?}", opened2.file_stats().map(|s| s.aggregates())),
            format!("{:?}", fresh2.file_stats().map(|s| s.aggregates()))
        );
        assert_ne!(
            format!("{:?}", opened1.file_stats().map(|s| s.aggregates())),
            format!("{:?}", opened2.file_stats().map(|s| s.aggregates()))
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_two_filesystems_same_relative_path_do_not_collide() -> VortexResult<()> {
        let session = test_session();

        let bytes_a = write_vortex_bytes(&session, buffer![1i32].into_array()).await?;
        let bytes_b = write_vortex_bytes(&session, buffer![10i64, 20, 30].into_array()).await?;

        let fs_a: FileSystemRef = Arc::new(SwappableFileSystem {
            namespace: Some(Arc::from("bucket-a")),
            current: Mutex::new((bytes_a, Some(Arc::from("v1")))),
            reads: Arc::new(AtomicUsize::new(0)),
        });
        let fs_b: FileSystemRef = Arc::new(SwappableFileSystem {
            namespace: Some(Arc::from("bucket-b")),
            current: Mutex::new((bytes_b, Some(Arc::from("v1")))),
            reads: Arc::new(AtomicUsize::new(0)),
        });
        let listing = FileListing {
            path: "shared.vortex".to_string(),
            size: None,
        };
        let opts = |o: VortexOpenOptions| o;

        let file_a = open_file(&fs_a, &listing, &session, &opts).await?;
        let file_b = open_file(&fs_b, &listing, &session, &opts).await?;

        assert_eq!(file_a.row_count(), 1);
        assert_eq!(file_b.row_count(), 3);
        assert_ne!(file_a.dtype(), file_b.dtype());
        Ok(())
    }

    #[tokio::test]
    async fn test_unchanged_version_reuses_footer_and_new_version_refreshes() -> VortexResult<()> {
        let session = test_session();

        let gen1 = write_vortex_bytes(&session, buffer![1i32].into_array()).await?;
        let gen2 = write_vortex_bytes(&session, buffer![1i32, 2, 3].into_array()).await?;

        let reads = Arc::new(AtomicUsize::new(0));
        let fs = Arc::new(SwappableFileSystem {
            namespace: Some(Arc::from("bucket-versioned")),
            current: Mutex::new((gen1, Some(Arc::from("etag-1")))),
            reads: Arc::clone(&reads),
        });
        let fs_ref: FileSystemRef = Arc::clone(&fs) as FileSystemRef;
        let listing = FileListing {
            path: "versioned.vortex".to_string(),
            size: None,
        };
        let opts = |o: VortexOpenOptions| o;

        let first = open_file(&fs_ref, &listing, &session, &opts).await?;
        assert_eq!(first.row_count(), 1);
        let reads_after_first = reads.load(Ordering::SeqCst);
        assert!(reads_after_first > 0);

        // Second open with the same namespace, path, and bound version must reuse the cached footer.
        let second = open_file(&fs_ref, &listing, &session, &opts).await?;
        assert_eq!(second.row_count(), 1);
        assert_eq!(reads.load(Ordering::SeqCst), reads_after_first);

        // Replacing with a new bound version must read the new footer.
        *fs.current.lock() = (gen2, Some(Arc::from("etag-2")));
        let third = open_file(&fs_ref, &listing, &session, &opts).await?;
        assert_eq!(third.row_count(), 3);
        assert!(reads.load(Ordering::SeqCst) > reads_after_first);
        Ok(())
    }

    #[tokio::test]
    async fn test_replacement_with_stale_listing_size_uses_opened_object_size() -> VortexResult<()>
    {
        let session = test_session();

        let gen1 = write_vortex_bytes(&session, buffer![1i32].into_array()).await?;
        let gen2 = write_vortex_bytes(&session, buffer![10i32, 20, 30].into_array()).await?;
        assert_ne!(gen1.len(), gen2.len());

        let fs = Arc::new(SwappableFileSystem {
            namespace: Some(Arc::from("fs-stale-size")),
            current: Mutex::new((gen1.clone(), None)),
            reads: Arc::new(AtomicUsize::new(0)),
        });
        let fs_ref: FileSystemRef = Arc::clone(&fs) as FileSystemRef;
        // Simulate a FileListing captured before the file was replaced with a larger one.
        let listing = FileListing {
            path: "replaced.vortex".to_string(),
            size: Some(gen1.len() as u64),
        };
        let opts = |o: VortexOpenOptions| o;

        let first = open_file(&fs_ref, &listing, &session, &opts).await?;
        assert_eq!(first.row_count(), 1);

        *fs.current.lock() = (gen2, None);

        let second = open_file(&fs_ref, &listing, &session, &opts).await?;
        assert_eq!(second.row_count(), 3);
        Ok(())
    }
}
