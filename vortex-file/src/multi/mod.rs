// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a multi-file [`DataSource`] from multiple Vortex files.

mod session;

use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::StreamExt;
use session::MultiFileSessionExt;
use tracing::debug;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::filesystem::FileListing;
use vortex_io::filesystem::FileSystemRef;
use vortex_layout::LayoutReaderRef;
use vortex_layout::scan::multi::LayoutReaderFactory;
use vortex_layout::scan::multi::MultiLayoutDataSource;
use vortex_scan::DataSource;
use vortex_scan::DataSourceRef;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexOpenOptions;
use crate::v2::FileStatsLayoutReader;

/// A builder that discovers multiple Vortex files from glob patterns and constructs a
/// multi-file [`DataSource`] to scan them as a single data source.
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
    v2: bool,
}

impl MultiFileDataSource {
    /// Create a new [`MultiFileDataSource`] builder.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            glob_sources: Vec::new(),
            open_options_fn: Arc::new(|opts| opts),
            v2: false,
        }
    }

    /// Add a path glob for file discovery.
    ///
    /// The glob path should be relative to the filesystem's base URL. Pass `None` for the
    /// filesystem to use the local filesystem (auto-created in [`Self::build`]).
    pub fn with_glob(mut self, glob: impl Into<String>, fs: Option<FileSystemRef>) -> Self {
        let glob_str = glob.into().trim_start_matches('/').to_string();
        self.glob_sources.push((glob_str, fs));
        self
    }

    /// Enable the v2 layout scan state machine instead of the legacy layout reader.
    ///
    /// When enabled, each file is scanned using the v2 layout scan state machine instead
    /// of the V1 layout reader pipeline.
    pub fn with_v2(mut self, v2: bool) -> Self {
        self.v2 = v2;
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
    pub async fn build(self) -> VortexResult<DataSourceRef> {
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

        // Collect files from all glob sources.
        let mut all_files: Vec<(FileListing, FileSystemRef)> = Vec::new();
        for (glob, maybe_fs) in &self.glob_sources {
            // Use the provided filesystem, or fall back to the local filesystem.
            // We know local_fs is Some when maybe_fs is None (by construction above).
            let fs = maybe_fs
                .as_ref()
                .or(local_fs.as_ref())
                .map(Arc::clone)
                .unwrap_or_else(|| {
                    unreachable!("local_fs is set when any glob lacks a filesystem")
                });
            let files: Vec<FileListing> = fs.glob(glob)?.try_collect().await?;
            for file in files {
                all_files.push((file, Arc::clone(&fs)));
            }
        }

        if all_files.is_empty() {
            let globs: Vec<_> = self.glob_sources.iter().map(|(g, _)| g.as_str()).collect();
            vortex_bail!("No files matched the glob pattern(s): {:?}", globs);
        }

        let file_count = all_files.len();
        let globs: Vec<_> = self.glob_sources.iter().map(|(g, _)| g.as_str()).collect();
        debug!(file_count, glob = ?globs, "discovered files");

        if self.v2 {
            self.build_v2(&all_files).await
        } else {
            self.build_v1(&all_files).await
        }
    }

    async fn build_v1(
        &self,
        files: &[(FileListing, FileSystemRef)],
    ) -> VortexResult<DataSourceRef> {
        let (first_file_listing, first_fs) = &files[0];
        let first_file = open_file(
            first_fs,
            first_file_listing,
            &self.session,
            self.open_options_fn.as_ref(),
        )
        .await?;
        let first_reader = layout_reader_with_stats(&first_file)?;

        let factories: Vec<Arc<dyn LayoutReaderFactory>> = files[1..]
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

        let inner = MultiLayoutDataSource::new_with_first(first_reader, factories, &self.session);

        debug!(file_count = files.len(), dtype = %inner.dtype(), "built MultiFileDataSource (v1)");

        Ok(Arc::new(inner))
    }

    async fn build_v2(
        &self,
        files: &[(FileListing, FileSystemRef)],
    ) -> VortexResult<DataSourceRef> {
        let mut children = Vec::with_capacity(files.len());
        for (file_listing, fs) in files {
            let file = open_file(
                fs,
                file_listing,
                &self.session,
                self.open_options_fn.as_ref(),
            )
            .await?;
            children.push(file.data_source2()?);
        }

        let dtype = children[0].dtype().clone();

        debug!(file_count = files.len(), dtype = %dtype, "built MultiFileDataSource (v2)");

        Ok(Arc::new(MultiV2DataSource { dtype, children }))
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
             Either enable the feature or provide a filesystem via .with_glob(..., Some(fs))."
    );
}

/// Open a single Vortex file, checking the session's footer cache.
async fn open_file(
    fs: &FileSystemRef,
    file: &FileListing,
    session: &VortexSession,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<crate::VortexFile> {
    debug!(path = %file.path, "opening vortex file");

    // Open the reader first so we can use its URI as the cache key.
    // The URI includes the full path (with any filesystem prefix), making it unique
    // even when different PrefixFileSystem instances strip paths to the same relative name.
    let source = fs.open_read(&file.path).await?;
    let cache_key = source
        .uri()
        .map(|u| u.to_string())
        .unwrap_or_else(|| file.path.clone());

    // Build open options. The DashMap Ref from multi_file() must not live across an await,
    // so we scope the cache lookup in a block.
    let options = {
        let mut options = open_options_fn(session.open_options());
        if let Some(size) = file.size {
            options = options.with_file_size(size);
        }
        if let Some(footer) = session.multi_file().get_footer(&cache_key) {
            options = options.with_footer(footer);
        }
        options
    };

    let vortex_file = options.open(source).await?;

    // Store footer in cache (scoped to avoid holding the Ref across subsequent code).
    session
        .multi_file()
        .put_footer(&cache_key, vortex_file.footer().clone());
    Ok(vortex_file)
}

/// Creates a layout reader from a VortexFile, wrapping with `FileStatsLayoutReader` when
/// file-level statistics are available.
fn layout_reader_with_stats(file: &crate::VortexFile) -> VortexResult<LayoutReaderRef> {
    let mut reader = file.layout_reader()?;
    if let Some(stats) = file.file_stats().cloned() {
        reader = Arc::new(FileStatsLayoutReader::new(
            reader,
            stats,
            file.session.clone(),
        ));
    }
    Ok(reader)
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
        Ok(Some(layout_reader_with_stats(&file)?))
    }
}

/// A [`DataSource`] that combines multiple v2 [`DataSourceRef`]s (one per file) into a single
/// scannable source.
struct MultiV2DataSource {
    dtype: DType,
    children: Vec<DataSourceRef>,
}

#[async_trait]
impl DataSource for MultiV2DataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Option<Precision<u64>> {
        let mut sum: u64 = 0;
        for child in &self.children {
            sum = sum.saturating_add(child.row_count()?.into_inner());
        }
        Some(Precision::exact(sum))
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        None
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("MultiV2DataSource partitions are not yet serializable");
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let dtype = scan_request.projection.return_dtype(&self.dtype)?;

        // Collect partitions from all child data sources.
        // Each V2LayoutDataSource produces exactly one partition, so we flatten them all
        // into a single partition list.
        let mut partitions = Vec::with_capacity(self.children.len());
        for child in &self.children {
            let child_scan = child.scan(scan_request.clone()).await?;
            let mut partition_stream = Box::new(child_scan).partitions();
            while let Some(p) = partition_stream.next().await {
                partitions.push(p?);
            }
        }

        Ok(Box::new(MultiV2Scan { dtype, partitions }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

/// A scan over multiple v2 files, yielding pre-collected partitions.
struct MultiV2Scan {
    dtype: DType,
    partitions: Vec<PartitionRef>,
}

impl DataSourceScan for MultiV2Scan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Option<Precision<usize>> {
        Some(Precision::exact(self.partitions.len()))
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter(self.partitions.into_iter().map(Ok)).boxed()
    }
}
