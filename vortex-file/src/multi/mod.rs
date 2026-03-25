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
use vortex_error::vortex_err;
use vortex_io::filesystem::FileListing;
use vortex_io::filesystem::FileSystemRef;
use vortex_layout::LayoutReaderRef;
use vortex_layout::scan::multi::LayoutReaderFactory;
use vortex_layout::scan::multi::MultiLayoutDataSource;
use vortex_scan::DataSource;
use vortex_scan::DataSourceRef;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexOpenOptions;
use crate::v2::FileStatsLayoutReader;

/// A builder that discovers multiple Vortex files from a glob pattern and constructs a
/// multi-file [`DataSource`] to scan them.
///
/// The primary interface is [`Self::with_glob`], which accepts a glob
/// pattern (optionally prefixed with `file://`). For non-local filesystems (S3, GCS, etc.),
/// callers must also provide a [`FileSystemRef`] via [`Self::with_filesystem`]).
///
/// # Examples
///
/// ```ignore
/// // Local files — filesystem is auto-created:
/// let ds = MultiFileDataSource::new(session)
///     .with_glob("/data/warehouse/*.vortex")
///     .build()
///     .await?;
///
/// // S3 — caller provides the filesystem:
/// let ds = MultiFileDataSource::new(session)
///     .with_filesystem(s3_fs)
///     .with_glob("prefix/*.vortex")
///     .build()
///     .await?;
/// ```
pub struct MultiFileDataSource {
    session: VortexSession,
    fs: Option<FileSystemRef>,
    glob: Option<String>,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    v2: bool,
}

impl MultiFileDataSource {
    /// Create a new [`MultiFileDataSource`] builder.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            fs: None,
            glob: None,
            open_options_fn: Arc::new(|opts| opts),
            v2: false,
        }
    }

    /// Set the path glob for file discovery.
    ///
    /// This path should be relative to the filesystem's base URL.
    pub fn with_glob(mut self, glob: impl Into<String>) -> Self {
        self.glob = Some(glob.into().trim_start_matches("/").to_string());
        self
    }

    /// Set the filesystem to use for file discovery and reading.
    ///
    /// Required for non-local URLs (S3, GCS, etc.). For `file://` or bare path URLs,
    /// a local filesystem is created automatically if none is provided.
    pub fn with_filesystem(mut self, fs: FileSystemRef) -> Self {
        self.fs = Some(fs);
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
    pub async fn build(mut self) -> VortexResult<DataSourceRef> {
        let glob = self
            .glob
            .take()
            .ok_or_else(|| vortex_err!("MultiFileDataSource requires a glob URL"))?;

        let fs = match self.fs.take() {
            Some(fs) => fs,
            None => create_local_filesystem(&self.session)?,
        };
        let files: Vec<FileListing> = fs.glob(&glob)?.try_collect().await?;

        if files.is_empty() {
            vortex_bail!("No files matched the glob pattern '{}'", glob);
        }

        let file_count = files.len();
        debug!(file_count, glob = %glob, "discovered files");

        if self.v2 {
            self.build_v2(&fs, &files).await
        } else {
            self.build_v1(&fs, &files).await
        }
    }

    async fn build_v1(
        &self,
        fs: &FileSystemRef,
        files: &[FileListing],
    ) -> VortexResult<DataSourceRef> {
        let first_file =
            open_file(fs, &files[0], &self.session, self.open_options_fn.as_ref()).await?;
        let first_reader = layout_reader_with_stats(&first_file)?;

        let factories: Vec<Arc<dyn LayoutReaderFactory>> = files[1..]
            .iter()
            .map(|f| {
                Arc::new(VortexFileReaderFactory {
                    fs: fs.clone(),
                    file: f.clone(),
                    session: self.session.clone(),
                    open_options_fn: self.open_options_fn.clone(),
                }) as Arc<dyn LayoutReaderFactory>
            })
            .collect();

        let inner = MultiLayoutDataSource::new_with_first(first_reader, factories, &self.session);

        debug!(file_count = files.len(), dtype = %inner.dtype(), "built MultiFileDataSource (v1)");

        Ok(Arc::new(inner))
    }

    async fn build_v2(
        &self,
        fs: &FileSystemRef,
        files: &[FileListing],
    ) -> VortexResult<DataSourceRef> {
        let mut children = Vec::with_capacity(files.len());
        for file_listing in files {
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
             Either enable the feature or provide a filesystem via .with_filesystem()."
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
