// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a [`MultiLayoutDataSource`] from multiple Vortex files.

pub(crate) mod scan_v2;
mod session;

use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use session::MultiFileSessionExt;
use tracing::debug;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::InstrumentedReadAt;
use vortex_io::VortexReadAt;
use vortex_io::filesystem::FileListing;
use vortex_io::filesystem::FileSystemRef;
use vortex_layout::LayoutReaderRef;
use vortex_layout::scan::multi::LayoutReaderFactory;
use vortex_layout::scan::multi::MultiLayoutDataSource;
use vortex_layout::scan::v2::scan2_enabled;
use vortex_metrics::Label;
use vortex_metrics::MetricsRegistry;
use vortex_scan::DataSource;
use vortex_scan::DataSourceRef;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::VortexOpenOptions;

const PATH_LABEL: &str = "file_path";

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
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

impl MultiFileDataSource {
    /// Create a new [`MultiFileDataSource`] builder.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            glob_sources: Vec::new(),
            open_options_fn: Arc::new(|opts| opts),
            metrics_registry: None,
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

    /// Configure a shared metrics registry for all files opened by this data source.
    ///
    /// This instruments both the underlying [`VortexReadAt`] and the Vortex segment source so
    /// callers can inspect read sizes, read durations, segment request coalescing, and segment
    /// cache behavior for scans that use this data source.
    pub fn with_metrics_registry(mut self, metrics_registry: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics_registry);
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

        // Open first file eagerly for dtype.
        let (first_file_listing, first_fs) = &all_files[0];
        let open_fn = self.open_options_fn.as_ref();
        let first_file = open_file(
            first_fs,
            first_file_listing,
            &self.session,
            self.metrics_registry.as_ref(),
            open_fn,
        )
        .await?;
        let first_reader = first_file.layout_reader()?;

        let factories: Vec<Arc<dyn LayoutReaderFactory>> = all_files[1..]
            .iter()
            .map(|(file, fs)| {
                Arc::new(VortexFileReaderFactory {
                    fs: Arc::clone(fs),
                    file: file.clone(),
                    session: self.session.clone(),
                    open_options_fn: Arc::clone(&self.open_options_fn),
                    metrics_registry: self.metrics_registry.clone(),
                }) as Arc<dyn LayoutReaderFactory>
            })
            .collect();

        let inner = MultiLayoutDataSource::new_with_first(first_reader, factories, &self.session);

        debug!(file_count, dtype = %inner.dtype(), "built MultiFileDataSource");

        Ok(inner)
    }

    /// Build the [`DataSource`] selected by `VORTEX_SCAN_IMPL`.
    ///
    /// The default is the existing LayoutReader-backed scan. Setting
    /// `VORTEX_SCAN_IMPL=v2` (or `scan2`/`scan3`/`native`) builds the ScanPlan-backed V2 scan.
    pub async fn build_data_source(self) -> VortexResult<DataSourceRef> {
        if scan2_enabled()? {
            Ok(Arc::new(scan_v2::build_scan_plan_data_source(self).await?))
        } else {
            Ok(Arc::new(self.build().await?))
        }
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
    metrics_registry: Option<&Arc<dyn MetricsRegistry>>,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<VortexFile> {
    tracing::trace!(path = %file.path, "opening vortex file");

    // Open the reader first so we can use its URI as the cache key.
    // The URI includes the full path (with any filesystem prefix), making it unique
    // even when different PrefixFileSystem instances strip paths to the same relative name.
    let source = fs.open_read(&file.path).await?;
    let labels = vec![Label::new(PATH_LABEL, file.path.clone())];
    let source = if let Some(metrics_registry) = metrics_registry {
        Arc::new(InstrumentedReadAt::new_with_labels(
            source,
            metrics_registry.as_ref(),
            labels.clone(),
        )) as Arc<dyn VortexReadAt>
    } else {
        source
    };
    let cache_key = source
        .uri()
        .map(|u| u.to_string())
        .unwrap_or_else(|| file.path.clone());

    // Build open options. The DashMap Ref from multi_file() must not live across an await,
    // so we scope the cache lookup in a block.
    let options = {
        let mut options = open_options_fn(session.open_options());
        if let Some(metrics_registry) = metrics_registry {
            options = options
                .with_metrics_registry(Arc::clone(metrics_registry))
                .with_labels(labels);
        }
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

/// A [`LayoutReaderFactory`] that lazily opens a single Vortex file and returns its layout reader.
struct VortexFileReaderFactory {
    fs: FileSystemRef,
    file: FileListing,
    session: VortexSession,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

#[async_trait]
impl LayoutReaderFactory for VortexFileReaderFactory {
    async fn open(&self) -> VortexResult<Option<LayoutReaderRef>> {
        let file = open_file(
            &self.fs,
            &self.file,
            &self.session,
            self.metrics_registry.as_ref(),
            self.open_options_fn.as_ref(),
        )
        .await?;

        Ok(Some(file.layout_reader()?))
    }
}
