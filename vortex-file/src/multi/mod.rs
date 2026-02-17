// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a [`MultiDataSource`] from multiple Vortex files.

mod session;

use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use session::MultiFileSessionExt;
use tracing::debug;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::api::DataSource;
use vortex_scan::api::DataSourceRef;
use vortex_scan::multi::DataSourceFactory;
use vortex_scan::multi::MultiDataSource;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexOpenOptions;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystemRef;

/// A builder that discovers multiple Vortex files from a glob pattern and constructs a
/// [`MultiDataSource`] to scan them as a single data source.
///
/// The primary interface is [`Self::with_glob`], which accepts a glob
/// pattern (optionally prefixed with `file://`). For non-local filesystems (S3, GCS, etc.),
/// callers must also provide a [`FileSystem`] via [`Self::with_filesystem`]).
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
}

impl MultiFileDataSource {
    /// Create a new [`MultiFileDataSource`] builder.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            fs: None,
            glob: None,
            open_options_fn: Arc::new(|opts| opts),
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

    /// Build the [`MultiDataSource`].
    ///
    /// Discovers files via glob, opens the first file eagerly to determine the schema,
    /// and creates lazy factories for the remaining files.
    pub async fn build(mut self) -> VortexResult<MultiDataSource> {
        let glob = self
            .glob
            .take()
            .ok_or_else(|| vortex_err!("MultiFileDataSource requires a glob URL"))?;

        let fs = self
            .fs
            .map(Ok)
            .unwrap_or_else(|| create_local_filesystem(&self.session))?;
        let files: Vec<FileListing> = fs.glob(&glob)?.try_collect().await?;

        if files.is_empty() {
            vortex_bail!("No files matched the glob pattern '{}'", glob);
        }

        let file_count = files.len();
        debug!(file_count, glob = %glob, "discovered files");

        // Open first file eagerly for dtype.
        let first_file =
            open_file(&fs, &files[0], &self.session, self.open_options_fn.as_ref()).await?;
        let first_ds = first_file.data_source()?;

        let factories: Vec<Arc<dyn DataSourceFactory>> = files[1..]
            .iter()
            .map(|f| {
                Arc::new(VortexFileFactory {
                    fs: fs.clone(),
                    file: f.clone(),
                    session: self.session.clone(),
                    open_options_fn: self.open_options_fn.clone(),
                }) as Arc<dyn DataSourceFactory>
            })
            .collect();

        let inner = MultiDataSource::lazy(first_ds, factories, &self.session);

        debug!(file_count, dtype = %inner.dtype(), "built MultiFileDataSource");

        Ok(inner)
    }
}

/// Creates a local filesystem backed by `object_store::local::LocalFileSystem`.
// TODO(ngates): create a native file system without an object_store dependency.
//  Turns out it's not a trivial change because we have always used object_store with its own
//  coalescing and concurrency configs, so we need to re-tune for local disk.
fn create_local_filesystem(session: &VortexSession) -> VortexResult<FileSystemRef> {
    if cfg!(feature = "object_store") {
        use vortex_io::session::RuntimeSessionExt;

        use crate::filesystem::object_store::ObjectStoreFileSystem;

        let store = Arc::new(object_store::local::LocalFileSystem::default());
        let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(store, session.handle()));
        Ok(fs)
    } else {
        vortex_bail!(
            "The 'object_store' feature is required for automatic local filesystem creation. \
             Either enable the feature or provide a filesystem via .with_filesystem()."
        );
    }
}

/// Open a single Vortex file, checking the session's footer cache.
async fn open_file(
    fs: &FileSystemRef,
    file: &FileListing,
    session: &VortexSession,
    open_options_fn: &(dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync),
) -> VortexResult<crate::VortexFile> {
    debug!(path = %file.path, "opening vortex file");

    // Build open options. The DashMap Ref from multi_file() must not live across an await,
    // so we scope the cache lookup in a block.
    let options = {
        let mut options = open_options_fn(session.open_options());
        if let Some(size) = file.size {
            options = options.with_file_size(size);
        }
        if let Some(footer) = session.multi_file().get_footer(&file.path) {
            options = options.with_footer(footer);
        }
        options
    };

    let source = fs.open_read(&file.path).await?;
    let vortex_file = options.open(source).await?;

    // Store footer in cache (scoped to avoid holding the Ref across subsequent code).
    session
        .multi_file()
        .put_footer(&file.path, vortex_file.footer().clone());
    Ok(vortex_file)
}

/// A [`DataSourceFactory`] that lazily opens a single Vortex file.
struct VortexFileFactory {
    fs: FileSystemRef,
    file: FileListing,
    session: VortexSession,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
}

#[async_trait]
impl DataSourceFactory for VortexFileFactory {
    async fn open(&self) -> VortexResult<Option<DataSourceRef>> {
        let file = open_file(
            &self.fs,
            &self.file,
            &self.session,
            self.open_options_fn.as_ref(),
        )
        .await?;
        Ok(Some(file.data_source()?))
    }
}
