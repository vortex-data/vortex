// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a [`MultiFileDataSource`].

use std::sync::Arc;

use glob::Pattern;
use tracing::debug;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scan::multi::DataSourceFactory;
use vortex_scan::multi::MultiDataSource;
use vortex_session::VortexSession;

use super::glob::expand_glob;
use super::glob::list_all;
use super::source::MultiFileDataSource;
use super::source::VortexFileFactory;
use super::source::data_source_from_file;
use crate::OpenOptionsSessionExt;
use crate::VortexOpenOptions;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystemRef;

/// How to handle schema differences across files in a [`MultiFileDataSource`].
#[derive(Debug, Clone, Default)]
pub enum SchemaResolution {
    /// All files must have exactly the same [`DType`](vortex_dtype::DType). Error on mismatch.
    #[default]
    Exact,
    /// Unify schemas: allow missing columns (filled with nulls) and compatible type upcasts.
    ///
    /// **Not yet implemented** — will return an error at build time.
    Union,
}

/// How files are discovered for a [`MultiFileDataSource`].
#[derive(Debug)]
pub enum FileDiscovery {
    /// Explicit list of file paths.
    Paths(Vec<String>),
    /// A glob pattern to expand against the filesystem.
    Glob(Pattern),
    /// List all files in the filesystem.
    ListAll,
}

/// Builder for constructing a [`MultiFileDataSource`].
///
/// By default, all files are discovered by listing the filesystem (equivalent to `ListAll`).
/// Use [`with_paths`](Self::with_paths) or [`with_glob`](Self::with_glob) to restrict
/// which files are included.
///
/// To scope the data source to a subdirectory, wrap the filesystem with
/// [`FileSystem::prefix`](crate::filesystem::FileSystem::prefix) before passing it to the builder.
///
/// # Examples
///
/// ```ignore
/// // Discover all files under a prefix:
/// let fs = fs.prefix("data/".into());
/// let ds = MultiFileDataSource::builder(session, fs)
///     .build()
///     .await?;
///
/// // From a glob pattern:
/// let fs = fs.prefix("data/".into());
/// let ds = MultiFileDataSource::builder(session, fs)
///     .with_glob(glob::Pattern::new("**/*.vortex")?)
///     .with_prefetch(16)
///     .build()
///     .await?;
///
/// // From explicit paths:
/// let ds = MultiFileDataSource::builder(session, fs)
///     .with_paths(vec!["a.vortex".into(), "b.vortex".into()])
///     .build()
///     .await?;
/// ```
pub struct MultiFileDataSourceBuilder {
    session: VortexSession,
    fs: FileSystemRef,
    discovery: FileDiscovery,
    schema_resolution: SchemaResolution,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    prefetch: Option<usize>,
    dtype: Option<DType>,
}

impl MultiFileDataSource {
    /// Create a new builder from a filesystem.
    ///
    /// To scope the data source to a subdirectory, wrap the filesystem with
    /// [`FileSystem::prefix`](crate::filesystem::FileSystem::prefix).
    pub fn builder(
        session: VortexSession,
        fs: FileSystemRef,
    ) -> MultiFileDataSourceBuilder {
        MultiFileDataSourceBuilder {
            session,
            fs,
            discovery: FileDiscovery::ListAll,
            schema_resolution: SchemaResolution::default(),
            open_options_fn: Arc::new(|opts| opts),
            prefetch: None,
            dtype: None,
        }
    }
}

impl MultiFileDataSourceBuilder {
    /// Set how files are discovered.
    pub fn with_discovery(mut self, discovery: FileDiscovery) -> Self {
        self.discovery = discovery;
        self
    }

    /// Set explicit file paths.
    pub fn with_paths(self, paths: Vec<String>) -> Self {
        self.with_discovery(FileDiscovery::Paths(paths))
    }

    /// Discover files by expanding a glob pattern against the filesystem.
    ///
    /// The pattern is relative to the filesystem root
    /// (e.g. `"**/*.vortex"`). Expansion happens eagerly during [`build`](Self::build).
    pub fn with_glob(self, pattern: Pattern) -> Self {
        self.with_discovery(FileDiscovery::Glob(pattern))
    }

    /// Set how schema differences across files should be handled.
    pub fn with_schema_resolution(mut self, resolution: SchemaResolution) -> Self {
        self.schema_resolution = resolution;
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

    /// Set the prefetch concurrency for lazy file opening.
    ///
    /// Higher values overlap more file-opening I/O with split execution but use more memory
    /// for in-flight metadata.
    ///
    /// Defaults to  [`std::thread::available_parallelism`].
    pub fn with_prefetch(mut self, prefetch: usize) -> Self {
        self.prefetch = Some(prefetch);
        self
    }

    /// Set an explicit [`DType`] for the data source.
    ///
    /// When provided, no file needs to be eagerly opened to determine the schema — all files
    /// are deferred and opened lazily during scanning. This is useful when the caller already
    /// knows the schema (e.g. from a catalog or a prior scan). Combine with
    /// [`with_open_options`](Self::with_open_options) to pass the dtype through to
    /// [`VortexOpenOptions::with_dtype`](crate::VortexOpenOptions::with_dtype) on each file
    /// open for additional I/O savings.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Build the [`MultiFileDataSource`].
    ///
    /// If a glob pattern was provided via [`with_glob`](Self::with_glob), it is expanded
    /// eagerly against the filesystem. If a [`DType`] was provided via
    /// [`with_dtype`](Self::with_dtype), all files are opened lazily during scanning.
    /// Otherwise, the first file is opened eagerly to determine the schema.
    #[tracing::instrument(name = "MultiFileDataSourceBuilder::build", skip(self))]
    pub async fn build(self) -> VortexResult<MultiFileDataSource> {
        if matches!(self.schema_resolution, SchemaResolution::Union) {
            vortex_bail!("SchemaResolution::Union is not yet implemented");
        }

        let discovery_kind = match &self.discovery {
            FileDiscovery::Paths(p) => format!("paths({})", p.len()),
            FileDiscovery::Glob(g) => format!("glob({})", g.as_str()),
            FileDiscovery::ListAll => "list_all".to_string(),
        };
        debug!(
            discovery = %discovery_kind,
            "building MultiFileDataSource"
        );

        let files = match self.discovery {
            FileDiscovery::Paths(ref paths) => paths
                .iter()
                .map(|path| FileListing {
                    path: path.clone(),
                    size: None,
                })
                .collect(),
            FileDiscovery::Glob(ref pattern) => expand_glob(&self.fs, pattern).await?,
            FileDiscovery::ListAll => list_all(&self.fs).await?,
        };

        debug!(
            file_count = files.len(),
            files = ?files,
            "discovered files"
        );

        if files.is_empty() {
            vortex_bail!("MultiFileDataSource requires at least one file");
        }

        let file_count = files.len();

        let (dtype, inner) = if let Some(ref dtype) = self.dtype {
            // DType provided externally — all files can be deferred.
            let factories = self.make_factories(&files);
            let inner = MultiDataSource::all_deferred(dtype.clone(), factories, &self.session);
            (dtype.clone(), inner)
        } else {
            let first = &files[0];
            debug!(path = %first.path, "opening first file eagerly for dtype");
            let mut first_options = (self.open_options_fn)(self.session.open_options());
            if let Some(size) = first.size {
                first_options = first_options.with_file_size(size);
            }
            let source = self.fs.open_read(&first.path).await?;
            let first_file = first_options.open(source).await?;

            let dtype = first_file.dtype().clone();
            debug!(dtype = %dtype, "determined dtype from first file");
            let first_ds = data_source_from_file(&first_file, &self.session)?;

            let factories = self.make_factories(&files[1..]);
            let inner = MultiDataSource::lazy(first_ds, factories, &self.session);
            (dtype, inner)
        };

        let inner = match self.prefetch {
            Some(prefetch) => inner.with_prefetch(prefetch),
            None => inner,
        };

        debug!(
            file_count,
            dtype = %dtype,
            "built MultiFileDataSource"
        );

        Ok(MultiFileDataSource::new(dtype, inner, file_count))
    }

    fn make_factories(&self, files: &[FileListing]) -> Vec<Arc<dyn DataSourceFactory>> {
        files
            .iter()
            .map(|file| {
                Arc::new(VortexFileFactory {
                    fs: self.fs.clone(),
                    file: file.clone(),
                    session: self.session.clone(),
                    open_options_fn: self.open_options_fn.clone(),
                }) as Arc<dyn DataSourceFactory>
            })
            .collect()
    }
}
