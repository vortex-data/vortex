// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a [`MultiFileDataSource`].

use std::sync::Arc;

use glob::Pattern;
use object_store::ObjectStore;
use object_store::path::Path;
use tracing::debug;
use url::Url;
use vortex_array::expr::Expression;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::multi::DataSourceFactory;
use vortex_scan::multi::MultiDataSource;
use vortex_session::VortexSession;

use super::glob::expand_glob;
use super::glob::list_all;
use super::source::DiscoveredFile;
use super::source::MultiFileDataSource;
use super::source::VortexFileFactory;
use super::source::data_source_from_file;
use crate::OpenOptionsSessionExt;
use crate::VortexOpenOptions;

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
    /// Explicit list of file paths (relative to the object store root).
    Paths(Vec<String>),
    /// A glob pattern to expand against the object store (relative to the object store root).
    Glob(Pattern),
    /// List all files under the prefix that match the configured file extension.
    ListAll,
}

/// Builder for constructing a [`MultiFileDataSource`].
///
/// By default, all files in the object store are discovered (equivalent to a `*` glob).
/// Use [`with_paths`](Self::with_paths) or [`with_glob`](Self::with_glob) to restrict
/// which files are included.
///
/// # Examples
///
/// ```ignore
/// // Discover all files:
/// let ds = MultiFileDataSourceBuilder::new(session, object_store, "s3://bucket/data/")
///     .build()
///     .await?;
///
/// // From a glob pattern:
/// let ds = MultiFileDataSourceBuilder::new(session, object_store, "s3://bucket/data/")
///     .with_glob(glob::Pattern::new("**/*.vortex")?)
///     .with_prefetch(16)
///     .build()
///     .await?;
///
/// // From explicit paths:
/// let ds = MultiFileDataSourceBuilder::new(session, object_store, "s3://bucket/data/")
///     .with_paths(vec!["a.vortex".into(), "b.vortex".into()])
///     .build()
///     .await?;
/// ```
pub struct MultiFileDataSourceBuilder {
    session: VortexSession,
    object_store: Arc<dyn ObjectStore>,
    base_url: Url,
    discovery: FileDiscovery,
    schema_resolution: SchemaResolution,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    prefetch: Option<usize>,
    filter: Option<Expression>,
    dtype: Option<DType>,
}

impl MultiFileDataSource {
    /// Create a new builder from an object store and base URL prefix.
    ///
    /// The `base_url` is used for display/debug purposes. It should typically match the
    /// location of the files (e.g. `"s3://bucket/data/"`).
    pub fn builder(
        session: VortexSession,
        object_store: Arc<dyn ObjectStore>,
        base_url: Url,
    ) -> MultiFileDataSourceBuilder {
        MultiFileDataSourceBuilder {
            session,
            object_store,
            base_url,
            discovery: FileDiscovery::ListAll,
            schema_resolution: SchemaResolution::default(),
            open_options_fn: Arc::new(|opts| opts),
            prefetch: None,
            filter: None,
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

    /// Set explicit file paths (relative to the object store root).
    pub fn with_paths(self, paths: Vec<String>) -> Self {
        self.with_discovery(FileDiscovery::Paths(paths))
    }

    /// Discover files by expanding a glob pattern against the object store.
    ///
    /// The pattern is relative to the object store root
    /// (e.g. `"data/**/*.vortex"`). Expansion happens eagerly during [`build`](Self::build).
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

    /// Set a filter expression for file-level pruning.
    ///
    /// Files whose statistics indicate they cannot match the filter will be skipped entirely.
    /// When no explicit dtype is provided, the first file is always opened (to determine the
    /// schema); deferred files may be skipped if their statistics prove the filter cannot match.
    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Build the [`MultiFileDataSource`].
    ///
    /// If a glob pattern was provided via [`with_glob`](Self::with_glob), it is expanded
    /// eagerly against the object store. If a [`DType`] was provided via
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
            base_url = %self.base_url,
            discovery = %discovery_kind,
            "building MultiFileDataSource"
        );

        debug!(
            "Discovering files in {}: {:?}",
            self.base_url, self.discovery
        );
        let base_url_path = Path::from_url_path(self.base_url.path())
            .map_err(|e| vortex_err!("Invalid base_url '{}': {}", self.base_url, e))?;
        let files = match self.discovery {
            FileDiscovery::Paths(ref paths) => paths
                .iter()
                .map(|path| {
                    // FIXME(ngates): join path to the base_url_path.
                    DiscoveredFile {
                        path: path.clone(),
                        size: None,
                    }
                })
                .collect(),
            FileDiscovery::Glob(ref pattern) => {
                expand_glob(&self.object_store, &base_url_path, pattern).await?
            }
            FileDiscovery::ListAll => list_all(&self.object_store, &base_url_path).await?,
        };

        debug!(
            base_url = %self.base_url,
            file_count = files.len(),
            files = ?files,
            "discovered files"
        );

        if files.is_empty() {
            vortex_bail!(
                "MultiFileDataSource requires at least one file (base_url: {})",
                self.base_url
            );
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
            let first_file = first_options
                .open_object_store(&self.object_store, &first.path)
                .await?;

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
            base_url = %self.base_url,
            file_count,
            dtype = %dtype,
            "built MultiFileDataSource"
        );

        Ok(MultiFileDataSource::new(
            dtype,
            inner,
            self.base_url.to_string(),
            file_count,
        ))
    }

    fn make_factories(&self, files: &[DiscoveredFile]) -> Vec<Arc<dyn DataSourceFactory>> {
        files
            .iter()
            .map(|file| {
                Arc::new(VortexFileFactory {
                    object_store: self.object_store.clone(),
                    file: file.clone(),
                    filter: self.filter.clone(),
                    session: self.session.clone(),
                    open_options_fn: self.open_options_fn.clone(),
                }) as Arc<dyn DataSourceFactory>
            })
            .collect()
    }
}
