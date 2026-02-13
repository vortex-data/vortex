// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a [`MultiFileDataSource`].

use std::sync::Arc;

use glob::Pattern;
use object_store::ObjectStore;
use object_store::path::Path;
use tracing::Instrument;
use tracing::debug;
use tracing::info_span;
use vortex_array::expr::Expression;
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
    base_url: String,
    /// The path prefix within the object store used for [`FileDiscovery::ListAll`].
    prefix: Path,
    /// File extension filter for [`FileDiscovery::ListAll`] (e.g. `".vortex"`).
    file_extension: String,
    discovery: FileDiscovery,
    schema_resolution: SchemaResolution,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    prefetch: Option<usize>,
    filter: Option<Expression>,
    dtype: Option<DType>,
}

impl MultiFileDataSourceBuilder {
    /// Create a new builder from an object store and base URL prefix.
    ///
    /// The `base_url` is used for display/debug purposes. It should typically match the
    /// location of the files (e.g. `"s3://bucket/data/"`).
    pub fn new(
        session: VortexSession,
        object_store: Arc<dyn ObjectStore>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            session,
            object_store,
            base_url: base_url.into(),
            prefix: Path::from(""),
            file_extension: ".vortex".to_string(),
            discovery: FileDiscovery::ListAll,
            schema_resolution: SchemaResolution::default(),
            open_options_fn: Arc::new(|opts| opts),
            prefetch: None,
            filter: None,
            dtype: None,
        }
    }

    /// Set the path prefix within the object store.
    ///
    /// This prefix is used by [`FileDiscovery::ListAll`] to scope file listing to a
    /// subdirectory of the object store. For example, if the object store represents
    /// `file:///` and the data lives at `/data/tables/`, set the prefix to `data/tables`.
    pub fn with_prefix(mut self, prefix: impl Into<Path>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set the file extension filter for [`FileDiscovery::ListAll`].
    ///
    /// Only files ending with this extension will be included. Defaults to `".vortex"`.
    pub fn with_file_extension(mut self, ext: impl Into<String>) -> Self {
        self.file_extension = ext.into();
        self
    }

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
    pub async fn build(self) -> VortexResult<MultiFileDataSource> {
        async {
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

            let file_paths = match self.discovery {
                FileDiscovery::Paths(ref paths) => paths.clone(),
                FileDiscovery::Glob(ref pattern) => {
                    expand_glob(&self.object_store, pattern).await?
                }
                FileDiscovery::ListAll => {
                    list_all(&self.object_store, &self.prefix, &self.file_extension).await?
                }
            };

            debug!(
                base_url = %self.base_url,
                file_count = file_paths.len(),
                files = ?file_paths,
                "discovered files"
            );

            if file_paths.is_empty() {
                vortex_bail!(
                    "MultiFileDataSource requires at least one file (base_url: {})",
                    self.base_url
                );
            }

            let file_count = file_paths.len();

            let (dtype, inner) = if let Some(ref dtype) = self.dtype {
                // DType provided externally — all files can be deferred.
                let factories = self.make_factories(&file_paths);
                let inner = MultiDataSource::all_deferred(dtype.clone(), factories, &self.session);
                (dtype.clone(), inner)
            } else {
                // Open the first file eagerly to determine the dtype.
                let first_path = &file_paths[0];
                debug!(path = %first_path, "opening first file eagerly for dtype");
                let first_options = (self.open_options_fn)(self.session.open_options());
                let first_file = first_options
                    .open_object_store(&self.object_store, first_path)
                    .await?;

                let dtype = first_file.dtype().clone();
                debug!(dtype = %dtype, "determined dtype from first file");
                let first_ds = data_source_from_file(&first_file, &self.session)?;

                let factories = self.make_factories(&file_paths[1..]);
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
                self.base_url,
                file_count,
            ))
        }
        .instrument(info_span!("MultiFileDataSourceBuilder::build"))
        .await
    }

    fn make_factories(&self, paths: &[String]) -> Vec<Arc<dyn DataSourceFactory>> {
        paths
            .iter()
            .map(|path| {
                Arc::new(VortexFileFactory {
                    object_store: self.object_store.clone(),
                    path: path.clone(),
                    filter: self.filter.clone(),
                    session: self.session.clone(),
                    open_options_fn: self.open_options_fn.clone(),
                }) as Arc<dyn DataSourceFactory>
            })
            .collect()
    }
}
