// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for constructing a [`MultiFileDataSource`].

use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use tracing::debug;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::api::DataSource;
use vortex_scan::api::DataSourceRef;
use vortex_scan::multi::DataSourceFactory;
use vortex_scan::multi::MultiDataSource;
use vortex_session::VortexSession;

use super::MultiFileDataSource;
use super::session::MultiFileSessionExt;
use crate::OpenOptionsSessionExt;
use crate::VortexOpenOptions;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::filesystem::FileSystemRef;

/// Builder for constructing a [`MultiFileDataSource`].
///
/// The primary interface is [`with_glob_url`](Self::with_glob_url), which accepts a glob
/// pattern (optionally prefixed with `file://`). For non-local filesystems (S3, GCS, etc.),
/// callers must also provide a [`FileSystem`] via [`with_filesystem`](Self::with_filesystem).
///
/// # Examples
///
/// ```ignore
/// // Local files — filesystem is auto-created:
/// let ds = MultiFileDataSource::builder(session)
///     .with_glob_url("/data/warehouse/*.vortex")
///     .build()
///     .await?;
///
/// // S3 — caller provides the filesystem:
/// let ds = MultiFileDataSource::builder(session)
///     .with_filesystem(s3_fs)
///     .with_glob_url("prefix/*.vortex")
///     .build()
///     .await?;
/// ```
pub struct MultiFileDataSourceBuilder {
    session: VortexSession,
    fs: Option<FileSystemRef>,
    glob_url: Option<String>,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
}

impl MultiFileDataSourceBuilder {
    pub(super) fn new(session: VortexSession) -> Self {
        Self {
            session,
            fs: None,
            glob_url: None,
            open_options_fn: Arc::new(|opts| opts),
        }
    }

    /// Set the glob URL for file discovery.
    ///
    /// For local files, this can be a bare path (`/data/*.vortex`) or a `file://` URL.
    /// For remote filesystems, this should be the glob pattern relative to the filesystem
    /// root — the filesystem must be provided via [`with_filesystem`](Self::with_filesystem).
    pub fn with_glob_url(mut self, glob_url: impl Into<String>) -> Self {
        self.glob_url = Some(glob_url.into());
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

    /// Build the [`MultiFileDataSource`].
    ///
    /// Discovers files via glob, opens the first file eagerly to determine the schema,
    /// and creates lazy factories for the remaining files.
    pub async fn build(mut self) -> VortexResult<MultiFileDataSource> {
        let glob_url = self
            .glob_url
            .take()
            .ok_or_else(|| vortex_err!("MultiFileDataSource requires a glob URL"))?;

        let (fs, glob_pattern) = self.resolve_filesystem(&glob_url)?;

        let files: Vec<FileListing> = glob_files(fs.as_ref(), &glob_pattern)?
            .try_collect()
            .await?;

        if files.is_empty() {
            vortex_bail!("No files matched the glob pattern '{}'", glob_url);
        }

        let file_count = files.len();
        debug!(file_count, glob = %glob_url, "discovered files");

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

        Ok(MultiFileDataSource { inner, file_count })
    }

    /// Resolve the filesystem from the builder configuration and glob URL.
    fn resolve_filesystem(&self, glob_url: &str) -> VortexResult<(FileSystemRef, String)> {
        if let Some(ref fs) = self.fs {
            return Ok((fs.clone(), glob_url.to_string()));
        }

        // Auto-create local filesystem for file:// or bare paths.
        let glob_pattern = if let Some(stripped) = glob_url.strip_prefix("file://") {
            stripped.to_string()
        } else if glob_url.starts_with('/')
            || glob_url.starts_with('.')
            || !glob_url.contains("://")
        {
            glob_url.to_string()
        } else {
            vortex_bail!(
                "A filesystem must be provided for non-local URLs. \
                 Use .with_filesystem() for URL: {}",
                glob_url
            );
        };

        let fs = create_local_filesystem(&self.session)?;
        Ok((fs, glob_pattern))
    }
}

/// Creates a local filesystem backed by `object_store::local::LocalFileSystem`.
// TODO(ngates): create a native file system without an object_store dependency.
//  Turns out it's not a trivial change because we have always used object_store with its own
//  coalescing and concurrency configs, so we need to re-tune for local disk.
#[cfg(feature = "object_store")]
fn create_local_filesystem(session: &VortexSession) -> VortexResult<FileSystemRef> {
    use vortex_io::session::RuntimeSessionExt;

    use crate::filesystem::object_store::ObjectStoreFileSystem;

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

// ---------------------------------------------------------------------------
// Internal glob helpers (moved from filesystem/glob.rs)
// ---------------------------------------------------------------------------

/// Expand a glob pattern against a filesystem, returning matching files as a stream.
#[cfg(feature = "object_store")]
fn glob_files<'a>(
    fs: &'a dyn FileSystem,
    pattern: &str,
) -> VortexResult<futures::stream::BoxStream<'a, VortexResult<FileListing>>> {
    validate_glob(pattern)?;

    let glob_pattern = glob::Pattern::new(pattern)
        .map_err(|e| vortex_err!("Invalid glob pattern '{}': {}", pattern, e))?;

    let prefix = list_prefix(pattern);
    let listing_prefix = if prefix.is_empty() {
        None
    } else {
        Some(prefix.trim_end_matches('/'))
    };

    debug!(?listing_prefix, pattern, "expanding glob");

    let stream = fs
        .list(listing_prefix)
        .try_filter(move |listing| {
            let matches = glob_pattern.matches(&listing.path);
            async move { matches }
        })
        .into_stream();

    Ok(Box::pin(stream))
}

#[cfg(not(feature = "object_store"))]
fn glob_files<'a>(
    _fs: &'a dyn FileSystem,
    _pattern: &str,
) -> VortexResult<futures::stream::BoxStream<'a, VortexResult<FileListing>>> {
    vortex_bail!("The 'object_store' feature is required for glob-based file discovery");
}

/// Returns the list prefix for a path pattern containing glob characters.
#[cfg(feature = "object_store")]
fn list_prefix(pattern: &str) -> &str {
    let glob_pos = pattern.find(['*', '?', '[']).unwrap_or(pattern.len());
    match pattern[..glob_pos].rfind('/') {
        Some(slash_pos) => &pattern[..=slash_pos],
        None => "",
    }
}

/// Validates that a glob pattern does not contain escaped glob characters.
#[cfg(feature = "object_store")]
fn validate_glob(pattern: &str) -> VortexResult<()> {
    for escape_pattern in ["\\*", "\\?", "\\["] {
        if pattern.contains(escape_pattern) {
            vortex_bail!(
                "Escaped glob characters are not allowed in patterns. Found '{}' in: {}",
                escape_pattern,
                pattern
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "object_store")]
mod tests {
    use super::*;

    #[test]
    fn test_list_prefix_with_wildcard_in_filename() {
        assert_eq!(list_prefix("folder/file*.txt"), "folder/");
    }

    #[test]
    fn test_list_prefix_with_wildcard_in_directory() {
        assert_eq!(list_prefix("folder/*/file.txt"), "folder/");
    }

    #[test]
    fn test_list_prefix_nested_directories() {
        assert_eq!(list_prefix("data/2023/*/logs/*.log"), "data/2023/");
    }

    #[test]
    fn test_list_prefix_wildcard_at_root() {
        assert_eq!(list_prefix("*.txt"), "");
    }

    #[test]
    fn test_list_prefix_no_wildcards() {
        assert_eq!(
            list_prefix("folder/subfolder/file.txt"),
            "folder/subfolder/"
        );
    }

    #[test]
    fn test_list_prefix_question_mark() {
        assert_eq!(list_prefix("folder/file?.txt"), "folder/");
    }

    #[test]
    fn test_list_prefix_bracket() {
        assert_eq!(list_prefix("folder/file[abc].txt"), "folder/");
    }

    #[test]
    fn test_list_prefix_empty() {
        assert_eq!(list_prefix(""), "");
    }

    #[test]
    fn test_validate_glob_valid() -> VortexResult<()> {
        validate_glob("path/*.txt")?;
        validate_glob("path/to/**/*.vortex")?;
        Ok(())
    }

    #[test]
    fn test_validate_glob_escaped_asterisk() {
        assert!(validate_glob("path\\*.txt").is_err());
    }

    #[test]
    fn test_validate_glob_escaped_question() {
        assert!(validate_glob("path\\?.txt").is_err());
    }

    #[test]
    fn test_validate_glob_escaped_bracket() {
        assert!(validate_glob("path\\[test].txt").is_err());
    }
}
