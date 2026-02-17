// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A filesystem abstraction for discovering and opening Vortex files.
//!
//! [`FileSystem`] provides a storage-agnostic interface for listing files under a prefix
//! and opening them for reading. Implementations can target local filesystems, object stores,
//! or any other storage backend.

#[cfg(feature = "object_store")]
pub mod object_store;
mod prefix;

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;

/// A file discovered during listing, with its path and optional size in bytes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileListing {
    /// The file path (relative to the filesystem root).
    pub path: String,
    /// The file size in bytes, if known from the listing metadata.
    pub size: Option<u64>,
}

/// A reference-counted handle to a file system.
pub type FileSystemRef = Arc<dyn FileSystem>;

/// A storage-agnostic filesystem interface for discovering and reading Vortex files.
///
/// Implementations handle the details of a particular storage backend (local disk, S3, GCS, etc.)
/// while consumers work through this uniform interface.
///
/// # Future Work
///
/// An `open_write` method will be added once [`VortexWrite`](vortex_io::VortexWrite) is
/// object-safe (it currently uses `impl Future` return types which prevent trait-object usage).
#[async_trait]
pub trait FileSystem: Send + Sync {
    /// Recursively list files whose paths start with `prefix`.
    ///
    /// When `prefix` is empty, all files are listed. Implementations must recurse into
    /// subdirectories so that the returned stream contains every file reachable under the prefix.
    ///
    /// Returns a stream of [`FileListing`] entries. The stream may yield entries in any order;
    /// callers should sort if deterministic ordering is required.
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>>;

    /// Open a file for reading at the given path.
    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>>;

    /// Expand a glob pattern, returning matching files as a stream.
    ///
    /// Extracts the directory prefix before the first glob character and uses it
    /// to narrow the [`list`](FileSystem::list) call. The full glob pattern is
    /// then applied as a filter over the listed entries.
    ///
    /// Escaped glob characters (`\*`, `\?`, `\[`) are not supported.
    fn glob<'a>(&'a self, pattern: &str) -> VortexResult<BoxStream<'a, VortexResult<FileListing>>> {
        validate_glob(pattern)?;

        let glob_pattern = glob::Pattern::new(pattern)
            .map_err(|e| vortex_err!("Invalid glob pattern '{}': {}", pattern, e))?;

        let listing_prefix = glob_list_prefix(pattern).trim_end_matches('/');

        let stream = self
            .list(listing_prefix)
            .try_filter(move |listing| {
                let matches = glob_pattern.matches(&listing.path);
                async move { matches }
            })
            .into_stream()
            .boxed();

        Ok(stream)
    }
}

/// Returns the directory prefix of a glob pattern.
///
/// Finds the first glob character and returns everything up to and including the last `/`
/// before it. For example, `data/2023/*/logs/*.log` returns `data/2023/`.
fn glob_list_prefix(pattern: &str) -> &str {
    let glob_pos = pattern.find(['*', '?', '[']).unwrap_or(pattern.len());
    match pattern[..glob_pos].rfind('/') {
        Some(slash_pos) => &pattern[..=slash_pos],
        None => "",
    }
}

/// Validates that a glob pattern does not contain escaped glob characters.
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
mod tests {
    use super::*;

    #[test]
    fn test_glob_list_prefix_with_wildcard_in_filename() {
        assert_eq!(glob_list_prefix("folder/file*.txt"), "folder/");
    }

    #[test]
    fn test_glob_list_prefix_with_wildcard_in_directory() {
        assert_eq!(glob_list_prefix("folder/*/file.txt"), "folder/");
    }

    #[test]
    fn test_glob_list_prefix_nested_directories() {
        assert_eq!(glob_list_prefix("data/2023/*/logs/*.log"), "data/2023/");
    }

    #[test]
    fn test_glob_list_prefix_wildcard_at_root() {
        assert_eq!(glob_list_prefix("*.txt"), "");
    }

    #[test]
    fn test_glob_list_prefix_no_wildcards() {
        assert_eq!(
            glob_list_prefix("folder/subfolder/file.txt"),
            "folder/subfolder/"
        );
    }

    #[test]
    fn test_glob_list_prefix_question_mark() {
        assert_eq!(glob_list_prefix("folder/file?.txt"), "folder/");
    }

    #[test]
    fn test_glob_list_prefix_bracket() {
        assert_eq!(glob_list_prefix("folder/file[abc].txt"), "folder/");
    }

    #[test]
    fn test_glob_list_prefix_empty() {
        assert_eq!(glob_list_prefix(""), "");
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
