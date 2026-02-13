// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! URL glob expansion for discovering Vortex files in object stores.
//!
//! Uses [`object_store::ObjectStore::list()`] with a computed prefix and client-side glob
//! filtering to discover files matching a pattern.

use std::sync::Arc;

use futures::StreamExt;
use futures::TryStreamExt;
use glob::Pattern;
use object_store::ObjectStore;
use object_store::path::Path;
use tracing::debug;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::source::DiscoveredFile;

/// List all files under `prefix` in the object store, returning sorted
/// [`DiscoveredFile`]s with path and size.
#[tracing::instrument(name = "list_all", skip(object_store))]
pub(super) async fn list_all(
    object_store: &Arc<dyn ObjectStore>,
    prefix: &Path,
) -> VortexResult<Vec<DiscoveredFile>> {
    debug!("listing all files");

    let mut files: Vec<DiscoveredFile> = object_store
        .list(Some(prefix))
        .map_ok(|meta| DiscoveredFile {
            path: meta.location.to_string(),
            size: Some(meta.size),
        })
        .try_collect()
        .await?;

    files.sort();
    debug!(file_count = files.len(), "listed all files");
    Ok(files)
}

/// Expand a glob pattern against an [`ObjectStore`], returning matching
/// [`DiscoveredFile`]s with path and size.
///
/// The `glob_pattern` should be a path pattern (not a full URL) relative to the store root,
/// e.g. `"data/year=2024/**/*.vortex"`.
///
/// # Algorithm
///
/// 1. Find the first glob character (`*`, `?`, `[`) in the pattern.
/// 2. Use everything before it (up to the last `/`) as the list prefix.
/// 3. List objects with that prefix.
/// 4. Filter using [`glob::Pattern`] matching.
/// 5. Return sorted file paths.
#[tracing::instrument(name = "expand_glob", skip(object_store))]
pub(super) async fn expand_glob(
    object_store: &Arc<dyn ObjectStore>,
    base_url_path: &Path,
    pattern: &Pattern,
) -> VortexResult<Vec<DiscoveredFile>> {
    let glob_str = pattern.as_str();

    validate_glob(glob_str)?;

    // Extract the static prefix from the glob pattern to narrow the listing.
    let prefix = list_prefix(glob_str);
    let listing_path = if prefix.is_empty() {
        base_url_path.clone()
    } else {
        Path::from(format!(
            "{}/{}",
            base_url_path.as_ref().trim_end_matches('/'),
            prefix.trim_end_matches('/')
        ))
    };
    let base_prefix = base_url_path.as_ref();

    debug!(%base_url_path, %listing_path, "expanding glob");

    let mut files: Vec<DiscoveredFile> = object_store
        .list(Some(&listing_path))
        .filter_map(|result| async {
            match result {
                Ok(meta) => {
                    let path_str = meta.location.to_string();
                    let relative = path_str
                        .strip_prefix(base_prefix)
                        .map(|s| s.trim_start_matches('/'))
                        .unwrap_or(&path_str);
                    pattern.matches(relative).then_some(DiscoveredFile {
                        path: path_str,
                        size: Some(meta.size),
                    })
                }
                // FIXME(ngates): do not ignore errors
                Err(_) => None,
            }
        })
        .collect()
        .await;

    files.sort();
    debug!(file_count = files.len(), "expanded glob");
    Ok(files)
}

/// Returns the list prefix for a path pattern containing glob characters.
///
/// The prefix is the directory path up to the first glob character, which is used to narrow
/// the `list()` call on the object store.
///
/// # Examples
///
/// - `"path/to/file_*.txt"` -> `"path/to/"`
/// - `"*.txt"` -> `""`
/// - `"path/to/specific/file.txt"` -> `"path/to/specific/"`
fn list_prefix(pattern: &str) -> &str {
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
