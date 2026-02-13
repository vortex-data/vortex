// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! URL glob expansion for discovering Vortex files in object stores.
//!
//! Uses [`object_store::ObjectStore::list()`] with a computed prefix and client-side glob
//! filtering to discover files matching a pattern.

use std::sync::Arc;

use futures::StreamExt;
use glob::Pattern;
use object_store::ObjectStore;
use tracing::Instrument;
use tracing::debug;
use tracing::info_span;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// List all files under `prefix` in the object store that end with `file_extension`,
/// returning sorted file paths relative to the store root.
pub(super) async fn list_all(
    object_store: &Arc<dyn ObjectStore>,
    prefix: &object_store::path::Path,
    file_extension: &str,
) -> VortexResult<Vec<String>> {
    let prefix_str = prefix.as_ref();
    async {
        debug!(prefix = prefix_str, file_extension, "listing all files");

        let mut paths: Vec<String> = object_store
            .list(Some(prefix))
            .filter_map(|result| async {
                match result {
                    Ok(meta) => {
                        let path_str = meta.location.to_string();
                        path_str.ends_with(file_extension).then_some(path_str)
                    }
                    Err(_) => None,
                }
            })
            .collect()
            .await;

        paths.sort();
        debug!(
            prefix = prefix_str,
            file_extension,
            file_count = paths.len(),
            "listed all files"
        );
        Ok(paths)
    }
    .instrument(info_span!("list_all", prefix = prefix_str, file_extension))
    .await
}

/// Expand a glob pattern against an [`ObjectStore`], returning matching file paths relative
/// to the store root.
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
pub(super) async fn expand_glob(
    object_store: &Arc<dyn ObjectStore>,
    pattern: &Pattern,
) -> VortexResult<Vec<String>> {
    let glob_str = pattern.as_str();

    async {
        validate_glob(glob_str)?;

        let prefix = list_prefix(glob_str);
        let prefix_path = object_store::path::Path::from(prefix);

        debug!(glob = glob_str, prefix, "expanding glob");

        let mut paths: Vec<String> = object_store
            .list(Some(&prefix_path))
            .filter_map(|result| async {
                match result {
                    Ok(meta) => {
                        let path_str = meta.location.to_string();
                        pattern.matches(&path_str).then_some(path_str)
                    }
                    Err(_) => None,
                }
            })
            .collect()
            .await;

        paths.sort();
        debug!(glob = glob_str, file_count = paths.len(), "expanded glob");
        Ok(paths)
    }
    .instrument(info_span!("expand_glob", glob = glob_str))
    .await
}

/// Returns the list prefix for a path pattern containing glob characters.
///
/// The prefix is the directory path up to the first glob character, which is used as the
/// `list()` prefix to narrow the object store listing.
///
/// # Examples
///
/// - `"path/to/file_*.txt"` → `"path/to/"`
/// - `"*.txt"` → `""`
/// - `"path/to/specific/file.txt"` → `"path/to/specific/"`
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
