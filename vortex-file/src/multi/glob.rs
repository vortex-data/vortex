// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Glob expansion for discovering Vortex files via a [`FileSystem`].
//!
//! Uses [`FileSystem::list()`] with a computed prefix and client-side glob
//! filtering to discover files matching a pattern.

use glob::Pattern;
use tracing::debug;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::filesystem::FileListing;
use crate::filesystem::FileSystemRef;

/// List all files in the filesystem, returning sorted [`FileListing`]s.
#[tracing::instrument(name = "list_all", skip(fs))]
pub(super) async fn list_all(fs: &FileSystemRef) -> VortexResult<Vec<FileListing>> {
    use futures::TryStreamExt;

    debug!("listing all files");

    let mut files: Vec<FileListing> = fs.list(None).try_collect().await?;

    files.sort();
    debug!(file_count = files.len(), "listed all files");
    Ok(files)
}

/// Expand a glob pattern against a [`FileSystem`], returning matching
/// [`FileListing`]s with path and size.
///
/// The `pattern` is matched against file paths relative to the filesystem root
/// (e.g. `"**/*.vortex"`).
///
/// # Algorithm
///
/// 1. Find the first glob character (`*`, `?`, `[`) in the pattern.
/// 2. Use everything before it (up to the last `/`) as the list prefix.
/// 3. List files with that prefix.
/// 4. Filter using [`glob::Pattern`] matching.
/// 5. Return sorted file paths.
#[tracing::instrument(name = "expand_glob", skip(fs))]
pub(super) async fn expand_glob(
    fs: &FileSystemRef,
    pattern: &Pattern,
) -> VortexResult<Vec<FileListing>> {
    use futures::TryStreamExt;

    let glob_str = pattern.as_str();

    validate_glob(glob_str)?;

    // Extract the static prefix from the glob pattern to narrow the listing.
    let prefix = list_prefix(glob_str);
    let listing_prefix = if prefix.is_empty() {
        None
    } else {
        Some(prefix.trim_end_matches('/'))
    };

    debug!(?listing_prefix, "expanding glob");

    let mut files: Vec<FileListing> = fs
        .list(listing_prefix)
        .try_filter_map(|listing| async move {
            Ok(pattern.matches(&listing.path).then_some(listing))
        })
        .try_collect()
        .await?;

    files.sort();
    debug!(file_count = files.len(), "expanded glob");
    Ok(files)
}

/// Returns the list prefix for a path pattern containing glob characters.
///
/// The prefix is the directory path up to the first glob character, which is used to narrow
/// the `list()` call on the filesystem.
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
