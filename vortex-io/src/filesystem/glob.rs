// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;

impl dyn FileSystem + '_ {
    /// Expand a glob pattern, returning matching files as a stream.
    ///
    /// Extracts the directory prefix before the first glob character and uses it
    /// to narrow the [`list`](FileSystem::list) call. The full glob pattern is
    /// then applied as a filter over the listed entries.
    ///
    /// Escaped glob characters (`\*`, `\?`, `\[`) are not supported.
    pub fn glob(&self, pattern: &str) -> VortexResult<BoxStream<'_, VortexResult<FileListing>>> {
        validate_glob(pattern)?;

        // If there are no glob characters, the pattern is an exact file path.
        // Return it directly without listing the filesystem.
        if !pattern.contains(['*', '?', '[']) {
            let listing = FileListing {
                path: pattern.to_string(),
                size: None,
            };
            return Ok(futures::stream::once(async { Ok(listing) }).boxed());
        }

        let glob_pattern = glob::Pattern::new(pattern)
            .map_err(|e| vortex_err!("Invalid glob pattern '{}': {}", pattern, e))?;

        let listing_prefix = glob_list_prefix(pattern).trim_end_matches('/');

        tracing::debug!(
            "Performing glob with pattern '{}' and listing prefix '{}'",
            pattern,
            listing_prefix
        );
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
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::TryStreamExt;
    use vortex_error::vortex_panic;

    use super::*;
    use crate::VortexReadAt;
    use crate::filesystem::FileSystem;

    /// A mock filesystem that panics if `list` is called.
    #[derive(Debug)]
    struct NoListFileSystem;

    #[async_trait]
    impl FileSystem for NoListFileSystem {
        fn list(&self, _prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
            vortex_panic!("list() should not be called for exact paths")
        }

        async fn open_read(&self, _path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            vortex_panic!("open_read() should not be called")
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            vortex_panic!("delete() should not be called")
        }
    }

    #[tokio::test]
    async fn test_glob_exact_path_skips_list() -> VortexResult<()> {
        let fs: &dyn FileSystem = &NoListFileSystem;
        let results: Vec<FileListing> = fs.glob("data/file.vortex")?.try_collect().await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "data/file.vortex");
        assert_eq!(results[0].size, None);
        Ok(())
    }

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
