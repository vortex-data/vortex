// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
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

        // If there are no glob characters, the pattern is an exact file path. `list` enumerates
        // entries *under* a prefix on a path-segment basis and never yields the prefix itself, so
        // listing an exact path would report an existing file as missing (and could surface prefix
        // collisions such as `foo.vortex.backup` when the caller asked for `foo.vortex`). Use
        // `head` to confirm the file exists and capture its size, yielding a single-element stream
        // when it does and an empty stream when it does not.
        if !pattern.contains(['*', '?', '[']) {
            let pattern = pattern.to_string();
            let stream = stream::once(async move { self.head(&pattern).await })
                .try_filter_map(|listing| async move { Ok(listing) })
                .boxed();
            return Ok(stream);
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

    /// A mock filesystem that resolves exact paths through [`head`](FileSystem::head) and
    /// panics if [`list`](FileSystem::list) is called. This encodes the invariant the fix
    /// depends on: the exact-path glob branch must never list, because an object store's `list`
    /// does not return the exact path of a file.
    #[derive(Debug)]
    struct HeadFileSystem {
        files: Vec<FileListing>,
    }

    impl HeadFileSystem {
        fn new(files: &[(&str, u64)]) -> Self {
            Self {
                files: files
                    .iter()
                    .map(|&(path, size)| FileListing {
                        path: path.to_string(),
                        size: Some(size),
                    })
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl FileSystem for HeadFileSystem {
        fn list(&self, _prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
            vortex_panic!("list() must not be called for an exact path; glob should use head()")
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            Ok(self
                .files
                .iter()
                .find(|listing| listing.path == path)
                .cloned())
        }

        async fn open_read(&self, _path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            vortex_panic!("open_read() should not be called")
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            vortex_panic!("delete() should not be called")
        }
    }

    #[tokio::test]
    async fn test_glob_exact_path_existing_returns_listing_with_size() -> VortexResult<()> {
        let fs = HeadFileSystem::new(&[("data/file.vortex", 1024)]);
        let fs_dyn: &dyn FileSystem = &fs;
        let results: Vec<FileListing> = fs_dyn.glob("data/file.vortex")?.try_collect().await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "data/file.vortex");
        assert_eq!(
            results[0].size,
            Some(1024),
            "exact-path glob should propagate the size reported by head"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_exact_path_missing_returns_empty_stream() -> VortexResult<()> {
        let fs = HeadFileSystem::new(&[]);
        let fs_dyn: &dyn FileSystem = &fs;
        let results: Vec<FileListing> = fs_dyn.glob("data/missing.vortex")?.try_collect().await?;
        assert!(
            results.is_empty(),
            "missing exact path should yield an empty stream"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_glob_exact_path_ignores_prefix_siblings() -> VortexResult<()> {
        // A real object store lists by prefix and would surface `foo.vortex.backup` when asked to
        // list `foo.vortex`. Resolving the exact path via head sidesteps that: only the requested
        // key is returned, and the panicking `list` proves the branch never enumerated.
        let fs = HeadFileSystem::new(&[("foo.vortex", 10), ("foo.vortex.backup", 20)]);
        let fs_dyn: &dyn FileSystem = &fs;
        let results: Vec<FileListing> = fs_dyn.glob("foo.vortex")?.try_collect().await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "foo.vortex");
        assert_eq!(results[0].size, Some(10));
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
