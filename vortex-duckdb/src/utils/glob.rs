// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::StreamExt;
use object_store::ObjectMeta;
use url::Url;
use vortex::error::{VortexResult, vortex_bail, vortex_err};

use super::object_store::s3_store;

/// Expand a glob pattern into a list of URLs.
///
/// Example: s3://bucket/files/*.vortex -> (urls, Some(object_store_metadata))
pub async fn expand_glob<T: AsRef<str>>(
    url_glob: T,
) -> VortexResult<(Vec<Url>, Option<Vec<ObjectMeta>>)> {
    let url_str = url_glob.as_ref();
    // We prefer using string prefix matching here over extracting a URL scheme
    // as local files with an absolute path but without the file:// prefix can't
    // be parsed into a URL.
    match &url_str[..url_str.len().min(5)] {
        "s3://" => s3::expand_glob(&url_glob).await,
        "gs://" => vortex_bail!("GCS glob expansion not yet implemented"),
        _ => local_filesystem::expand_glob(url_str),
    }
}

mod s3 {
    use object_store::ObjectMeta;

    use super::*;

    /// Expand a glob pattern into a list of S3 URLs.
    ///
    /// Makes a single request based on the position of the first glob character
    /// and filters the results on the client side. In case no glob characters
    /// are provided, the last directory in the path is used as the list prefix.
    pub(super) async fn expand_glob<T: AsRef<str>>(
        url_glob: T,
    ) -> VortexResult<(Vec<Url>, Option<Vec<ObjectMeta>>)> {
        validate_glob(&url_glob)?;
        assert_eq!("s3://", &url_glob.as_ref()[..5]);
        let url = Url::parse(url_glob.as_ref())?;

        let bucket = url
            .host_str()
            .ok_or_else(|| vortex_err!("Failed to extract bucket name from URL: {url}"))?;

        let list_prefix = list_prefix(url_path(&url)?);
        let object_store = s3_store(bucket)?;

        // The AWS S3 `ListObjectsV2` API returns multiple objects per HTTP
        // request (up to 1000 by default), but the object store stream
        // interface yields them to you one at a time.
        let stream = object_store.list(Some(&object_store::path::Path::from(list_prefix)));

        let glob_pattern = glob::Pattern::new(url_glob.as_ref())
            .map_err(|e| vortex_err!("Invalid glob pattern: {}", e))?;

        let matching_paths = process_object_store_stream(stream, &glob_pattern, bucket).await?;

        let (urls, metadata) = matching_paths.into_iter().unzip();
        Ok((urls, Some(metadata)))
    }

    /// Validates that a glob pattern does not contain escaped glob characters.
    /// Returns an error if backslash-escaped characters like \*, \?, etc. are found.
    pub(super) fn validate_glob<T: AsRef<str>>(pattern: T) -> VortexResult<()> {
        let pattern_str = pattern.as_ref();

        // Check for backslash escape sequences.
        for escape_pattern in ["\\*", "\\?", "\\["] {
            if pattern_str.contains(escape_pattern) {
                vortex_bail!(
                    "Escaped glob characters are not allowed in patterns. Found '{}' in: {}",
                    escape_pattern,
                    pattern_str
                );
            }
        }

        Ok(())
    }

    /// Returns the path from an S3 URL.
    ///
    /// Example: "s3://bucket/path/to/file.txt" -> "path/to/file.txt"
    pub(super) fn url_path(url: &Url) -> VortexResult<&str> {
        url.path()
            .strip_prefix("/")
            .ok_or_else(|| vortex_err!("Invalid URL: {url}"))
    }

    /// Returns the list prefix for a URL path which can contain glob characters.
    ///
    /// Unlike `aws s3 ls`, the object store crate does not support support
    /// incomplete file names as a prefix. Therefore, the prefix is the
    /// directory path up to the first glob character.
    ///
    /// Example: "path/to/file_*.txt" -> "path/to/"
    pub(super) fn list_prefix<T: AsRef<str>>(url_path: T) -> String {
        let url_path = url_path.as_ref();
        // Find first glob character index.
        let special_char_index = url_path
            .find(|c| ['*', '?', '['].contains(&c))
            .unwrap_or(url_path.len());

        match &url_path[..special_char_index].rsplit_once('/') {
            Some((dir_path, _)) => format!("{dir_path}/"),
            None => url_path[..special_char_index].to_string(),
        }
    }

    /// Process an object store stream and filter the results on the client
    /// based on the glob pattern.
    async fn process_object_store_stream(
        stream: impl futures::Stream<Item = object_store::Result<ObjectMeta>>,
        glob_pattern: &glob::Pattern,
        bucket: &str,
    ) -> VortexResult<Vec<(Url, ObjectMeta)>> {
        let matching_paths: Vec<(Url, ObjectMeta)> = stream
            .map(|object_meta| async move {
                if let Ok(object_meta) = object_meta {
                    let url_string = format!("s3://{}/{}", bucket, object_meta.location);
                    if glob_pattern.matches(&url_string) {
                        if let Ok(parsed_url) = Url::parse(&url_string) {
                            return Some((parsed_url, object_meta));
                        }
                    }
                }
                None
            })
            .buffer_unordered(16)
            .filter_map(|result| async { result })
            .collect()
            .await;

        Ok(matching_paths)
    }
}

mod local_filesystem {
    use super::*;

    /// Expand a glob pattern into a list of local disk URLs.
    /// Returns URLs without metadata for simplicity and performance.
    pub(super) fn expand_glob<T: AsRef<str>>(
        url_glob: T,
    ) -> VortexResult<(Vec<Url>, Option<Vec<ObjectMeta>>)> {
        let paths = glob::glob(url_glob.as_ref())
            .map_err(|e| vortex_err!("Failed to glob files: {}", e))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| vortex_err!("Failed to glob files: {}", e))?;

        let urls = paths
            .into_iter()
            .map(|p| {
                let path_clone = p
                    .canonicalize()
                    .map_err(|_| vortex_err!("Cannot canonicalize file path: {:?}", p))?;
                Url::from_file_path(&path_clone)
                    .map_err(|_| vortex_err!("Invalid file path: {:?}", path_clone))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((urls, None))
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs::{self, File};
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_expand_local_disk_glob_relative_path() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = "test.txt";

        let original_dir = env::current_dir().unwrap();
        env::set_current_dir(temp_dir.path()).unwrap();

        File::create(file_path).unwrap();
        let result = local_filesystem::expand_glob(file_path).unwrap();

        assert_eq!(result.0.len(), 1);
        assert_eq!(
            result.0[0].to_file_path().unwrap(),
            PathBuf::from(file_path).canonicalize().unwrap()
        );

        env::set_current_dir(&original_dir).unwrap();
    }

    #[test]
    fn test_expand_local_disk_glob_single_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        File::create(&file_path).unwrap();

        let glob_pattern = file_path.to_string_lossy().to_string();
        let result = local_filesystem::expand_glob(&glob_pattern).unwrap();

        assert_eq!(result.0.len(), 1);
        assert_eq!(
            result.0[0].to_file_path().unwrap(),
            file_path.canonicalize().unwrap()
        );
    }

    #[test]
    fn test_expand_local_disk_glob_wildcard() {
        let temp_dir = TempDir::new().unwrap();

        File::create(temp_dir.path().join("file1.txt")).unwrap();
        File::create(temp_dir.path().join("file2.txt")).unwrap();
        File::create(temp_dir.path().join("other.log")).unwrap();

        let glob_pattern = format!("{}/*.txt", temp_dir.path().display());
        let result = local_filesystem::expand_glob(&glob_pattern).unwrap();

        assert_eq!(result.0.len(), 2);

        let file_names: Vec<String> = result
            .0
            .iter()
            .map(|url| {
                url.to_file_path()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();

        assert!(file_names.contains(&"file1.txt".to_string()));
        assert!(file_names.contains(&"file2.txt".to_string()));
    }

    #[test]
    fn test_expand_local_disk_glob_no_matches() {
        let temp_dir = TempDir::new().unwrap();
        let glob_pattern = format!("{}/*.nonexistent", temp_dir.path().display());
        let result = local_filesystem::expand_glob(&glob_pattern).unwrap();
        assert_eq!(result.0.len(), 0);
    }

    #[test]
    fn test_expand_local_disk_glob_subdirectories() {
        let temp_dir = TempDir::new().unwrap();

        // Create nested directory structure
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();

        File::create(temp_dir.path().join("root.txt")).unwrap();
        File::create(subdir.join("nested.txt")).unwrap();

        let glob_pattern = format!("{}/**/*.txt", temp_dir.path().display());
        let result = local_filesystem::expand_glob(&glob_pattern).unwrap();

        assert_eq!(result.0.len(), 2);
    }

    #[test]
    fn test_extract_s3_url_path() {
        // Test valid S3 URL
        let url = Url::parse("s3://bucket/path/to/file.txt").unwrap();
        let result = s3::url_path(&url).unwrap();
        assert_eq!(result, "path/to/file.txt");

        // Test URL with nested path
        let url = Url::parse("s3://my-bucket/folder/subfolder/data.parquet").unwrap();
        let result = s3::url_path(&url).unwrap();
        assert_eq!(result, "folder/subfolder/data.parquet");

        // Test URL with root path
        let url = Url::parse("s3://bucket/file.txt").unwrap();
        let result = s3::url_path(&url).unwrap();
        assert_eq!(result, "file.txt");

        // Test URL without leading slash should fail
        let url = Url::parse("s3://bucket").unwrap();
        let result = s3::url_path(&url);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_list_prefix() {
        // Test with wildcard in filename
        let result = s3::list_prefix("folder/file*.txt");
        assert_eq!(result, "folder/");

        // Test with wildcard in directory
        let result = s3::list_prefix("folder/*/file.txt");
        assert_eq!(result, "folder/");

        // Test with nested directories and wildcard
        let result = s3::list_prefix("data/2023/*/logs/*.log");
        assert_eq!(result, "data/2023/");

        // Test with wildcard at root level
        let result = s3::list_prefix("*.txt");
        assert_eq!(result, "");

        // Test with no wildcards
        let result = s3::list_prefix("folder/subfolder/file.txt");
        assert_eq!(result, "folder/subfolder/");

        // Test with question mark wildcard
        let result = s3::list_prefix("folder/file?.txt");
        assert_eq!(result, "folder/");

        // Test with bracket wildcards
        let result = s3::list_prefix("folder/file[abc].txt");
        assert_eq!(result, "folder/");

        // Test empty path
        let result = s3::list_prefix("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_s3_url_parsing_integration() {
        // Test complete S3 URL parsing workflow
        let url = Url::parse("s3://my-bucket/data/year=2023/month=*/day=*/events.parquet").unwrap();

        let url_path = s3::url_path(&url).unwrap();
        assert_eq!(url_path, "data/year=2023/month=*/day=*/events.parquet");

        let list_prefix = s3::list_prefix(url_path);
        assert_eq!(list_prefix, "data/year=2023/");
    }

    #[test]
    fn test_s3_url_parsing_edge_cases() {
        // Test URL with multiple consecutive wildcards
        let url = Url::parse("s3://bucket/logs/**/*.log").unwrap();
        let url_path = s3::url_path(&url).unwrap();
        let list_prefix = s3::list_prefix(url_path);
        assert_eq!(list_prefix, "logs/");

        // Test URL with wildcard at the beginning
        let url = Url::parse("s3://bucket/*.txt").unwrap();
        let url_path = s3::url_path(&url).unwrap();
        let list_prefix = s3::list_prefix(url_path);
        assert_eq!(list_prefix, "");

        // Test deeply nested path with wildcard
        let url = Url::parse("s3://bucket/a/b/c/d/e/f/g/*.json").unwrap();
        let url_path = s3::url_path(&url).unwrap();
        let list_prefix = s3::list_prefix(url_path);
        assert_eq!(list_prefix, "a/b/c/d/e/f/g/");
    }

    #[test]
    fn test_s3_url_parsing_no_wildcards() {
        let url = Url::parse("s3://bucket/path/to/specific/file.txt").unwrap();
        let url_path = s3::url_path(&url).unwrap();
        let list_prefix = s3::list_prefix(url_path);
        assert_eq!(list_prefix, "path/to/specific/");
    }

    #[test]
    fn test_validate_glob_valid_pattern() {
        assert!(s3::validate_glob("s3://bucket/path/*.txt").is_ok());
    }

    #[test]
    fn test_validate_glob_escaped_asterisk() {
        let result = s3::validate_glob("s3://bucket/path\\*.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("\\*"));
    }

    #[test]
    fn test_validate_glob_escaped_question_mark() {
        let result = s3::validate_glob("s3://bucket/path\\?.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("\\?"));
    }

    #[test]
    fn test_validate_glob_escaped_bracket() {
        let result = s3::validate_glob("s3://bucket/path\\[test].txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("\\["));
    }
}
