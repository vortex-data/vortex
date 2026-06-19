// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::absolute;
use std::sync::Arc;

use itertools::Itertools;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex_utils::aliases::hash_map::HashMap;

use crate::RUNTIME;
use crate::SESSION;
use crate::duckdb::BindInputRef;
use crate::duckdb::ExtractedValue;

/// Parse a glob string into a [`Url`].
///
/// Accepts full URLs (e.g. `s3://bucket/prefix/*.vortex`, `file:///data/*.vortex`) as well as
/// bare file paths. For bare paths, the path is made absolute (without requiring it to exist)
/// so that relative paths such as `./data/*.vortex` or `../data/*.vortex` are resolved correctly.
fn parse_glob_url(glob_url_str: &str) -> VortexResult<Url> {
    Url::parse(glob_url_str).or_else(|_| {
        let path = absolute(Path::new(glob_url_str))
            .map_err(|e| vortex_err!("Failed making {glob_url_str} absolute: {e}"))?;
        // `absolute()` does not normalize `..` components, so `/a/b/../c` stays as-is.
        // Normalizing manually avoids `..` being percent-encoded in the resulting URL.
        let path = normalize_path(path);
        Url::from_file_path(path).map_err(|_| vortex_err!("Neither URL nor path: {glob_url_str}"))
    })
}

/// Normalize a path by resolving `.` and `..` components without accessing the filesystem.
fn normalize_path(path: std::path::PathBuf) -> std::path::PathBuf {
    use std::path::Component;
    let mut normalized = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            c => normalized.push(c),
        }
    }
    normalized
}

fn resolve_filesystem(base_url: &Url) -> VortexResult<FileSystemRef> {
    let object_store: Arc<dyn ObjectStore> = match base_url.scheme() {
        "file" => Arc::new(LocalFileSystem::new()),
        "s3" => Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(base_url.host_str().ok_or_else(|| {
                    vortex_err!("Failed to extract bucket name from URL: {base_url}")
                })?)
                .build()?,
        ),
        other => vortex_bail!("Unsupported URL scheme '{other}'"),
    };

    Ok(Arc::new(ObjectStoreFileSystem::new(
        Arc::new(Compat::new(object_store)),
        RUNTIME.handle(),
    )))
}

/// Shared bind logic for both single-glob and multi-glob variants.
pub fn bind_multi_file_scan(input: &BindInputRef) -> VortexResult<MultiLayoutDataSource> {
    let glob_url_parameter = input
        .get_parameter(0)
        .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

    // The input to the table function can either be a single glob, or a List of glob patterns.
    let glob_strings: Vec<String> = match glob_url_parameter.extract() {
        ExtractedValue::Varchar(glob) => {
            vec![glob.to_string()]
        }
        ExtractedValue::List(globs) => globs
            .into_iter()
            .map(|glob| {
                let ExtractedValue::Varchar(string) = glob.extract() else {
                    vortex_bail!("list element must be Varchar type")
                };

                Ok(string.to_string())
            })
            .try_collect()?,
        _ => vortex_bail!("Invalid argument to read_vortex table function"),
    };

    // Parse each glob URL and resolve its filesystem.
    let mut glob_urls: Vec<Url> = Vec::with_capacity(glob_strings.len());
    for glob_str in &glob_strings {
        glob_urls.push(parse_glob_url(glob_str)?);
    }

    // Cache filesystems by base URL to avoid resolving the same filesystem multiple times.
    let mut fs_cache: HashMap<Url, FileSystemRef> = HashMap::new();
    for glob_url in &glob_urls {
        let mut base_url = glob_url.clone();
        base_url.set_path("");
        if !fs_cache.contains_key(&base_url) {
            let fs = resolve_filesystem(&base_url)?;
            fs_cache.insert(base_url, fs);
        }
    }

    RUNTIME.block_on(async {
        let mut builder = MultiFileDataSource::new(SESSION.clone());

        for glob_url in &glob_urls {
            let mut base_url = glob_url.clone();
            base_url.set_path("");
            let fs = fs_cache
                .get(&base_url)
                .map(Arc::clone)
                .unwrap_or_else(|| unreachable!("fs should be cached for all base URLs"));
            builder = builder.with_glob(glob_url.path(), Some(fs));
        }

        builder.build().await
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_parse_glob_url_s3() -> VortexResult<()> {
        let url = parse_glob_url("s3://my-bucket/prefix/*.vortex")?;
        assert_eq!(url.scheme(), "s3");
        assert_eq!(url.host_str(), Some("my-bucket"));
        assert_eq!(url.path(), "/prefix/*.vortex");
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_file_scheme() -> VortexResult<()> {
        let url = parse_glob_url("file:///absolute/path/data.vortex")?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(url.path(), "/absolute/path/data.vortex");
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_absolute_glob_path() -> VortexResult<()> {
        let tmpdir = tempfile::tempdir()?;
        let glob = format!("{}/*.vortex", tmpdir.path().display());
        let url = parse_glob_url(&glob)?;
        assert_eq!(url.scheme(), "file");
        assert!(url.path().ends_with("/*.vortex"));
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_absolute_existing_path() -> VortexResult<()> {
        let tmpfile = tempfile::NamedTempFile::new()?;
        let canonical = std::fs::canonicalize(tmpfile.path())?;
        let path_str = canonical
            .to_str()
            .ok_or_else(|| vortex_err!("canonical path is not valid UTF-8"))?;
        let url = parse_glob_url(path_str)?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(url.path(), path_str);
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_relative_path() -> VortexResult<()> {
        // Create a tempfile in the current working directory so we can refer to it
        // by a relative name (just the filename, without any directory component).
        let tmpfile = tempfile::NamedTempFile::new_in(".")?;
        let filename = tmpfile
            .path()
            .file_name()
            .ok_or_else(|| vortex_err!("temp file missing file name"))?
            .to_str()
            .ok_or_else(|| vortex_err!("temp file name is not valid UTF-8"))?;

        let url = parse_glob_url(filename)?;
        assert_eq!(url.scheme(), "file");
        // The relative name must have been resolved to an absolute path.
        assert!(url.path().ends_with(filename));
        assert!(url.path().starts_with('/'));
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_relative_glob_path() -> VortexResult<()> {
        // A relative path with a glob character (e.g. `./data/*.vortex`) must also resolve
        // correctly.
        let tmpdir = tempfile::tempdir_in(".")?;
        let dir_name = tmpdir
            .path()
            .file_name()
            .ok_or_else(|| vortex_err!("temp dir missing file name"))?
            .to_str()
            .ok_or_else(|| vortex_err!("temp dir name is not valid UTF-8"))?;
        let glob = format!("./{dir_name}/*.vortex");
        let url = parse_glob_url(&glob)?;
        assert_eq!(url.scheme(), "file");
        assert!(url.path().starts_with('/'));
        assert!(url.path().ends_with("/*.vortex"));
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_nonexistent_path() -> VortexResult<()> {
        // absolute() does not require the path to exist, so a non-existent path succeeds.
        let url = parse_glob_url("/nonexistent/path/file.vortex")?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(url.path(), "/nonexistent/path/file.vortex");
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_parent_relative_path() -> VortexResult<()> {
        // A path starting with `..` must be resolved to an absolute path without
        // percent-encoding the `..` component in the resulting URL.
        let tmpfile = tempfile::NamedTempFile::new_in("..")?;
        let filename = tmpfile
            .path()
            .file_name()
            .ok_or_else(|| vortex_err!("temp file missing file name"))?
            .to_str()
            .ok_or_else(|| vortex_err!("temp file name is not valid UTF-8"))?;
        let relative = format!("../{filename}");

        let url = parse_glob_url(&relative)?;
        assert_eq!(url.scheme(), "file");
        // The resolved path must be absolute and must not contain encoded dots.
        assert!(url.path().starts_with('/'));
        assert!(
            !url.path().contains("%2E"),
            "path must not contain percent-encoded dots"
        );
        assert!(url.path().ends_with(filename));
        Ok(())
    }

    // Use absolute paths so the expected result is cwd-independent.
    #[rstest]
    #[case("/a/./b", "/a/b")]
    #[case("/a/b/./c", "/a/b/c")]
    #[case("/a/../b", "/b")]
    #[case("/a/b/../c", "/a/c")]
    #[case("/a/b/../../c", "/c")]
    #[case("/a/./b/.././c", "/a/c")]
    #[case("/a/b/../..", "/")]
    fn test_parse_glob_url_dot_normalization(
        #[case] input: &str,
        #[case] expected_path: &str,
    ) -> VortexResult<()> {
        let url = parse_glob_url(input)?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(
            url.path(),
            expected_path,
            "input {input:?} should normalize to {expected_path:?}"
        );
        Ok(())
    }
}
