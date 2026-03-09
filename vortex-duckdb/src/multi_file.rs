// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::absolute;
use std::sync::Arc;

use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::scan::api::DataSourceRef;

use crate::RUNTIME;
use crate::SESSION;
use crate::datasource::DataSourceTableFunction;
use crate::duckdb::BindInputRef;
use crate::duckdb::ClientContextRef;
use crate::duckdb::LogicalType;
use crate::filesystem::resolve_filesystem;

/// Parse a glob string into a [`Url`].
///
/// Accepts full URLs (e.g. `s3://bucket/prefix/*.vortex`, `file:///data/*.vortex`) as well as
/// bare file paths. For bare paths, the path is made absolute (without requiring it to exist)
/// so that relative paths such as `./data/*.vortex` are resolved correctly.
fn parse_glob_url(glob_url_str: &str) -> VortexResult<Url> {
    Url::parse(glob_url_str).or_else(|_| {
        let path = absolute(Path::new(glob_url_str))
            .map_err(|e| vortex_err!("Failed making {glob_url_str} absolute: {e}"))?;
        Url::from_file_path(path)
            .map_err(|_| vortex_err!("Neither URL nor path: {glob_url_str}"))
    })
}

/// Vortex multi-file scan table function (`vortex_scan` / `read_vortex`).
///
/// Takes a file glob parameter and resolves it into a [`MultiFileDataSource`].
/// All other table function logic is provided by the blanket [`DataSourceTableFunction`]
/// implementation.
#[derive(Debug)]
pub struct VortexMultiFileScan;

impl DataSourceTableFunction for VortexMultiFileScan {
    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    fn bind(ctx: &ClientContextRef, input: &BindInputRef) -> VortexResult<DataSourceRef> {
        let glob_url_parameter = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        // Parse the URL and separate the base URL (keep scheme, host, etc.) from the path.
        let glob_url_string = glob_url_parameter.as_string();
        let glob_url_str = glob_url_string.as_str();
        let glob_url = parse_glob_url(glob_url_str)?;

        let mut base_url = glob_url.clone();
        base_url.set_path("");

        let fs = resolve_filesystem(&base_url, ctx)?;

        RUNTIME.block_on(async {
            let builder = MultiFileDataSource::new(SESSION.clone())
                .with_filesystem(fs)
                .with_glob(glob_url.path());
            let ds = builder.build().await?;
            VortexResult::Ok(Arc::new(ds) as DataSourceRef)
        })
    }
}

#[cfg(test)]
mod tests {
    #[allow(clippy::wildcard_imports)]
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
        let tmpdir = tempfile::tempdir().unwrap();
        let glob = format!("{}/*.vortex", tmpdir.path().display());
        let url = parse_glob_url(&glob)?;
        assert_eq!(url.scheme(), "file");
        assert!(url.path().ends_with("/*.vortex"));
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_absolute_existing_path() -> VortexResult<()> {
        let tmpfile = tempfile::NamedTempFile::new().unwrap();
        let canonical = std::fs::canonicalize(tmpfile.path()).unwrap();
        let path_str = canonical.to_str().unwrap();
        let url = parse_glob_url(path_str)?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(url.path(), path_str);
        Ok(())
    }

    #[test]
    fn test_parse_glob_url_relative_path() -> VortexResult<()> {
        // Create a tempfile in the current working directory so we can refer to it
        // by a relative name (just the filename, without any directory component).
        let tmpfile = tempfile::NamedTempFile::new_in(".").unwrap();
        let filename = tmpfile.path().file_name().unwrap().to_str().unwrap();

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
        let tmpdir = tempfile::tempdir_in(".").unwrap();
        let dir_name = tmpdir.path().file_name().unwrap().to_str().unwrap();
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
}
