// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parsing of user-supplied URI-or-path strings into URLs, shared by the language bindings
//! that construct a [`MultiFileDataSource`](super::MultiFileDataSource).

#[cfg(any(
    unix,
    windows,
    target_os = "redox",
    target_os = "wasi",
    target_os = "hermit"
))]
use std::path::Component;
#[cfg(any(
    unix,
    windows,
    target_os = "redox",
    target_os = "wasi",
    target_os = "hermit"
))]
use std::path::Path;
#[cfg(any(
    unix,
    windows,
    target_os = "redox",
    target_os = "wasi",
    target_os = "hermit"
))]
use std::path::PathBuf;
#[cfg(any(
    unix,
    windows,
    target_os = "redox",
    target_os = "wasi",
    target_os = "hermit"
))]
use std::path::absolute;

use url::Url;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// Parse a URI-or-path string into a [`Url`]:
/// * full URLs (`s3://...`, `file:///...`) are used as-is,
/// * bare (relative or absolute) file paths are made absolute, have `.`/`..` components
///   normalized, and become `file://` URLs. On targets without a filesystem (e.g.
///   `wasm32-unknown-unknown`), bare paths are rejected instead.
///
/// Glob characters are preserved, so glob patterns like `/data/*.vortex` parse as expected.
pub fn parse_uri_or_path(uri_or_path: &str) -> VortexResult<Url> {
    // `Url::parse` accepts Windows absolute paths like `C:\foo` as a URL with a
    // single-letter scheme (`c`). No real URL scheme is one character, so treat any
    // single-letter scheme as a filesystem path instead.
    if let Ok(url) = Url::parse(uri_or_path)
        && url.scheme().len() > 1
    {
        return Ok(url);
    }
    file_path_to_url(uri_or_path)
}

/// Convert a bare filesystem path into a `file://` URL, absolutizing it and normalizing
/// `.`/`..` components. The cfg predicate matches `Url::from_file_path`'s availability in
/// the `url` crate.
#[cfg(any(
    unix,
    windows,
    target_os = "redox",
    target_os = "wasi",
    target_os = "hermit"
))]
fn file_path_to_url(path: &str) -> VortexResult<Url> {
    let abs =
        absolute(Path::new(path)).map_err(|e| vortex_err!("failed to absolutize {path}: {e}"))?;
    Url::from_file_path(normalize_path(abs))
        .map_err(|_| vortex_err!("neither URL nor path: {path}"))
}

/// `Url::from_file_path` does not exist on targets without a filesystem, such as
/// `wasm32-unknown-unknown`, so bare paths cannot be interpreted there.
#[cfg(not(any(
    unix,
    windows,
    target_os = "redox",
    target_os = "wasi",
    target_os = "hermit"
)))]
fn file_path_to_url(path: &str) -> VortexResult<Url> {
    Err(vortex_err!(
        "bare file paths are not supported on this platform: {path}"
    ))
}

/// Normalize `.` and `..` without touching the filesystem.
#[cfg(any(
    unix,
    windows,
    target_os = "redox",
    target_os = "wasi",
    target_os = "hermit"
))]
fn normalize_path(path: PathBuf) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_full_url() -> VortexResult<()> {
        let url = parse_uri_or_path("s3://bucket/prefix/*.vortex")?;
        assert_eq!(url.scheme(), "s3");
        assert_eq!(url.host_str(), Some("bucket"));
        assert_eq!(url.path(), "/prefix/*.vortex");
        Ok(())
    }

    #[test]
    fn test_file_scheme() -> VortexResult<()> {
        let url = parse_uri_or_path("file:///absolute/path/data.vortex")?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(url.path(), "/absolute/path/data.vortex");
        Ok(())
    }

    #[test]
    fn test_absolute_path() -> VortexResult<()> {
        // Use a drive-prefixed input on Windows so `absolute()` doesn't inject the cwd drive
        // and the expected URL path is predictable. The path does not need to exist.
        #[cfg(unix)]
        let (input, expected_path) = ("/tmp/data/*.vortex", "/tmp/data/*.vortex");
        #[cfg(windows)]
        let (input, expected_path) = (r"C:\tmp\data\*.vortex", "/C:/tmp/data/*.vortex");
        let url = parse_uri_or_path(input)?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(url.path(), expected_path);
        Ok(())
    }

    #[test]
    fn test_relative_path_resolves_against_cwd() -> VortexResult<()> {
        for input in ["data/table", "./data/table", "../data/table"] {
            let url = parse_uri_or_path(input)?;
            assert_eq!(url.scheme(), "file");
            assert!(url.path().starts_with('/'));
            assert!(url.path().ends_with("/data/table"));
            assert!(
                !url.path().contains("%2E"),
                "path must not contain percent-encoded dots"
            );
        }
        Ok(())
    }

    #[test]
    fn test_single_letter_scheme_is_path() -> VortexResult<()> {
        // Regression: `Url::parse("C:\\tmp")` succeeds with scheme="c"; the function must
        // treat that as a filesystem path, not a URL. Exercised on all platforms because
        // the check lives in `parse_uri_or_path`, not in an OS-specific branch.
        let url = parse_uri_or_path(r"C:\tmp\data\*.vortex")?;
        assert_eq!(url.scheme(), "file");
        assert_ne!(url.scheme(), "c");
        Ok(())
    }

    // Use absolute paths so the expected result is cwd-independent.
    #[cfg(unix)]
    #[rstest]
    #[case("/a/./b", "/a/b")]
    #[case("/a/b/./c", "/a/b/c")]
    #[case("/a/../b", "/b")]
    #[case("/a/b/../c", "/a/c")]
    #[case("/a/b/../../c", "/c")]
    #[case("/a/./b/.././c", "/a/c")]
    #[case("/a/b/../..", "/")]
    fn test_dot_normalization(
        #[case] input: &str,
        #[case] expected_path: &str,
    ) -> VortexResult<()> {
        let url = parse_uri_or_path(input)?;
        assert_eq!(url.scheme(), "file");
        assert_eq!(
            url.path(),
            expected_path,
            "input {input:?} should normalize to {expected_path:?}"
        );
        Ok(())
    }
}
