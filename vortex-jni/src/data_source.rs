// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for [`vortex::scan::DataSource`] (see the equivalent types in
//! `vortex-ffi/src/data_source.rs`).
//!
//! Glob handling mirrors `vortex-duckdb`'s `VortexMultiFileScan`:
//! * full URLs (`s3://...`, `file:///...`) are used as-is,
//! * bare file paths are made absolute and have `.`/`..` components normalized,
//! * filesystems are cached per base URL so repeated globs against the same bucket share
//!   a single client.

use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::path::absolute;
use std::sync::Arc;

use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jlong;
use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats::Precision;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::DataSourceRef;
use vortex::utils::aliases::hash_map::HashMap;

use crate::RUNTIME;
use crate::dtype::export_dtype_to_arrow;
use crate::errors::try_or_throw;
use crate::file::extract_properties;
use crate::object_store::object_store_fs;
use crate::session::session_ref;

/// Wraps an `Arc<dyn DataSource>` behind a single pointer.
pub(crate) struct NativeDataSource {
    inner: DataSourceRef,
}

impl NativeDataSource {
    fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    /// SAFETY: pointer must have been returned from [`Self::into_raw`].
    pub(crate) unsafe fn from_ptr<'a>(ptr: jlong) -> &'a Self {
        debug_assert!(ptr != 0, "null data source pointer");
        unsafe { &*(ptr as *const Self) }
    }

    pub(crate) fn inner(&self) -> &DataSourceRef {
        &self.inner
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDataSource_open(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    uris: JObjectArray,
    options: JObject,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let session = unsafe { session_ref(session_ptr) };
        let properties = extract_properties(env, &options)?;

        let mut glob_strings = Vec::new();
        let uri_count = uris.len(env)?;
        for idx in 0..uri_count {
            let uri = uris.get_element(env, idx)?;
            let uri = env.cast_local::<JString>(uri)?;
            let uri: String = uri.try_to_string(env)?;
            let uri = uri.trim();
            if !uri.is_empty() {
                glob_strings.push(uri.to_owned());
            }
        }
        if glob_strings.is_empty() {
            return Err(vortex_err!("no paths provided").into());
        }

        let glob_urls: Vec<Url> = glob_strings
            .iter()
            .map(|g| parse_glob_url(g.as_str()))
            .collect::<VortexResult<_>>()?;

        let mut fs_cache: HashMap<Url, FileSystemRef> = HashMap::new();
        for glob_url in &glob_urls {
            let base = base_url(glob_url);
            if !fs_cache.contains_key(&base) {
                let fs = object_store_fs(glob_url, &properties, session.handle())?;
                fs_cache.insert(base, fs);
            }
        }

        let mut builder = MultiFileDataSource::new(session.clone());
        for glob_url in &glob_urls {
            let base = base_url(glob_url);
            let fs = fs_cache
                .get(&base)
                .cloned()
                .unwrap_or_else(|| unreachable!("fs cached for every base url"));
            builder = builder.with_glob(glob_url.path(), Some(fs));
        }

        let inner = RUNTIME
            .block_on(builder.build())
            .map(|ds| Arc::new(ds) as DataSourceRef)?;
        Ok(Box::new(NativeDataSource { inner }).into_raw())
    })
}

/// Parse a glob string into a [`Url`]. Accepts full URLs and bare (relative or absolute)
/// file paths — see the module docs for details.
fn parse_glob_url(glob: &str) -> VortexResult<Url> {
    // `Url::parse` accepts Windows absolute paths like `C:\foo` as a URL with a
    // single-letter scheme (`c`). No real URL scheme is one character, so treat any
    // single-letter scheme as a filesystem path instead.
    if let Ok(url) = Url::parse(glob)
        && url.scheme().len() > 1
    {
        return Ok(url);
    }
    let path =
        absolute(Path::new(glob)).map_err(|e| vortex_err!("failed to absolutize {glob}: {e}"))?;
    let path = normalize_path(path);
    Url::from_file_path(path).map_err(|_| vortex_err!("neither URL nor path: {glob}"))
}

/// Normalize `.` and `..` without touching the filesystem.
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

/// URL with the path cleared, used as a cache key for filesystem reuse.
fn base_url(url: &Url) -> Url {
    let mut base = url.clone();
    base.set_path("");
    base
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDataSource_free(
    _env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) {
    if pointer == 0 {
        return;
    }
    drop(unsafe { Box::from_raw(pointer as *mut NativeDataSource) });
}

/// Export the data source's schema into the Arrow C Data Interface schema struct at
/// `schema_addr`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDataSource_arrowSchema(
    mut env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
    schema_addr: jlong,
) {
    try_or_throw(&mut env, |_| {
        if schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        let ds = unsafe { NativeDataSource::from_ptr(pointer) };
        export_dtype_to_arrow(ds.inner.dtype(), schema_addr)?;
        Ok(())
    });
}

/// Write the row count into the two-slot jlong pair `out`:
/// `out[0]` receives the row count (0 when unknown), `out[1]` the cardinality (0=unknown, 1=estimate, 2=exact).
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDataSource_rowCount(
    mut env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
    out: JLongArray,
) {
    try_or_throw(&mut env, |env| {
        let ds = unsafe { NativeDataSource::from_ptr(pointer) };
        let (rows, cardinality) = match ds.inner.row_count() {
            Some(Precision::Exact(r)) => (r as jlong, 2),
            Some(Precision::Inexact(r)) => (r as jlong, 1),
            None => (0, 0),
        };
        out.set_region(env, 0, &[rows, cardinality])?;
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_glob_url_full_url() {
        let url = parse_glob_url("s3://bucket/prefix/*.vortex").unwrap();
        assert_eq!(url.scheme(), "s3");
        assert_eq!(url.host_str(), Some("bucket"));
        assert_eq!(url.path(), "/prefix/*.vortex");
    }

    #[test]
    fn test_parse_glob_url_absolute_path() {
        // Use a drive-prefixed input on Windows so `absolute()` doesn't inject the cwd drive
        // and the expected URL path is predictable.
        #[cfg(unix)]
        let (input, expected_path) = ("/tmp/data/*.vortex", "/tmp/data/*.vortex");
        #[cfg(windows)]
        let (input, expected_path) = (r"C:\tmp\data\*.vortex", "/C:/tmp/data/*.vortex");
        let url = parse_glob_url(input).unwrap();
        assert_eq!(url.scheme(), "file");
        assert_eq!(url.path(), expected_path);
    }

    #[test]
    fn test_parse_glob_url_normalizes_dots() {
        #[cfg(unix)]
        let (input, expected_path) = ("/a/b/../c/./d", "/a/c/d");
        #[cfg(windows)]
        let (input, expected_path) = (r"C:\a\b\..\c\.\d", "/C:/a/c/d");
        let url = parse_glob_url(input).unwrap();
        assert_eq!(url.path(), expected_path);
    }

    #[test]
    fn test_parse_glob_url_single_letter_scheme_is_path() {
        // Regression: `Url::parse("C:\\tmp")` succeeds with scheme="c"; the function must
        // treat that as a filesystem path, not a URL. Exercised on all platforms because
        // the check lives in `parse_glob_url`, not in an OS-specific branch.
        let url = parse_glob_url(r"C:\tmp\data\*.vortex").unwrap();
        assert_eq!(url.scheme(), "file");
        assert_ne!(url.scheme(), "c");
    }

    #[test]
    fn test_base_url_strips_path() {
        let url = Url::parse("s3://bucket/a/b/c").unwrap();
        let base = base_url(&url);
        assert_eq!(base.scheme(), "s3");
        assert_eq!(base.host_str(), Some("bucket"));
        assert_eq!(base.path(), "");
    }
}
