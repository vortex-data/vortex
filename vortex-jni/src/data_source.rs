// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for [`vortex::scan::DataSource`] (see the equivalent types in
//! `vortex-ffi/src/data_source.rs`).
//!
//! Globs are parsed with [`parse_uri_or_path`], so full URLs (`s3://...`, `file:///...`)
//! and bare file paths are both accepted. Filesystems are cached per base URL so repeated
//! globs against the same bucket share a single client.

use std::sync::Arc;

use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jint;
use jni::sys::jlong;
use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats::Precision;
use vortex::file::multi::MultiFileDataSource;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::DataSourceRef;
use vortex::utils::aliases::hash_map::HashMap;

use crate::RUNTIME;
use crate::dtype::export_dtype_to_arrow;
use crate::errors::try_or_throw;
use crate::file::extract_properties;
use crate::io::JavaFileSystem;
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
            .map(|g| parse_uri_or_path(g.as_str()))
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

/// Open a data source over caller-provided `dev.vortex.io.NativeReadable` objects.
///
/// Unlike [`Java_dev_vortex_jni_NativeDataSource_open`], no storage client is created
/// on the native side: every read is an upcall into the corresponding Java object.
/// `paths` are opaque identifiers (typically the original file locations) used for
/// debugging and deduplication, `lengths` are the known file sizes in bytes.
/// `read_concurrency` caps in-flight `readFully` upcalls across *all* files of the
/// data source; values `<= 0` select the library default.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDataSource_openFiles(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    readables: JObjectArray,
    paths: JObjectArray,
    lengths: JLongArray,
    read_concurrency: jint,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let session = unsafe { session_ref(session_ptr) };

        let count = readables.len(env)?;
        if count == 0 {
            throw_runtime!("no readables provided");
        }
        if paths.len(env)? != count || lengths.len(env)? != count {
            throw_runtime!("readables, paths, and lengths must have equal length");
        }

        let mut sizes = vec![0 as jlong; count];
        lengths.get_region(env, 0, &mut sizes)?;

        let vm = env.get_java_vm()?;
        let concurrency = usize::try_from(read_concurrency).ok().filter(|c| *c > 0);
        let mut fs = JavaFileSystem::new(vm, session.handle(), concurrency);
        let mut ordered_paths = Vec::with_capacity(count);
        for idx in 0..count {
            let path_obj = paths.get_element(env, idx)?;
            let path: String = env.cast_local::<JString>(path_obj)?.try_to_string(env)?;
            if path.contains(['*', '?', '[']) {
                throw_runtime!("path '{path}' contains glob characters, which are unsupported");
            }
            let size = u64::try_from(sizes[idx])
                .map_err(|_| vortex_err!("negative length for path '{path}'"))?;

            let readable = readables.get_element(env, idx)?;
            if readable.is_null() {
                throw_runtime!("null readable for path '{path}'");
            }
            let readable = Arc::new(env.new_global_ref(&readable)?);

            // `MultiFileDataSource::with_glob` strips leading slashes (object-store paths
            // are bucket-relative), so key the registry by the same normalized form.
            let key = path.trim_start_matches('/').to_string();
            fs.insert(key.clone(), readable, size)?;
            ordered_paths.push(key);
        }

        let fs: FileSystemRef = Arc::new(fs);
        let mut builder = MultiFileDataSource::new(session.clone());
        for path in ordered_paths {
            builder = builder.with_glob(path, Some(Arc::clone(&fs)));
        }

        let inner = RUNTIME
            .block_on(builder.build())
            .map(|ds| Arc::new(ds) as DataSourceRef)?;
        Ok(Box::new(NativeDataSource { inner }).into_raw())
    })
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
            Precision::Exact(r) => (r as jlong, 2),
            Precision::Inexact(r) => (r as jlong, 1),
            Precision::Absent => (0, 0),
        };
        out.set_region(env, 0, &[rows, cardinality])?;
        Ok(())
    });
}

/// Write the byte size into the two-slot jlong pair `out`:
/// `out[0]` receives the size in bytes (0 when unknown), `out[1]` the precision (0=unknown, 1=estimate, 2=exact).
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDataSource_byteSize(
    mut env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
    out: JLongArray,
) {
    try_or_throw(&mut env, |env| {
        let ds = unsafe { NativeDataSource::from_ptr(pointer) };
        let (bytes, precision) = match ds.inner.byte_size() {
            Precision::Exact(b) => (b as jlong, 2),
            Precision::Inexact(b) => (b as jlong, 1),
            Precision::Absent => (0, 0),
        };
        out.set_region(env, 0, &[bytes, precision])?;
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_url_strips_path() {
        let url = Url::parse("s3://bucket/a/b/c").unwrap();
        let base = base_url(&url);
        assert_eq!(base.scheme(), "s3");
        assert_eq!(base.host_str(), Some("bucket"));
        assert_eq!(base.path(), "");
    }
}
