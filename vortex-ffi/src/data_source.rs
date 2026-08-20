// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;
use std::ptr;
use std::slice;

use bytes::Bytes;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::expr::stats::Precision::Absent;
use vortex::expr::stats::Precision::Exact;
use vortex::expr::stats::Precision::Inexact;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::scan::DataSource;

use crate::RUNTIME;
use crate::box_wrapper;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::vx_error;
use crate::scan::vx_estimate;
use crate::scan::vx_estimate_type;
use crate::session::vx_session;
use crate::string::vx_view;

// MultiLayoutDataSource's fields are Arc'd inside
box_wrapper!(
    /// A reference to one or more possibly remote paths.
    ///
    /// Creating vx_data_source opens the first matched path to read the schema.
    /// All other I/O is deferred until a scan is requested. Multiple vx_scan's
    /// may be requested from a single vx_data_source.
    ///
    /// Copying a vx_data_source via vx_data_source_clone is a cheap operation.
    MultiLayoutDataSource,
    vx_data_source);

/// Options for creating a data source.
#[repr(C)]
pub struct vx_data_source_options {
    /// Required: paths to files, tables, or layout trees. Each entry may be a
    /// glob pattern like "*.vortex". Must point to an array of size
    /// "paths_len". "paths" bytes are copied.
    pub paths: *const vx_view,
    /// Number of entries in "paths".
    pub paths_len: usize,
}

#[cfg(test)]
impl Default for vx_data_source_options {
    fn default() -> Self {
        vx_data_source_options {
            paths: ptr::null(),
            paths_len: 0,
        }
    }
}

#[cfg(vortex_asan)]
unsafe extern "C" {
    pub fn __lsan_disable();
    pub fn __lsan_enable();
}

unsafe fn data_source_new(
    session: *const vx_session,
    opts: *const vx_data_source_options,
) -> VortexResult<*const vx_data_source> {
    vortex_ensure!(!session.is_null());
    vortex_ensure!(!opts.is_null());

    let session = vx_session::as_ref(session);

    let opts = unsafe { &*opts };
    vortex_ensure!(!opts.paths.is_null());
    vortex_ensure!(opts.paths_len > 0, "empty paths");

    let paths = unsafe { slice::from_raw_parts(opts.paths, opts.paths_len) };
    let mut data_source = MultiFileDataSource::new(session.clone());
    for path in paths {
        data_source = data_source.with_glob(unsafe { path.as_str() }?, None);
    }

    let data_source = RUNTIME.block_on(async {
        // TODO(myrrc): see https://github.com/vortex-data/vortex/issues/7324
        #[cfg(vortex_asan)]
        unsafe {
            __lsan_disable();
        }
        let data_source = data_source.build().await;
        #[cfg(vortex_asan)]
        unsafe {
            __lsan_enable();
        }
        data_source
    })?;
    Ok(vx_data_source::new(data_source))
}

/// Create a data source.
/// The first matched file is opened eagerly. to read the schema. All other I/O
/// is deferred until a scan is requested.
///
/// On error, returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_new(
    session: *const vx_session,
    options: *const vx_data_source_options,
    err: *mut *mut vx_error,
) -> *const vx_data_source {
    try_or(err, ptr::null(), || unsafe {
        data_source_new(session, options)
    })
}

/// Create a data source from a single in-memory Vortex file.
///
/// "buffer_len" is the length of "buffer" in bytes.
/// The bytes are borrowed, not copied: the caller must keep "buffer" alive and
/// unmodified until the data source is freed.
///
/// On error, returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_new_buffer(
    session: *const vx_session,
    buffer: *const c_void,
    buffer_len: usize,
    err: *mut *mut vx_error,
) -> *const vx_data_source {
    try_or(err, ptr::null(), || {
        vortex_ensure!(!session.is_null());
        vortex_ensure!(!buffer.is_null());

        let session = vx_session::as_ref(session);
        let bytes: &'static [u8] =
            unsafe { slice::from_raw_parts(buffer.cast::<u8>(), buffer_len) };
        let buffer = ByteBuffer::from(Bytes::from_static(bytes));
        let file = session.open_options().open_buffer(buffer)?;
        let ds = MultiLayoutDataSource::new_with_first(
            file.layout_reader()?,
            Vec::new(),
            vec![Some(buffer_len as u64)],
            session,
        );

        Ok(vx_data_source::new(ds))
    })
}

/// Increase reference count on vx_data_source
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_clone(
    ptr: *const vx_data_source,
) -> *const vx_data_source {
    vx_data_source::new(vx_data_source::as_ref(ptr).clone())
}

/// Return data source's dtype
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_dtype(ds: *const vx_data_source) -> *const vx_dtype {
    vx_dtype::new(vx_data_source::as_ref(ds).dtype().clone())
}

/// Write data source's row count estimate into "row_count".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_get_row_count(
    ds: *const vx_data_source,
    row_count: *mut vx_estimate,
) {
    let rc = unsafe { &mut *row_count };
    match vx_data_source::as_ref(ds).row_count() {
        Exact(rows) => {
            rc.r#type = vx_estimate_type::VX_ESTIMATE_EXACT;
            rc.estimate = rows;
        }
        Inexact(rows) => {
            rc.r#type = vx_estimate_type::VX_ESTIMATE_INEXACT;
            rc.estimate = rows;
        }
        Absent => {
            rc.r#type = vx_estimate_type::VX_ESTIMATE_UNKNOWN;
        }
    }
}

// Object store error: Generic LocalFileSystem error: Unable to convert
// URL "file:///C:%255CWindows%255CSystemTemp%255C.tmpRXzX38" to filesystem path
// https://github.com/servo/rust-url/issues/1077
#[cfg(not(windows))]
#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::fs::read;
    use std::ptr;

    use crate::data_source::vx_data_source_dtype;
    use crate::data_source::vx_data_source_free;
    use crate::data_source::vx_data_source_get_row_count;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_new_buffer;
    use crate::data_source::vx_data_source_options;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::scan::vx_estimate;
    use crate::scan::vx_estimate_type;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::string::vx_view;
    use crate::tests::SAMPLE_ROWS;
    use crate::tests::assert_error;
    use crate::tests::assert_no_error;
    use crate::tests::write_sample;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_invalid() {
        unsafe {
            let session = vx_session_new();
            let mut error = ptr::null_mut();

            let ds = vx_data_source_new(ptr::null_mut(), ptr::null(), &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let ds = vx_data_source_new(session, ptr::null(), &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let mut opts = vx_data_source_options::default();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let missing = vx_view::from_str("test.vortex");
            opts.paths = &raw const missing;
            opts.paths_len = 1;
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let missing_glob = vx_view::from_str("definitely-missing-dir/*.vortex");
            opts.paths = &raw const missing_glob;
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_row_count() {
        unsafe {
            let session = vx_session_new();
            let (sample, struct_array) = write_sample(session);

            let path = vx_view::from_str(sample.path().to_str().unwrap());
            let opts = vx_data_source_options {
                paths: &raw const path,
                paths_len: 1,
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());

            let ffi_dtype = vx_data_source_dtype(ds);
            let dtype = vx_dtype::as_ref(ffi_dtype);
            assert_eq!(dtype, struct_array.dtype());

            let mut row_count = vx_estimate::default();
            vx_data_source_get_row_count(ds, &raw mut row_count);
            assert_eq!(row_count.r#type, vx_estimate_type::VX_ESTIMATE_EXACT);
            assert_eq!(row_count.estimate, SAMPLE_ROWS as u64);

            vx_dtype_free(ffi_dtype);
            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_many_paths() {
        let dir = tempfile::tempdir().unwrap();

        unsafe {
            let session = vx_session_new();
            let (sample, _) = write_sample(session);

            let comma_path = dir.path().join("with,comma.vortex");
            std::fs::copy(sample.path(), &comma_path).unwrap();

            let paths = [
                vx_view::from_str(sample.path().to_str().unwrap()),
                vx_view::from_str(comma_path.to_str().unwrap()),
            ];
            let opts = vx_data_source_options {
                paths: paths.as_ptr(),
                paths_len: paths.len(),
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());

            let mut row_count = vx_estimate::default();
            vx_data_source_get_row_count(ds, &raw mut row_count);
            assert_eq!(row_count.estimate, 2 * SAMPLE_ROWS as u64);

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_buffer() {
        unsafe {
            let session = vx_session_new();
            let (sample, struct_array) = write_sample(session);

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new_buffer(session, ptr::null(), 0, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let file = read(sample).unwrap();
            let ds = vx_data_source_new_buffer(
                session,
                file.as_ptr() as *const c_void,
                file.len(),
                &raw mut error,
            );
            assert_no_error(error);
            assert!(!ds.is_null());

            let ffi_dtype = vx_data_source_dtype(ds);
            let dtype = vx_dtype::as_ref(ffi_dtype);
            assert_eq!(dtype, struct_array.dtype());

            let mut row_count = vx_estimate::default();
            vx_data_source_get_row_count(ds, &raw mut row_count);
            assert_eq!(row_count.r#type, vx_estimate_type::VX_ESTIMATE_EXACT);
            assert_eq!(row_count.estimate, SAMPLE_ROWS as u64);

            vx_dtype_free(ffi_dtype);
            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }
}
