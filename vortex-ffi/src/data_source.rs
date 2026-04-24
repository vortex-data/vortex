// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(non_camel_case_types)]
#![deny(missing_docs)]

use std::ffi::c_char;
use std::ptr;
use std::sync::Arc;

use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::expr::stats::Precision::Exact;
use vortex::expr::stats::Precision::Inexact;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::scan::DataSource;
use vortex::scan::DataSourceRef;

use crate::RUNTIME;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::vx_error;
use crate::session::vx_session;
use crate::to_string;

crate::arc_dyn_wrapper!(
    /// A reference to one or more (possibly remote) paths.
    /// Creating vx_data_source opens the first matched path to read the schema.
    /// All other I/O is deferred until a scan is requested. Multiple scans may
    /// be requested from a single data source.
    dyn DataSource,
    vx_data_source);

/// Options for creating a data source.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_data_source_options {
    /// Required: paths to files, tables, or layout trees.
    /// May be a glob pattern like "*.vortex".
    /// If you want to include multiple paths, concat them with a comma:
    /// "file1.vortex,../file2.vortex".
    pub paths: *const c_char,
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

    let glob = unsafe { to_string(opts.paths) };
    let mut data_source = MultiFileDataSource::new(session.clone());
    for glob in glob.split(',') {
        data_source = data_source.with_glob(glob, None);
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
    Ok(vx_data_source::new(Arc::new(data_source) as DataSourceRef))
}

/// Create a data source.
/// The first matched file is opened eagerly. to read the schema. All other I/O
/// is deferred until a scan is requested. The returned pointer is owned by the
/// caller and must be freed with vx_data_source_free.
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

/// Return the schema of the data source as a non-owned dtype.
/// The returned pointer is valid as long as "ds" is alive. Do not free it.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_dtype(ds: *const vx_data_source) -> *const vx_dtype {
    vx_dtype::new_ref(vx_data_source::as_ref(ds).dtype())
}

#[repr(C)]
#[cfg_attr(test, derive(PartialEq, Debug))]
enum vx_cardinality {
    VX_CARD_UNKNOWN = 0,
    VX_CARD_ESTIMATE = 1,
    VX_CARD_MAXIMUM = 2,
}

#[repr(C)]
pub struct vx_data_source_row_count {
    cardinality: vx_cardinality,
    /// Set only when "cardinality" is not VX_CARD_UNKNOWN
    rows: u64,
}

/// Write data source's row count estimate into "row_count".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_get_row_count(
    ds: *const vx_data_source,
    row_count: *mut vx_data_source_row_count,
) {
    let rc = unsafe { &mut *row_count };
    match vx_data_source::as_ref(ds).row_count() {
        Some(Exact(rows)) => {
            rc.cardinality = vx_cardinality::VX_CARD_MAXIMUM;
            rc.rows = rows;
        }
        Some(Inexact(rows)) => {
            rc.cardinality = vx_cardinality::VX_CARD_ESTIMATE;
            rc.rows = rows;
        }
        None => {
            rc.cardinality = vx_cardinality::VX_CARD_UNKNOWN;
        }
    }
}

// Object store error: Generic LocalFileSystem error: Unable to convert
// URL "file:///C:%255CWindows%255CSystemTemp%255C.tmpRXzX38" to filesystem path
// https://github.com/servo/rust-url/issues/1077
#[cfg(not(windows))]
#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ptr;

    use crate::data_source::vx_cardinality;
    use crate::data_source::vx_data_source_dtype;
    use crate::data_source::vx_data_source_free;
    use crate::data_source::vx_data_source_get_row_count;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_options;
    use crate::data_source::vx_data_source_row_count;
    use crate::dtype::vx_dtype;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
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

            opts.paths = c"test.vortex".as_ptr();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            opts.paths = c"*.vortex".as_ptr();
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

            let path = CString::new(sample.path().to_str().unwrap()).unwrap();
            let opts = vx_data_source_options {
                paths: path.as_ptr(),
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());

            let dtype = vx_dtype::as_ref(vx_data_source_dtype(ds));
            assert_eq!(dtype, struct_array.dtype());

            let mut row_count = vx_data_source_row_count {
                cardinality: vx_cardinality::VX_CARD_UNKNOWN,
                rows: 0,
            };
            vx_data_source_get_row_count(ds, &raw mut row_count);
            assert_eq!(row_count.cardinality, vx_cardinality::VX_CARD_MAXIMUM);
            assert_eq!(row_count.rows, SAMPLE_ROWS as u64);

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }
}
