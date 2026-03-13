#![allow(non_camel_case_types)]

use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::sync::Arc;

use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::stats::Precision::Exact;
use vortex::expr::stats::Precision::Inexact;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::scan::api::DataSource;
use vortex::scan::api::DataSourceRef;

use crate::error::vx_error;
use crate::RUNTIME;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::session::vx_session;
use crate::to_string;

crate::arc_dyn_wrapper!(
    /// A data source is a reference to multiple possibly remote files. When
    /// created, it opens first file to determine the schema from DType, all
    /// other operations are deferred till a scan is requested. You can request
    /// multiple file scans from a data source
    dyn DataSource,
    vx_data_source);

pub struct VxFileHandle;
pub type vx_file_handle = *const VxFileHandle;

pub type vx_list_callback =
    Option<unsafe extern "C" fn(userdata: *mut c_void, name: *const c_char, is_dir: c_int)>;
pub type vx_glob_callback =
    Option<unsafe extern "C" fn(userdata: *mut c_void, file: *const c_char)>;

pub type vx_fs_use_vortex =
    Option<unsafe extern "C" fn(schema: *const c_char, path: *const c_char) -> c_int>;
pub type vx_fs_set_userdata = Option<unsafe extern "C" fn(userdata: *mut c_void)>;

pub type vx_fs_open =
    Option<unsafe extern "C" fn(userdata: *mut c_void, path: *const c_char, err: *mut *mut vx_error)>;
pub type vx_fs_create =
    Option<unsafe extern "C" fn(userdata: *mut c_void, path: *const c_char, err: *mut *mut vx_error)>;

pub type vx_fs_list = Option<
    unsafe extern "C" fn(
        userdata: *const c_void,
        path: *const c_char,
        callback: vx_list_callback,
        error: *mut *mut vx_error,
    ),
>;

pub type vx_fs_close = Option<unsafe extern "C" fn(handle: vx_file_handle)>;
pub type vx_fs_size =
    Option<unsafe extern "C" fn(handle: vx_file_handle, err: *mut *mut vx_error) -> u64>;

pub type vx_fs_read = Option<
    unsafe extern "C" fn(
        handle: vx_file_handle,
        offset: u64,
        len: usize,
        buffer: *mut u8,
        err: *mut *mut vx_error,
    ) -> u64,
>;

pub type vx_fs_write = Option<
    unsafe extern "C" fn(
        handle: vx_file_handle,
        offset: u64,
        len: usize,
        buffer: *mut u8,
        err: *mut *mut vx_error,
    ) -> u64,
>;

pub type vx_fs_sync = Option<unsafe extern "C" fn(handle: vx_file_handle, err: *mut *mut vx_error)>;

pub type vx_glob = Option<
    unsafe extern "C" fn(glob: *const c_char, callback: vx_glob_callback, err: *mut *mut vx_error),
>;

pub type vx_cache = *mut c_void;
pub type vx_cache_key = *const c_char;

pub type vx_cache_init = Option<unsafe extern "C" fn(err: *mut *mut vx_error) -> vx_cache>;
pub type vx_cache_free = Option<unsafe extern "C" fn(cache: vx_cache, err: *mut *mut vx_error)>;
pub type vx_cache_get = Option<
    unsafe extern "C" fn(
        cache: vx_cache,
        key: vx_cache_key,
        value: *mut *mut c_void,
        err: *mut *mut vx_error,
    ),
>;
pub type vx_cache_put = Option<
    unsafe extern "C" fn(cache: vx_cache, key: vx_cache_key, value: *mut c_void, err: *mut *mut vx_error),
>;
pub type vx_cache_delete =
    Option<unsafe extern "C" fn(cache: vx_cache, key: vx_cache_key, err: *mut *mut vx_error)>;

#[repr(C)]
/// Host must either implement all or none of fs_* callbacks.
pub struct vx_data_source_options {
    // TODO what if the program wants to read a Vortex file from an existing buffer?
    files: *const c_char,

    /// Whether to use Vortex filesystem or host's filesystem.
    /// Return 1 to use Vortex for a given schema ("file", "s3") and path.
    /// Return 0 to use host's filesystem.
    fs_use_vortex: vx_fs_use_vortex,
    fs_set_userdata: vx_fs_set_userdata,
    fs_open: vx_fs_open,
    fs_create: vx_fs_create,
    fs_list: vx_fs_list,
    fs_close: vx_fs_close,
    fs_size: vx_fs_size,
    fs_read: vx_fs_read,
    fs_write: vx_fs_write,
    fs_sync: vx_fs_sync,

    glob: vx_glob,

    cache_init: vx_cache_init,
    cache_free: vx_cache_free,
    cache_get: vx_cache_get,
    cache_put: vx_cache_put,
    cache_delete: vx_cache_delete,
}

unsafe fn data_source_new(
    session: *const vx_session,
    opts: *const vx_data_source_options,
) -> VortexResult<*const vx_data_source> {
    if session.is_null() {
        vortex_bail!("empty session");
    }
    let session = vx_session::as_ref(session).clone();

    if opts.is_null() {
        vortex_bail!("empty opts");
    }
    let opts = unsafe { &*opts };

    if opts.files.is_null() {
        vortex_bail!("empty opts.files");
    }
    let glob = unsafe { to_string(opts.files) };

    RUNTIME.block_on(async {
        let data_source = MultiFileDataSource::new(session)
            //.with_filesystem(fs)
            .with_glob(glob)
            .build()
            .await?;
        Ok(vx_data_source::new(Arc::new(data_source) as DataSourceRef))
    })
}

/// Create a new owned datasource which must be freed by the caller
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_new(
    session: *const vx_session,
    opts: *const vx_data_source_options,
    err: *mut *mut vx_error,
) -> *const vx_data_source {
    try_or_default(err, || unsafe { data_source_new(session, opts) })
}

#[unsafe(no_mangle)]
/// Create a non-owned dtype referencing dataframe.
/// This dtype's lifetime is bound to underlying data source.
/// Caller should not free this dtype manually
pub unsafe extern "C-unwind" fn vx_data_source_dtype(ds: *const vx_data_source) -> *const vx_dtype {
    vx_dtype::new_ref(vx_data_source::as_ref(ds).dtype())
}

#[repr(C)]
enum vx_cardinality {
    VX_CARD_UNKNOWN = 0,
    VX_CARD_ESTIMATE = 1,
    VX_CARD_MAXIMUM = 2,
}

#[repr(C)]
pub struct vx_data_source_row_count {
    cardinality: vx_cardinality,
    rows: u64,
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_get_row_count(
    ds: *const vx_data_source,
    rc: *mut vx_data_source_row_count,
) {
    let rc = unsafe { &mut *rc };
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
