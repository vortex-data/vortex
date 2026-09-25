// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;
use std::ptr;
use std::slice;
use std::sync::Arc;
use std::sync::LazyLock;

use bytes::Bytes;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::buffer::ByteBuffer;
use vortex::cloud::Registry;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::expr::stats::Precision::Absent;
use vortex::expr::stats::Precision::Exact;
use vortex::expr::stats::Precision::Inexact;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::scan::DataSource;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::box_wrapper;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::vx_error;
use crate::read_at::read_at_from_ffi;
use crate::read_at::vx_readat;
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
    vx_data_source,
    drain_on_free
);

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

/// Parse `glob` as an object-store URL. A parse failure, a `file` scheme, or a
/// single-character scheme (a Windows drive path) is a local path.
fn object_store_url(glob: &str) -> Option<Url> {
    let url = Url::parse(glob).ok()?;
    (url.scheme().len() > 1 && url.scheme() != "file").then_some(url)
}

/// Resolve `url` through the shared [`vortex::cloud::Registry`] into the
/// store-relative path and a filesystem over the store.
fn registry_filesystem(
    url: &Url,
    session: &VortexSession,
) -> VortexResult<(String, FileSystemRef)> {
    static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

    let (store, path) = REGISTRY.resolve(url)?;
    let store = Arc::new(Compat::new(store));
    let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(store, session.handle()));
    Ok((path.to_string(), fs))
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
        let glob = unsafe { path.as_str() }?;
        // Object-store URLs get an explicit filesystem; anything else is a local glob.
        data_source = match object_store_url(glob) {
            Some(url) => {
                let (store_path, fs) = registry_filesystem(&url, session)?;
                data_source.with_glob(store_path, Some(fs))
            }
            None => data_source.with_glob(glob, None),
        };
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

/// Create a data source that reads through caller-supplied callbacks instead of
/// Vortex's own I/O.
///
/// Unlike vx_data_source_new_buffer, this keeps I/O pruning: only the segments a
/// scan needs are fetched, rather than the whole file up front.
///
/// "reader" is read during this call only; its callbacks and context must stay
/// valid until "release" runs. A rejected descriptor leaves ownership with the
/// caller and never calls "release"; once accepted, "release" runs before this
/// call returns if it fails, and otherwise before vx_data_source_free returns.
///
/// On error, returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_new_readat(
    session: *const vx_session,
    reader: *const vx_readat,
    err: *mut *mut vx_error,
) -> *const vx_data_source {
    let ds = try_or(err, ptr::null(), || {
        vortex_ensure!(!session.is_null());

        let session = vx_session::as_ref(session);
        let source = unsafe { read_at_from_ffi(reader) }?;

        let (file, len) = RUNTIME.block_on(async {
            let len = source.size().await?;
            let file = session.open_options().open(source).await?;
            VortexResult::Ok((file, len))
        })?;

        let ds = MultiLayoutDataSource::new_with_first(
            file.layout_reader()?,
            Vec::new(),
            vec![Some(len)],
            session,
        );

        Ok(vx_data_source::new(ds))
    });

    // A failure past descriptor validation leaves the reader owned by a spawned
    // task with no data source to free, so release it here instead.
    if ds.is_null() {
        RUNTIME.drain();
    }
    ds
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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex::array::array_session;
    use vortex::array::arrays::StructArray;
    use vortex::array::assert_arrays_eq;
    use vortex_array::VortexSessionExecute;

    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::data_source::vx_data_source_dtype;
    use crate::data_source::vx_data_source_free;
    use crate::data_source::vx_data_source_get_row_count;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_new_buffer;
    use crate::data_source::vx_data_source_new_readat;
    use crate::data_source::vx_data_source_options;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::read_at::vx_readat;
    use crate::scan::vx_data_source_scan;
    use crate::scan::vx_estimate;
    use crate::scan::vx_estimate_type;
    use crate::scan::vx_partition_free;
    use crate::scan::vx_partition_next;
    use crate::scan::vx_scan_free;
    use crate::scan::vx_scan_next_partition;
    use crate::session::vx_session;
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
    fn test_object_store_url_routing() {
        unsafe {
            let session = vx_session_new();
            let mut error = ptr::null_mut();

            // A URL scheme must route through the object-store registry, not
            // the local filesystem. The Azure builder deterministically
            // fails without configuration, and that failure can only arise
            // on the registry path.
            let url = vx_view::from_str("az://account/container/missing.vortex");
            let opts = vx_data_source_options {
                paths: &raw const url,
                paths_len: 1,
            };
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert!(ds.is_null());
            assert!(!error.is_null());
            let message = crate::error::vx_error_message(error).as_str().unwrap();
            // Local-path handling would report an unmatched glob pattern;
            // the registry path fails earlier, in the store builder.
            assert!(
                !message.contains("glob"),
                "URL was resolved as a local path: {message}"
            );
            assert_error(error);

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

    /// Backing store for the callback reader: the whole file in memory, plus a
    /// count of `release` calls so the test can prove it runs exactly once.
    struct ReadAtCtx {
        data: Vec<u8>,
        releases: AtomicUsize,
    }

    unsafe extern "C" fn read_at_cb(
        ctx: *mut c_void,
        offset: u64,
        dst: *mut u8,
        length: usize,
    ) -> i64 {
        let ctx = unsafe { &*ctx.cast::<ReadAtCtx>() };
        let Ok(start) = usize::try_from(offset) else {
            return -1;
        };
        let Some(end) = start.checked_add(length) else {
            return -1;
        };
        if end > ctx.data.len() {
            return -1;
        }
        unsafe { ptr::copy_nonoverlapping(ctx.data[start..end].as_ptr(), dst, length) };
        length as i64
    }

    unsafe extern "C" fn release_cb(ctx: *mut c_void) {
        let ctx = unsafe { &*ctx.cast::<ReadAtCtx>() };
        ctx.releases.fetch_add(1, Ordering::SeqCst);
    }

    /// A reader whose callback always fails, to check the error surfaces
    /// instead of handing uninitialized bytes to the scan.
    unsafe extern "C" fn failing_read_at_cb(
        _ctx: *mut c_void,
        _offset: u64,
        _dst: *mut u8,
        _length: usize,
    ) -> i64 {
        -7
    }

    /// Reports success while leaving the buffer untouched - what a naive
    /// `pread(2)` wrapper does at EOF or on EINTR.
    unsafe extern "C" fn short_read_at_cb(
        _ctx: *mut c_void,
        _offset: u64,
        _dst: *mut u8,
        length: usize,
    ) -> i64 {
        (length / 2) as i64
    }

    /// A written sample file, the callback state that serves it, and the array
    /// it should read back as.
    struct Sample {
        ctx: Box<ReadAtCtx>,
        len: u64,
        array: StructArray,
        _file: tempfile::NamedTempFile,
    }

    impl Sample {
        fn new(session: *const vx_session) -> Self {
            let (file, array) = unsafe { write_sample(session) };
            let data = read(file.path()).unwrap();
            let len = data.len() as u64;
            Self {
                ctx: Box::new(ReadAtCtx {
                    data,
                    releases: AtomicUsize::new(0),
                }),
                len,
                array,
                _file: file,
            }
        }

        /// An anonymous descriptor over this sample. Tests needing a name or a
        /// missing callback adjust the returned struct.
        fn reader(
            &self,
            read_at: unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> i64,
        ) -> vx_readat {
            vx_readat {
                ctx: (&raw const *self.ctx).cast::<c_void>().cast_mut(),
                len: self.len,
                concurrency: 0,
                name: vx_view::from_str(""),
                read_at: Some(read_at),
                release: Some(release_cb),
            }
        }

        fn releases(&self) -> usize {
            self.ctx.releases.load(Ordering::SeqCst)
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_readat() {
        unsafe {
            let session = vx_session_new();
            let sample = Sample::new(session);
            let mut reader = sample.reader(read_at_cb);
            reader.name = vx_view::from_str("test://sample.vortex");

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new_readat(session, &raw const reader, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());

            let ffi_dtype = vx_data_source_dtype(ds);
            let mut row_count = vx_estimate::default();
            vx_data_source_get_row_count(ds, &raw mut row_count);
            assert_eq!(row_count.r#type, vx_estimate_type::VX_ESTIMATE_EXACT);
            assert_eq!(row_count.estimate, SAMPLE_ROWS as u64);

            assert_eq!(sample.releases(), 0);

            vx_dtype_free(ffi_dtype);
            vx_data_source_free(ds);
            vx_session_free(session);

            assert_eq!(sample.releases(), 1);
        }
    }

    /// Descriptor validation fails before ownership transfers, so the caller
    /// keeps the context and `release` must not run.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_readat_invalid() {
        unsafe {
            let session = vx_session_new();
            let sample = Sample::new(session);
            let mut error = ptr::null_mut();

            let ds = vx_data_source_new_readat(ptr::null(), ptr::null(), &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new_readat(session, ptr::null(), &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            // read_at is required.
            let mut error = ptr::null_mut();
            let mut reader = sample.reader(read_at_cb);
            reader.read_at = None;
            let ds = vx_data_source_new_readat(session, &raw const reader, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            assert_eq!(sample.releases(), 0);
            vx_session_free(session);
        }
    }

    /// A failing callback must surface as an error. Ownership has already
    /// transferred by then, so `release` still runs.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_readat_callback_failure() {
        unsafe {
            let session = vx_session_new();
            let sample = Sample::new(session);
            let reader = sample.reader(failing_read_at_cb);

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new_readat(session, &raw const reader, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());
            assert_eq!(sample.releases(), 1);

            vx_session_free(session);
        }
    }

    /// A short read must fail rather than reach `set_len`, which would expose
    /// the unwritten tail of the buffer to the scan.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_readat_short_read() {
        unsafe {
            let session = vx_session_new();
            let sample = Sample::new(session);
            let reader = sample.reader(short_read_at_cb);

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new_readat(session, &raw const reader, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());

            // The descriptor was accepted before the read failed, so Vortex owns
            // the context and must have released it before returning.
            assert_eq!(sample.releases(), 1);

            vx_session_free(session);
        }
    }

    /// A name view of NULL with a non-zero length is a caller bug, not an
    /// anonymous source.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_readat_invalid_name() {
        unsafe {
            let session = vx_session_new();
            let sample = Sample::new(session);
            let mut reader = sample.reader(read_at_cb);
            reader.name = vx_view {
                ptr: ptr::null(),
                len: 5,
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new_readat(session, &raw const reader, &raw mut error);
            assert_error(error);
            assert!(ds.is_null());
            assert_eq!(sample.releases(), 0);

            vx_session_free(session);
        }
    }

    /// Scan data through the callbacks, not just the footer: this is the path
    /// that exercises data segments, coalescing and concurrent reads.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scan_readat() {
        let mut ctx_exec = array_session().create_execution_ctx();
        unsafe {
            let session = vx_session_new();
            let sample = Sample::new(session);
            let reader = sample.reader(read_at_cb);

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new_readat(session, &raw const reader, &raw mut error);
            assert_no_error(error);

            let scan = vx_data_source_scan(ds, ptr::null(), ptr::null_mut(), &raw mut error);
            assert_no_error(error);
            let partition = vx_scan_next_partition(scan, &raw mut error);
            assert_no_error(error);
            let array = vx_partition_next(partition, &raw mut error);
            assert_no_error(error);
            assert!(!array.is_null());

            assert_arrays_eq!(vx_array::as_ref(array), sample.array, &mut ctx_exec);

            vx_array_free(array);
            vx_partition_free(partition);
            vx_scan_free(scan);
            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }
}
