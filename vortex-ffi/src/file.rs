//! FFI interface for Vortex File I/O.

use std::ffi::{c_char, c_int};
use std::sync::Arc;

use object_store::local::LocalFileSystem;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::expr::{Identity, select};
use vortex::file::{GenericVortexFile, VortexFile, VortexOpenOptions};
use vortex::io::ObjectStoreReadAt;

use crate::stream::{FFIArrayStream, FFIArrayStreamInner};
use crate::{RUNTIME, to_string};

#[repr(C)]
pub struct FFIFile {
    pub(crate) inner: VortexFile<GenericVortexFile<ObjectStoreReadAt>>,
}

/// Scan options provided by an FFI client calling the `File_scan` function.
#[repr(C)]
pub struct FileScanOptions {
    /// Column names to project out in the scan. These must be null-terminated C strings.
    pub projection: *const *const c_char,
    /// Number of columns in `projection`.
    pub projection_len: c_int,
    // TODO(aduffy): add predicate pushdown here somehow.
}

/// Open a file at the given path on the file system.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_open(path: *const c_char) -> *mut FFIFile {
    // TODO(aduffy): switch the ObjectStore based on scheme. Need to find a reasonable way to do this.
    let object_store = Arc::new(LocalFileSystem::new());
    let read_at = ObjectStoreReadAt::new(object_store, to_string(path).into(), None);

    let result = RUNTIME.block_on(async move { VortexOpenOptions::file(read_at).open().await });

    let file = result.vortex_expect("open");
    let ffi_file = FFIFile { inner: file };

    Box::into_raw(Box::new(ffi_file))
}

/// Get a readonly pointer to the DType of the data inside of the file.
///
/// The pointer's lifetime is tied to the lifetime of the underlying file, so it should not be
/// dereferenced after the file has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_dtype(file: *const FFIFile) -> *const DType {
    assert!(!file.is_null(), "File_dtype: file is null");

    let file = &*file;
    file.inner.dtype()
}

/// Build a new Scan that will stream batches of `FFIArray` from the file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_scan(
    file: *const FFIFile,
    opts: *const FileScanOptions,
) -> *mut FFIArrayStream {
    let file = unsafe { &*file };

    let mut stream = file.inner.scan();

    if !opts.is_null() {
        let opts = &*opts;
        let mut field_names = Vec::new();
        for i in 0..opts.projection_len {
            let col_name = unsafe { *opts.projection.offset(i as isize) };
            let col_name: Arc<str> = to_string(col_name).into();
            field_names.push(col_name);
        }
        stream = stream.with_projection(select(field_names, Identity::new_expr()));
    }

    let stream = stream
        .into_array_stream()
        .vortex_expect("into_array_stream");

    let inner = Some(Box::new(FFIArrayStreamInner {
        stream: Box::pin(stream),
    }));

    Box::into_raw(Box::new(FFIArrayStream {
        inner,
        current: None,
    }))
}

/// Free the file and all associated resources.
///
/// This function will not automatically free any `FFIArrayStream`s that were built from this
/// file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_free(file: *mut FFIFile) {
    drop(Box::from_raw(file));
}
