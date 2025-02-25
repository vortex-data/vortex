//! FFI interface for Vortex File I/O.

use std::ffi::c_char;
use std::sync::Arc;

use futures::StreamExt;
use object_store::local::LocalFileSystem;
use vortex::error::VortexExpect;
use vortex::file::{GenericVortexFile, VortexFile, VortexOpenOptions};
use vortex::io::ObjectStoreReadAt;
use vortex::stream::ArrayStream;

use crate::dtype::FFIDType;
use crate::stream::{FFIArrayStream, FFIArrayStreamInner};
use crate::{RUNTIME, to_string};

#[repr(C)]
pub struct FFIFile {
    pub(crate) inner: VortexFile<GenericVortexFile<ObjectStoreReadAt>>,
}

// #[repr(C)]
// pub struct FFIFileScanOptions {
// }

/// Open a file at the given path on the file system.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIFile_open(path: *const c_char) -> *mut FFIFile {
    // TODO(aduffy): switch the ObjectStore based on scheme. Need to find a reasonable way to do this.
    let object_store = Arc::new(LocalFileSystem::new());
    let read_at = ObjectStoreReadAt::new(object_store, to_string(path).into(), None);

    let result = RUNTIME.block_on(async move { VortexOpenOptions::file(read_at).open().await });

    let file = result.vortex_expect("open");
    let ffi_file = FFIFile { inner: file };

    Box::into_raw(Box::new(ffi_file))
}

/// Build a new Scan that will stream batches of `FFIArray` from the file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIFile_scan(
    file: *const FFIFile,
    // options: Option<*const FFIFileScanOptions>,
) -> *mut FFIArrayStream {
    // We can say there are a specific set of of row indices to provide instead.

    let file = unsafe { &*file };
    let stream = file
        .inner
        .scan()
        .into_array_stream()
        .vortex_expect("into_array_stream");
    let dtype = Box::new(FFIDType::from(stream.dtype()));
    let inner = Some(Box::new(FFIArrayStreamInner {
        stream: stream.boxed(),
    }));

    Box::into_raw(Box::new(FFIArrayStream {
        dtype,
        inner,
        current: None,
    }))
}

/// Free the file and all associated resources.
///
/// This function will not automatically free any `FFIArrayStream`s that were built from this
/// file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIFile_free(file: *mut FFIFile) {
    drop(Box::from_raw(file));
}
