// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]
#![expect(non_camel_case_types)]
#![forbid(clippy::todo)]
#![forbid(clippy::unimplemented)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod data_source;
mod dtype;
mod error;
mod expression;
mod log;
mod ptype;
mod scalar;
mod scan;
mod session;
mod sink;
mod string;
mod struct_array;
mod struct_fields;

use std::sync::Arc;
use std::sync::LazyLock;

pub use array::vx_array;
pub use array::vx_array_ref;
pub use dtype::vx_dtype;
pub use error::try_or;
pub use error::vx_error;
pub use error::vx_error_free;
pub use log::vx_log_level;
pub use scan::vx_partition;
pub use scan::vx_partition_into_array_stream;
pub use session::vx_session;
pub use session::vx_session_free;
pub use session::vx_session_new_with;
pub use session::vx_session_ref;
pub use sink::vx_array_sink;
pub use sink::vx_array_sink_open_file_with_strategy;
pub use string::vx_view;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::current::CurrentThreadWorkerPool;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A shared, caller-driven runtime for all FFI operations.
///
/// By default the runtime owns no threads: host threads drive its executor while they are inside
/// an FFI call. Concurrent calls from multiple host threads can therefore execute runtime tasks in
/// parallel without any Vortex-owned worker threads.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

/// Optional Vortex-owned background workers that drive the shared FFI runtime.
///
/// Creating the pool does not create threads. It starts empty so callers retain the default
/// host-thread-owned execution model until they opt in with [`vx_runtime_set_worker_threads`].
static POOL: LazyLock<CurrentThreadWorkerPool> = LazyLock::new(|| RUNTIME.new_pool());

/// Define a native FFI type that uses a [`Box`] wrapper.
#[macro_export]
macro_rules! box_wrapper {
    ($(#[$meta:meta])* $T:ty, $ffi_ident:ident) => {
        paste::paste! {
            $(#[$meta])*
            pub struct $ffi_ident($T);

            #[expect(dead_code)]
            impl $ffi_ident {
                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new_box(obj: Box<$T>) -> *mut $ffi_ident {
                    Box::into_raw(obj).cast::<$ffi_ident>()
                }

                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new(obj: $T) -> *mut $ffi_ident {
                    Box::into_raw(Box::new(obj)).cast::<$ffi_ident>()
                }

                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref<'a>(ptr: *const $ffi_ident) -> &'a $T {
                    use vortex::error::VortexExpect;
                    // TODO(joe): propagate this error up instead of expecting
                    &unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract a borrowed mutable reference from a mut pointer.
                pub(crate) fn as_mut<'a>(ptr: *mut $ffi_ident) -> &'a mut $T {
                    use vortex::error::VortexExpect;
                    // TODO(joe): propagate this error up instead of expecting
                    &mut unsafe { ptr.as_mut() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract an owned reference.
                pub(crate) fn into_box(ptr: *mut $ffi_ident) -> Box<$T>{
                    if ptr.is_null() {
                        vortex::error::vortex_panic!("null pointer");
                    }
                    unsafe { Box::from_raw(ptr.cast::<$T>()) }
                }
            }

            #[doc = r" Free a " $ffi_ident]
            // These allows only matter once the destructor is re-exported (e.g. `vx_error_free`):
            // its `# Safety` lives at the C boundary, and its doc links a private wrapper type.
            #[allow(clippy::missing_safety_doc)]
            #[allow(rustdoc::private_intra_doc_links)]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<$ffi_ident _free>](ptr: *const $ffi_ident) {
                if !ptr.is_null() {
                    std::mem::drop(unsafe { Box::from_raw(ptr.cast::<$T>().cast_mut()) })
                }
            }
        }
    };
}

/// Set the number of background worker threads driving the shared FFI runtime.
///
/// Calling this with a non-zero count opts the process into a Vortex-owned thread pool. These
/// background threads drive the same executor as host threads currently inside FFI calls. If this
/// function is never called, Vortex creates no runtime worker threads and execution remains
/// entirely host-thread-driven.
///
/// This setting is process-global and affects all FFI sessions. Passing zero restores the
/// host-thread-only configuration by signalling all background workers to stop. Increasing the
/// count starts workers immediately; decreasing it signals excess workers to stop.
#[unsafe(no_mangle)]
pub extern "C" fn vx_runtime_set_worker_threads(worker_threads: usize) {
    POOL.set_workers(worker_threads);
}

/// Return the configured number of Vortex-owned background worker threads.
///
/// Zero means the runtime is entirely driven by host threads entering FFI calls.
#[unsafe(no_mangle)]
pub extern "C" fn vx_runtime_worker_count() -> usize {
    POOL.worker_count()
}

/// Return the shared FFI runtime for layered FFI crates that drive Vortex streams produced through
/// `vortex-ffi`.
///
/// Streams from `vortex-ffi` partitions spawn their scan work onto this runtime's executor, so a
/// consumer crate (for example `vortex-cuda`'s Arrow device stream export) must drive them on this
/// same runtime rather than a private one.
pub fn ffi_runtime() -> &'static CurrentThreadRuntime {
    &RUNTIME
}

/// SAFETY: name must be a vx_view with non-NULL pointer
pub(crate) unsafe fn to_field_name(name: vx_view) -> VortexResult<FieldName> {
    let name: Arc<str> = Arc::from(unsafe { name.as_str() }?);
    Ok(name.into())
}

/// SAFETY: names must be a non-NULL pointer valid for reads up to len.
pub(crate) unsafe fn to_field_names(
    names: *const vx_view,
    len: usize,
) -> VortexResult<Vec<FieldName>> {
    vortex_ensure!(!names.is_null() || len == 0, "null names pointer");
    (0..len)
        .map(|i| unsafe { to_field_name(*names.add(i)) })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use rand::RngExt;
    use tempfile::NamedTempFile;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::validity::Validity;

    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::error::vx_error;
    use crate::error::vx_error_free;
    use crate::error::vx_error_message;
    use crate::session::vx_session;
    use crate::sink::vx_array_sink_close;
    use crate::sink::vx_array_sink_open_file;
    use crate::sink::vx_array_sink_push;
    use crate::string::vx_view;
    use crate::vx_runtime_set_worker_threads;
    use crate::vx_runtime_worker_count;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn runtime_worker_pool_configuration() {
        assert_eq!(vx_runtime_worker_count(), 0);
        vx_runtime_set_worker_threads(2);
        assert_eq!(vx_runtime_worker_count(), 2);
        vx_runtime_set_worker_threads(0);
        assert_eq!(vx_runtime_worker_count(), 0);
    }

    /// Panic if error is NULL. Free the error if it's not
    pub(crate) fn assert_error(error: *mut vx_error) {
        assert!(!error.is_null(), "Expected error");
        unsafe { vx_error_free(error) };
    }

    /// Panic if error is not NULL.
    pub(crate) fn assert_no_error(error: *mut vx_error) {
        if !error.is_null() {
            let message;
            unsafe {
                message = vx_error_message(error).as_str().unwrap().to_owned();
                vx_error_free(error);
            }
            panic!("{message}");
        }
    }

    fn random_str(length: usize) -> String {
        const CHARSET: &[u8] = b"0123456789";
        let mut rng = rand::rng();

        (0..length)
            .map(|_| {
                let idx = rng.random_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    pub const SAMPLE_ROWS: usize = 200;

    /// Write 200 rows of Struct { age=i32, height=i32, name=String } into a
    /// temporary file
    pub(crate) unsafe fn write_sample(session: *const vx_session) -> (NamedTempFile, StructArray) {
        let age = (0..SAMPLE_ROWS as u64).map(Some);
        let age = PrimitiveArray::from_option_iter(age);

        let height = (0..SAMPLE_ROWS as u64).map(|x| Some(200 * x));
        let height = PrimitiveArray::from_option_iter(height);

        let name = (0..SAMPLE_ROWS).map(random_str);
        let name = VarBinViewArray::from_iter_str(name);

        let struct_array = StructArray::try_new(
            ["age", "height", "name"].into(),
            vec![age.into_array(), height.into_array(), name.into_array()],
            SAMPLE_ROWS,
            Validity::NonNullable,
        )
        .unwrap();

        let file = NamedTempFile::new().unwrap();
        let path = vx_view::from_str(file.path().to_str().unwrap());
        let dtype = struct_array.dtype();

        unsafe {
            let vx_dtype_ptr = vx_dtype::new(dtype.clone());
            let mut error = ptr::null_mut();
            let sink = vx_array_sink_open_file(session, path, vx_dtype_ptr, &raw mut error);
            let array = vx_array::new(struct_array.clone().into_array());
            vx_array_sink_push(sink, array, &raw mut error);
            vx_array_sink_close(sink, &raw mut error);
            vx_array_free(array);
            vx_dtype_free(vx_dtype_ptr);
        }

        (file, struct_array)
    }
}
