// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod data_source;
mod dtype;
mod error;
mod expression;
mod log;
mod macros;
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

/// A shared runtime for all FFI operations.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

/// Background workers that drive the shared FFI runtime.
///
/// The pool starts empty so existing callers retain current-thread execution until they opt in
/// with [`vx_runtime_set_worker_threads`].
static POOL: LazyLock<CurrentThreadWorkerPool> = LazyLock::new(|| RUNTIME.new_pool());

/// Set the number of background worker threads driving the shared FFI runtime.
///
/// This setting is process-global. Passing zero disables background execution. Increasing the
/// count starts workers immediately; decreasing it signals excess workers to stop.
#[unsafe(no_mangle)]
pub extern "C" fn vx_runtime_set_worker_threads(worker_threads: usize) {
    POOL.set_workers(worker_threads);
}

/// Return the configured number of background worker threads driving the shared FFI runtime.
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
    use std::sync::Arc;

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
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype.clone()));
            let mut error = ptr::null_mut();
            let sink = vx_array_sink_open_file(session, path, vx_dtype_ptr, &raw mut error);
            let array = vx_array::new(Arc::new(struct_array.clone().into_array()));
            vx_array_sink_push(sink, array, &raw mut error);
            vx_array_sink_close(sink, &raw mut error);
            vx_array_free(array);
            vx_dtype_free(vx_dtype_ptr);
        }

        (file, struct_array)
    }
}
