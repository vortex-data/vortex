// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::c_char;

use futures::SinkExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteSummary;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;

use crate::RUNTIME;
use crate::array::vx_array;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::session::vx_session;

#[expect(non_camel_case_types)]
/// The `sink` interface is used to collect array chunks and place them into a resource
/// (e.g. an array stream or file (`vx_array_sink_open_file`)).
///
/// ## Thread Safety
///
/// This struct is **not** thread-safe for concurrent operations. While the underlying
/// `Sender` is thread-safe, the FFI wrapper should only be accessed from a single thread
/// to avoid race conditions between `push` and `close` operations. The `close` operation
/// consumes the sink, making any subsequent operations undefined behavior.
///
/// Multiple threads may safely hold pointers to the same sink, but only one thread should
/// perform operations on it at a time, and coordination is required to ensure `close` is
/// called exactly once after all `push` operations are complete.
pub struct vx_array_sink {
    sink: Sender<VortexResult<ArrayRef>>,
    writer: Task<VortexResult<WriteSummary>>,
}

/// Opens a writable array stream, where sink is used to push values into the stream.
/// To close the stream close the sink with `vx_array_sink_close`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_open_file(
    session: *const vx_session,
    path: *const c_char,
    dtype: *const vx_dtype,
    error_out: *mut *mut vx_error,
) -> *mut vx_array_sink {
    try_or_default(error_out, || {
        let session = vx_session::as_ref(session);

        if path.is_null() {
            vortex_bail!("null path");
        }
        let path = unsafe { CStr::from_ptr(path) }
            .to_string_lossy()
            .to_string();

        let file_dtype = vx_dtype::as_ref(dtype);
        // The channel size 32 was chosen arbitrarily.
        let (sink, rx) = mpsc::channel(32);
        let array_stream = ArrayStreamAdapter::new(file_dtype.clone(), rx.into_stream());

        let writer = session.handle().spawn(async move {
            let mut file = async_fs::File::create(path).await?;
            session.write_options().write(&mut file, array_stream).await
        });

        Ok(Box::into_raw(Box::new(vx_array_sink { sink, writer })))
    })
}

/// Push an array into a file sink.
/// Does not take ownership of array
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_push(
    sink: *mut vx_array_sink,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        vortex_ensure!(!array.is_null());
        vortex_ensure!(!sink.is_null());

        let array = vx_array::as_ref(array);
        let sink = unsafe { &mut *sink };
        RUNTIME
            .block_on(sink.sink.send(Ok(array.clone())))
            .map_err(|e| vortex_err!("Send error: {e}"))
    })
}

/// Closes an array sink, must be called to ensure all the values pushed to the sink are written
/// to the external resource.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_close(
    sink: *mut vx_array_sink,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        let vx_array_sink { sink, writer } = *unsafe { Box::from_raw(sink) };
        drop(sink);

        RUNTIME.block_on(async {
            let _footer = writer.await?;
            VortexResult::Ok(())
        })?;

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::sync::Arc;

    use tempfile::NamedTempFile;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;

    use super::*;
    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::error::vx_error_free;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sink_basic_workflow() {
        unsafe {
            let session = vx_session_new();

            let temp_file = NamedTempFile::new().unwrap();
            let path = CString::new(temp_file.path().to_str().unwrap()).unwrap();

            let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let sink =
                vx_array_sink_open_file(session, path.as_ptr(), vx_dtype_ptr, &raw mut error);
            assert!(error.is_null());
            assert!(!sink.is_null());

            // Create and push an array
            let array = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let vx_array_ptr = vx_array::new(Arc::new(array.into_array()));

            vx_array_sink_push(sink, vx_array_ptr, &raw mut error);
            assert!(error.is_null());

            // Close the sink
            vx_array_sink_close(sink, &raw mut error);
            assert!(error.is_null());

            // Cleanup
            vx_array_free(vx_array_ptr);
            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sink_multiple_arrays() {
        unsafe {
            let session = vx_session_new();

            let temp_file = NamedTempFile::new().unwrap();
            let path = CString::new(temp_file.path().to_str().unwrap()).unwrap();

            let dtype = DType::Primitive(vortex::dtype::PType::U64, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let sink =
                vx_array_sink_open_file(session, path.as_ptr(), vx_dtype_ptr, &raw mut error);
            assert!(error.is_null());

            // Push multiple arrays
            for i in 0..3 {
                let start = i * 3;
                let array = PrimitiveArray::new(
                    buffer![start as u64, (start + 1) as u64, (start + 2) as u64],
                    Validity::NonNullable,
                );
                let vx_array_ptr = vx_array::new(Arc::new(array.into_array()));

                vx_array_sink_push(sink, vx_array_ptr, &raw mut error);
                assert!(error.is_null());

                vx_array_free(vx_array_ptr);
            }

            vx_array_sink_close(sink, &raw mut error);
            assert!(error.is_null());

            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sink_invalid_path() {
        unsafe {
            let session = vx_session_new();

            // Use a path that will fail during file creation (read-only directory on most systems)
            let invalid_path = CString::new("/dev/null/invalid.vortex").unwrap();
            let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let sink = vx_array_sink_open_file(
                session,
                invalid_path.as_ptr(),
                vx_dtype_ptr,
                &raw mut error,
            );

            // The sink creation may succeed but close should fail due to invalid path
            if !sink.is_null() {
                // Push an array
                let array = PrimitiveArray::new(buffer![1i32], Validity::NonNullable);
                let vx_array_ptr = vx_array::new(Arc::new(array.into_array()));
                vx_array_sink_push(sink, vx_array_ptr, &raw mut error);
                vx_array_free(vx_array_ptr);

                // Close should fail due to invalid path
                vx_array_sink_close(sink, &raw mut error);
                // Either error is set or operation succeeds (depends on filesystem)
                if !error.is_null() {
                    vx_error_free(error);
                }
            } else {
                // Sink creation failed, which is also valid
                if !error.is_null() {
                    vx_error_free(error);
                }
            }

            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sink_null_path() {
        unsafe {
            let session = vx_session_new();

            let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            // This should return null and set error due to null path
            let sink =
                vx_array_sink_open_file(session, std::ptr::null(), vx_dtype_ptr, &raw mut error);

            assert!(sink.is_null());
            assert!(!error.is_null());

            vx_error_free(error);
            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }
}
