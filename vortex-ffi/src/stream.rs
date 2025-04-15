use std::pin::Pin;
use std::ptr;

use futures::StreamExt;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, vortex_bail};
use vortex::stream::ArrayStream;

use crate::array::{FFIArray, FFIArray_free};
use crate::error::{FFIError, into_c_error};

/// FFI-exposed stream interface.
pub struct FFIArrayStream {
    pub inner: Option<Box<FFIArrayStreamInner>>,
    pub current: Option<Box<FFIArray>>,
}

/// FFI-compatible interface for dealing with a stream array.
pub struct FFIArrayStreamInner {
    pub(crate) stream: Pin<Box<dyn ArrayStream>>,
}

/// Gets the dtype from an array `stream`, if the stream is finished the `DType` is null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_dtype(stream: *const FFIArrayStream) -> *const DType {
    let Some(inner) = unsafe { stream.as_ref() }
        .vortex_expect("null stream")
        .inner
        .as_ref()
    else {
        return ptr::null();
    };

    inner.stream.dtype()
}

/// Attempt to advance the `current` pointer of the stream.
///
/// A return value of `true` indicates that another element was pulled from the stream, and a return
/// of `false` indicates that the stream is finished.
///
/// It is an error to call this function again after the stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_next(
    stream: *mut FFIArrayStream,
    error: *mut *mut FFIError,
) -> bool {
    let result = (|| {
        let stream = unsafe { stream.as_mut() }.vortex_expect("stream null");
        let Some(inner) = stream.inner.as_mut() else {
            vortex_bail!("FFIArrayStream_next called after finish")
        };

        let element = futures::executor::block_on(inner.stream.next());

        if let Some(element) = element {
            let inner = element?;
            let ffi_array = FFIArray { inner };
            stream.current = Some(Box::new(ffi_array));

            Ok(true)
        } else {
            // Drop the element and stream pointers.
            stream.current.take();
            stream.inner.take();

            Ok(false)
        }
    })();

    unsafe { into_c_error(result, false, error) }
}

/// Predicate function to check if the array stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_finished(stream: *const FFIArrayStream) -> bool {
    unsafe { stream.as_ref().vortex_expect("null stream") }
        .inner
        .is_none()
}

/// Get the current array batch from the stream. Returns a unique pointer.
///
/// If this is called on an already finished stream the return value will be null.
///
/// # Safety
///
/// This function is unsafe because it dereferences the `stream` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_current(stream: *mut FFIArrayStream) -> *mut FFIArray {
    let stream = unsafe { stream.as_mut().vortex_expect("null stream") };

    if let Some(current) = stream.current.take() {
        Box::into_raw(current)
    } else {
        ptr::null_mut()
    }
}

/// Free the array stream and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_free(stream: *mut FFIArrayStream) {
    assert!(!stream.is_null(), "stream null");
    let mut stream = Box::from_raw(stream);

    if let Some(current) = stream.current.take() {
        FFIArray_free(Box::into_raw(current));
    }

    drop(stream.inner.take())
}
