use std::pin::Pin;
use std::ptr;

use futures::StreamExt;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, vortex_bail};
use vortex::stream::ArrayStream;

use crate::array::{VXArray, vx_array_free};
use crate::error::{VXError, try_or};

/// FFI-exposed stream interface.
pub struct VXArrayStream {
    pub inner: Option<Box<VXArrayStreamInner>>,
    pub current: Option<Box<VXArray>>,
}

/// FFI-compatible interface for dealing with a stream array.
pub struct VXArrayStreamInner {
    pub(crate) stream: Pin<Box<dyn ArrayStream>>,
}

/// Gets the dtype from an array `stream`, if the stream is finished the `DType` is null
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_dtype(
    stream: *const VXArrayStream,
) -> *const DType {
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
pub unsafe extern "C-unwind" fn vx_array_stream_next(
    stream: *mut VXArrayStream,
    error: *mut *mut VXError,
) -> bool {
    try_or(error, false, || {
        let stream = unsafe { stream.as_mut() }.vortex_expect("stream null");
        let Some(inner) = stream.inner.as_mut() else {
            vortex_bail!("vx_array_stream_next called after finish")
        };

        let element = futures::executor::block_on(inner.stream.next());

        if let Some(element) = element {
            let inner = element?;
            stream.current = Some(Box::new(VXArray { inner }));

            Ok(true)
        } else {
            // Drop the element and stream pointers.
            stream.current.take();
            stream.inner.take();

            Ok(false)
        }
    })
}

/// Predicate function to check if the array stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_finished(stream: *const VXArrayStream) -> bool {
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
pub unsafe extern "C-unwind" fn vx_array_stream_current(
    stream: *mut VXArrayStream,
) -> *mut VXArray {
    let stream = unsafe { stream.as_mut().vortex_expect("null stream") };

    if let Some(current) = stream.current.take() {
        Box::into_raw(current)
    } else {
        ptr::null_mut()
    }
}

/// Free the array stream and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_free(stream: *mut VXArrayStream) {
    assert!(!stream.is_null(), "stream null");
    let mut stream = Box::from_raw(stream);

    if let Some(current) = stream.current.take() {
        vx_array_free(Box::into_raw(current));
    }

    drop(stream.inner.take())
}
