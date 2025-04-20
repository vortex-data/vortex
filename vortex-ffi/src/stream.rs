use std::pin::Pin;
use std::ptr;
use std::ptr::null_mut;

use futures::StreamExt;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, vortex_bail};
use vortex::stream::ArrayStream;

use crate::array::vx_array;
use crate::error::{try_or, vx_error};

/// FFI-exposed stream interface.
#[allow(non_camel_case_types)]
pub struct vx_array_stream {
    pub inner: Option<Box<ArrayStreamInner>>,
}

/// FFI-compatible interface for dealing with a stream array.
pub struct ArrayStreamInner {
    pub(crate) stream: Pin<Box<dyn ArrayStream>>,
}

/// Gets the dtype from an array `stream`, if the stream is finished the `DType` is null
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_dtype(
    stream: *const vx_array_stream,
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
    stream: *mut vx_array_stream,
    error: *mut *mut vx_error,
) -> *mut vx_array {
    try_or(error, null_mut(), || {
        let stream = unsafe { stream.as_mut() }.vortex_expect("stream null");
        let Some(inner) = stream.inner.as_mut() else {
            vortex_bail!("vx_array_stream_next called after finish")
        };

        let element = futures::executor::block_on(inner.stream.next());

        if let Some(element) = element {
            Ok(Box::into_raw(Box::new(vx_array { inner: element? })))
        } else {
            // Drop the stream pointers.
            stream.inner.take();

            Ok(null_mut())
        }
    })
}

/// Predicate function to check if the array stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_finished(stream: *const vx_array_stream) -> bool {
    unsafe { stream.as_ref().vortex_expect("null stream") }
        .inner
        .is_none()
}

/// Free the array stream and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_free(stream: *mut vx_array_stream) {
    assert!(!stream.is_null(), "stream null");
    drop(unsafe { Box::from_raw(stream) });
}
