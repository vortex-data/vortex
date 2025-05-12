use std::ptr;

use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::stream::{ArrayStream, SendableArrayStream};

/// FFI-exposed stream interface.
#[allow(non_camel_case_types)]
pub struct vx_array_stream {
    pub inner: Option<Box<ArrayStreamInner>>,
}

/// FFI-compatible interface for dealing with a stream array.
pub struct ArrayStreamInner {
    pub(crate) stream: SendableArrayStream,
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
