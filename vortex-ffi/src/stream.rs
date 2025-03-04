use futures::StreamExt;
use futures::stream::BoxStream;
use vortex::ArrayRef;
use vortex::error::{VortexExpect, VortexResult};

use crate::RUNTIME;
use crate::array::{FFIArray, FFIArray_free};

/// FFI-exposed stream interface.
#[repr(C)]
pub struct FFIArrayStream {
    pub inner: Option<Box<FFIArrayStreamInner>>,
    pub current: Option<Box<FFIArray>>,
}

/// FFI-compatible interface for dealing with a stream array.
#[repr(C)]
pub struct FFIArrayStreamInner {
    pub(crate) stream: BoxStream<'static, VortexResult<ArrayRef>>,
}

/// Attempt to advance the `current` pointer of the stream.
///
/// A return value of `true` indicates that another element was pulled from the stream, and a return
/// of `false` indicates that the stream is finished.
///
/// It is an error to call this function again after the stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_next(stream: *mut FFIArrayStream) -> bool {
    let stream = &mut *stream;
    let inner = stream
        .inner
        .as_mut()
        .vortex_expect("FFIArrayStream_next called after finish");

    let element = RUNTIME.block_on(async { inner.stream.next().await });

    if let Some(element) = element {
        let inner = element.vortex_expect("element");
        let ffi_array = FFIArray { inner };
        stream.current = Some(Box::new(ffi_array));

        true
    } else {
        // Drop the element and stream pointers.
        stream.current.take();
        stream.inner.take();

        false
    }
}

/// Predicate function to check if the array stream is finished.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_finished(stream: *const FFIArrayStream) -> bool {
    let stream = &*stream;
    stream.inner.is_none()
}

/// Get the current array batch from the stream. Returns a unique pointer.
///
/// It is an error to call this function if the stream is already finished.
///
/// # Safety
///
/// This function is unsafe because it dereferences the `stream` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_current(stream: *mut FFIArrayStream) -> *mut FFIArray {
    let stream = &mut *stream;

    let current = stream
        .current
        .take()
        .vortex_expect("FFIArrayStream_current");

    Box::into_raw(current)
}

/// Free the array stream and all associated resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIArrayStream_free(stream: *mut FFIArrayStream) -> i32 {
    let mut stream = Box::from_raw(stream);

    if let Some(current) = stream.current.take() {
        FFIArray_free(Box::into_raw(current));
    }

    drop(stream.inner.take());

    0
}
