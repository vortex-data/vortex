use std::ptr::null_mut;

use mpsc::Sender;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::stream::{ArrayStreamAdapter, ArrayStreamExt};

use crate::array::vx_array;
use crate::error::{try_or, vx_error};
use crate::stream::{ArrayStreamInner, vx_array_stream};

#[allow(non_camel_case_types)]
/// The `sink` object is a writeable array stream, used to go from an external iterator of values
/// into a `vx_array_stream`.
pub struct vx_array_sink {
    sink: Sender<VortexResult<ArrayRef>>,
}

#[allow(non_camel_case_types)]
/// The result of `vx_array_stream_sink_create`.
pub struct vx_array_stream_sink_create_result {
    sink:  Option<vx_array_sink>,
    stream:  Option<vx_array_stream>,
}

/// Opens a writable array stream, where sink is used to push values into the stream.
/// To close the stream close the sink with `vx_array_sink_close`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_sink_create(
    dtype: *const DType,
    error: *mut *mut vx_error,
) -> *mut vx_array_stream_sink_create_result {
    try_or(
        error,
        null_mut(),
        || {
            let file_dtype = unsafe { dtype.as_ref().vortex_expect("null dtype") };

            let (tx, rx) = mpsc::channel(32);
            let array_stream = ArrayStreamAdapter::new(file_dtype.clone(), ReceiverStream::new(rx));
            Ok(Box::into_raw(Box::new(vx_array_stream_sink_create_result {
                sink:Some(vx_array_sink { sink: tx }),
                stream: Some(vx_array_stream {
                    inner: Some(Box::new(ArrayStreamInner {
                        stream: array_stream.boxed(),
                    })),
                }),
            })))
        },
    )
}

/// Moves the sink out of the result type.
/// If the sink has already been taken the second call returns null
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_sink_create_result_get_sink(result: * mut vx_array_stream_sink_create_result) -> *mut vx_array_sink {
    let result = unsafe {result.as_mut()}.vortex_expect("null result");
    let sink = result.sink.take();
    let Some(sink) = sink else {
        return null_mut();
    };
    Box::into_raw(Box::new(sink))
}

/// Moves the stream out of the result type.
/// If the stream has already been taken the second call returns null
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_sink_create_result_get_stream(result: *mut vx_array_stream_sink_create_result) -> *mut vx_array_stream {
    let result = unsafe {result.as_mut()}.vortex_expect("null result");
    let stream = result.stream.take();
    let Some(stream) = stream else {
        return null_mut();
    };
    Box::into_raw(Box::new(stream))
}

// Free the result and any element not already moved out.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_sink_create_result_free(result: *mut vx_array_stream_sink_create_result) {
    drop(Box::from_raw(result))
}

/// Pushed a single array chunk into a file sink.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_push(
    sink: *mut vx_array_sink,
    array: *const vx_array,
    error: *mut *mut vx_error,
) {
    let array = unsafe { array.as_ref().vortex_expect("null array") };
    let sink = unsafe { sink.as_ref().vortex_expect("null array stream") };
    try_or(error, (), || {
        sink.sink
            .blocking_send(Ok(array.inner.clone()))
            .map_err(|e| vortex_err!("send error {}", e.to_string()))
    })
}

/// Closes an array sink.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_close(sink: *mut vx_array_sink) {
    drop(Box::from_raw(sink))
}
