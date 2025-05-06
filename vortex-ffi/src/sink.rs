use std::ptr;

use mpsc::Sender;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult};
use vortex::stream::{ArrayStreamAdapter, ArrayStreamExt};

use crate::array::vx_array;
use crate::error::{try_or, vx_error};
use crate::stream::{ArrayStreamInner, vx_array_stream};

/// An array stream sink writing all values into file path used in creation.
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct vx_array_stream_sink {
    pub sink: vx_array_sink,
    pub stream: vx_array_stream,
}

#[allow(non_camel_case_types)]
pub struct vx_array_sink {
    sink: Sender<VortexResult<ArrayRef>>,
}

// /// Opens a writable array stream
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_sink_create(
    dtype: *const DType,
    error: *mut *mut vx_error,
) -> *mut vx_array_stream_sink {
    try_or(error, ptr::null_mut(), || {
        let file_dtype = unsafe { dtype.as_ref().vortex_expect("null dtype") };

        let (tx, rx) = mpsc::channel(32);
        let array_stream = ArrayStreamAdapter::new(file_dtype.clone(), ReceiverStream::new(rx));
        Ok(Box::into_raw(Box::new(vx_array_stream_sink {
            sink: vx_array_sink { sink: tx },
            stream: vx_array_stream {
                inner: Some(Box::new(ArrayStreamInner {
                    stream: array_stream.boxed(),
                })),
            },
        })))
    })
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
        Ok(sink.sink.blocking_send(Ok(array.inner.clone())).unwrap())
    })
}

/// Closes an array sink.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_close(sink: *mut vx_array_stream_sink) {
    let _ = Box::from_raw(sink);
}
