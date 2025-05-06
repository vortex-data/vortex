use mpsc::Sender;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::dtype::DType;
use vortex::error::{vortex_err, VortexExpect, VortexResult};
use vortex::stream::{ArrayStreamAdapter, ArrayStreamExt};

use crate::array::vx_array;
use crate::error::{try_or, vx_error};
use crate::stream::{ArrayStreamInner, vx_array_stream};

#[allow(non_camel_case_types)]
pub struct vx_array_sink {
    sink: Sender<VortexResult<ArrayRef>>,
}

/// Opens a writable array stream, where sink is used to push values into the stream.
/// To close the stream close the sink with `vx_array_sink_close`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_sink_create(
    dtype: *const DType,
    sink_out: *mut *mut vx_array_sink,
    stream_out: *mut *mut vx_array_stream,
    error: *mut *mut vx_error,
) {
    try_or(error, (), || {
        let file_dtype = unsafe { dtype.as_ref().vortex_expect("null dtype") };

        let (tx, rx) = mpsc::channel(32);
        let array_stream = ArrayStreamAdapter::new(file_dtype.clone(), ReceiverStream::new(rx));
        unsafe {sink_out.write(Box::into_raw( Box::new(vx_array_sink { sink: tx })))};
        unsafe {stream_out.write(Box::into_raw( Box::new(vx_array_stream {
            inner: Some(Box::new(ArrayStreamInner {
                stream: array_stream.boxed(),
            }))})))};
        Ok(())
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
        sink.sink.blocking_send(Ok(array.inner.clone())).map_err(|e| vortex_err!("send error {}", e.to_string()))
    })
}

/// Closes an array sink.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_close(sink: *mut vx_array_sink) {
    let _ = Box::from_raw(sink);
}
