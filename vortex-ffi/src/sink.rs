use std::ffi::{CStr, c_char};
use std::ptr;

use mpsc::Sender;
use tokio::fs::File;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::file::VortexWriteOptions;
use vortex::stream::ArrayStreamAdapter;

use crate::RUNTIME;
use crate::array::vx_array;
use crate::error::{try_or, vx_error};

#[allow(non_camel_case_types)]
/// The `sink` interface is used to collect array chunks and place them into a resource
/// (e.g. an array stream or file (`vx_array_sink_open_file`)).
pub struct vx_array_sink {
    sink: Sender<VortexResult<ArrayRef>>,
    writer: JoinHandle<VortexResult<File>>,
}

/// Opens a writable array stream, where sink is used to push values into the stream.
/// To close the stream close the sink with `vx_array_sink_close`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_open_file(
    path: *const c_char,
    dtype: *const DType,
    error: *mut *mut vx_error,
) -> *mut vx_array_sink {
    try_or(error, ptr::null_mut(), || {
        let path = unsafe { path.as_ref() }.vortex_expect("null path");
        let path = unsafe { CStr::from_ptr(path) }
            .to_string_lossy()
            .to_string();

        let file_dtype = unsafe { dtype.as_ref().vortex_expect("null dtype") };
        // The channel size 32 was chosen arbitrarily.
        let (sink, rx) = mpsc::channel(32);
        let array_stream = ArrayStreamAdapter::new(file_dtype.clone(), ReceiverStream::new(rx));

        let writer = RUNTIME.spawn(async move {
            let file = File::create(path).await?;
            VortexWriteOptions::default()
                .write(file, array_stream)
                .await
        });

        Ok(Box::into_raw(Box::new(vx_array_sink { sink, writer })))
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
        sink.sink
            .blocking_send(Ok(array.inner.clone()))
            .map_err(|e| vortex_err!("send error {}", e.to_string()))
    })
}

/// Closes an array sink, must be called to ensure all the values pushed to the sink are written
/// to the external resource.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_sink_close(
    sink: *mut vx_array_sink,
    error: *mut *mut vx_error,
) {
    try_or(error, (), || {
        let vx_array_sink { sink, writer } = *Box::from_raw(sink);
        drop(sink);

        RUNTIME.block_on(async {
            let file = writer.await??;
            file.sync_all().await?;
            VortexResult::Ok(())
        })?;

        Ok(())
    })
}
