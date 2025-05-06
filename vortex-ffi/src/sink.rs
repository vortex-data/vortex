use std::ffi::{c_char, CStr};
use std::ptr;
use tokio::fs::File;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult};
use vortex::file::VortexWriteOptions;
use vortex::stream::ArrayStreamAdapter;
use crate::array::vx_array;
use crate::error::{try_or, vx_error};
use crate::RUNTIME;

/// An array stream sink writing all values into file path used in creation.
#[allow(non_camel_case_types)]
pub struct vx_array_stream_file_sink {
    pub(crate) sender: mpsc::Sender<VortexResult<ArrayRef>>,
    pub(crate) writer: JoinHandle<VortexResult<File>>,
}

/// Opens an array stream
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_file_sink_open(
    path: *const c_char,
    dtype: *const DType,
    error: *mut *mut vx_error,
) -> *mut vx_array_stream_file_sink {
    try_or(error, ptr::null_mut(), || {
        let path = CStr::from_ptr(path).to_str()?;
        let file_dtype = unsafe { dtype.as_ref().vortex_expect("null dtype") };

        let file = RUNTIME.block_on(File::create(path))?;

        let (tx, rx) = mpsc::channel(32);
        let array_stream = ArrayStreamAdapter::new(file_dtype.clone(), ReceiverStream::new(rx));
        let writer = RUNTIME.spawn(VortexWriteOptions::default().write(file, array_stream));

        Ok(Box::into_raw(Box::new(vx_array_stream_file_sink {
            sender: tx,
            writer,
        })))
    })
}

/// Pushed a single array chunk into a file sink.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_file_sink_push_array(
    array_stream: *mut vx_array_stream_file_sink,
    array: *const vx_array,
    error: *mut *mut vx_error,
) {
    let array = unsafe { array.as_ref().vortex_expect("null array") };
    let array_stream = unsafe { array_stream.as_ref().vortex_expect("null array stream") };
    try_or(error, (), || {
        Ok(array_stream
            .sender
            .blocking_send(Ok(array.inner.clone()))
            .unwrap())
    })
}

/// Closes a array stream ensuring that all array pushed into the sink are written to the file.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_array_stream_file_sink_close(
    array_stream: *mut vx_array_stream_file_sink,
    error: *mut *mut vx_error,
) {
    try_or(error, (), || {
        let array_stream = Box::from_raw(array_stream);

        let vx_array_stream_file_sink { sender, writer, .. } = *array_stream;
        // Close the sender stream.
        drop(sender);

        RUNTIME.block_on(async {
            let file = writer.await??;
            file.sync_all().await?;
            VortexResult::Ok(())
        })
    })
}
