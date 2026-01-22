// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA stream utility functions.

use cudarc::driver::CudaStream;
use cudarc::driver::result::stream;
use kanal::Sender;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// Registers a callback and asynchronously waits for its completion.
///
/// This function can be used to asynchronously wait for events previously
/// submitted to the stream to complete, e.g. async buffer allocations.
///
/// Note: This is not equivalent to calling sync on a stream but only awaits
/// the registered callback to complete.
///
/// # Arguments
///
/// * `stream` - The CUDA stream to wait on
///
/// # Errors
///
/// Returns an error if registering the stream callback fails or if the callback
/// channel closes unexpectedly.
pub async fn await_stream_callback(stream: &CudaStream) -> VortexResult<()> {
    let rx = register_stream_callback(stream)?;

    rx.recv()
        .await
        .map_err(|e| vortex_err!("CUDA stream callback channel closed unexpectedly: {}", e))
}

/// Registers a host function callback on the stream.
///
/// # Returns
///
/// An async receiver that receives a message when all preceding work on the
/// stream completes.
///
/// # Errors
///
/// Returns an error if registering the host callback function fails.
fn register_stream_callback(stream: &CudaStream) -> VortexResult<kanal::AsyncReceiver<()>> {
    let (tx, rx) = kanal::bounded::<()>(1);

    let tx_ptr = Box::into_raw(Box::new(tx));

    /// Called from CUDA driver thread when all preceding work on the stream completes.
    unsafe extern "C" fn callback(user_data: *mut std::ffi::c_void) {
        // SAFETY: The memory of `tx` is manually managed has not been freed
        // before. We have unique ownership and can therefore free it.
        let tx = unsafe { Box::from_raw(user_data as *mut Sender<()>) };

        // Blocking send as we're in a callback invoked by the CUDA driver.
        #[expect(clippy::expect_used)]
        tx.send(())
            // A send should never fail. Panic otherwise.
            .expect("CUDA callback receiver dropped unexpectedly");
    }

    // SAFETY:
    // 1. Valid handle from the borrowed `CudaStream`.
    // 2. Valid function pointer with the the correct signature
    // 3. Valid user data pointer which is consumed exactly once
    unsafe {
        stream::launch_host_function(
            stream.cu_stream(),
            callback,
            tx_ptr as *mut std::ffi::c_void,
        )
        .map_err(|err| {
            // SAFETY: Registration failed, so the callback will never run.
            // We have unique ownership and can therefore free it.
            drop(Box::from_raw(tx_ptr));
            vortex_err!("Failed to register CUDA stream callback: {}", err)
        })?;
    }

    Ok(rx.to_async())
}
