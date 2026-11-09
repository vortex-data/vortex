// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Events and event operations on raw handles.

use std::sync::Arc;

use crate::DriverError;
use crate::context::CudaContext;
use crate::error::log_error;
use crate::stream::CudaStream;
use crate::sys;

/// A marker in a stream's work that other streams and the host can wait for.
pub struct CudaEvent {
    event: sys::CUevent,
    context: Arc<CudaContext>,
}

// SAFETY: Event handles are usable from any thread.
unsafe impl Send for CudaEvent {}
// SAFETY: As above.
unsafe impl Sync for CudaEvent {}

impl std::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEvent")
            .field("event", &self.event)
            .finish()
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind_to_thread() {
            log_error("binding context to destroy event", error);
        }
        // SAFETY: Owned events are destroyed once; pending work completes regardless.
        if let Err(error) = unsafe { sys::cuEventDestroy_v2(self.event) }.result() {
            log_error("destroying CUDA event", error);
        }
    }
}

impl CudaEvent {
    pub(crate) fn create(
        context: &Arc<CudaContext>,
        flags: sys::CUevent_flags,
    ) -> Result<Self, DriverError> {
        context.bind_to_thread()?;
        let mut event = std::ptr::null_mut();
        // SAFETY: The context is current; the driver writes the new event.
        unsafe { sys::cuEventCreate(&raw mut event, flags.0) }.result()?;
        Ok(Self {
            event,
            context: Arc::clone(context),
        })
    }

    /// The raw event handle; do not destroy it.
    pub fn cu_event(&self) -> sys::CUevent {
        self.event
    }

    /// The event's context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Records the work queued on `stream` so far; earlier waits are unaffected.
    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.bind()?;
        stream.record_raw(self.event)
    }

    /// Blocks until the recorded work completes.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.context.bind_to_thread()?;
        // SAFETY: The event is valid for this object's lifetime.
        unsafe { sys::cuEventSynchronize(self.event) }.result()
    }

    /// Whether the recorded work completed; errors count as incomplete.
    pub fn is_complete(&self) -> bool {
        matches!(self.try_is_complete(), Ok(true))
    }

    /// Whether the recorded work completed, without waiting.
    pub fn try_is_complete(&self) -> Result<bool, DriverError> {
        // SAFETY: The event is valid for this object's lifetime.
        match unsafe { sys::cuEventQuery(self.event) } {
            sys::CUresult::CUDA_SUCCESS => Ok(true),
            sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            error => Err(DriverError(error)),
        }
    }

    /// Milliseconds from this event to `end`, after both complete.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
        self.synchronize()?;
        end.synchronize()?;
        let mut milliseconds = 0.0;
        // SAFETY: Both events completed and were created with timing.
        unsafe { sys::cuEventElapsedTime(&raw mut milliseconds, self.event, end.event) }
            .result()?;
        Ok(milliseconds)
    }
}

/// Blocks until the work recorded in `event` completes.
///
/// # Safety
///
/// `event` must be a valid event handle.
pub unsafe fn synchronize(event: sys::CUevent) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees.
    unsafe { sys::cuEventSynchronize(event) }.result()
}
