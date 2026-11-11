// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streams and stream operations on raw handles.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::DriverError;
use crate::context::CudaContext;
use crate::error::log_error;
use crate::event::CudaEvent;
use crate::sys;
use crate::sys::CUresult;

/// A process-unique stream identity: stream handles can be reused after a stream is destroyed,
/// so access tracking never compares raw handles.
pub(crate) type StreamId = u64;

/// The next [`StreamId`].
static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);

/// A queue of device work, executed in order.
pub struct CudaStream {
    id: StreamId,
    stream: sys::CUstream,
    context: Arc<CudaContext>,
}

// SAFETY: Stream handles are usable from any thread.
unsafe impl Send for CudaStream {}
// SAFETY: As above.
unsafe impl Sync for CudaStream {}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("id", &self.id)
            .field("stream", &self.stream)
            .finish()
    }
}

impl PartialEq for CudaStream {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if let Err(error) = self.bind() {
            log_error("binding context to destroy stream", error);
        }
        // SAFETY: Owned streams are destroyed exactly once; queued work still completes.
        if let Err(error) = unsafe { sys::cuStreamDestroy_v2(self.stream) }.result() {
            log_error("destroying CUDA stream", error);
        }
    }
}

impl CudaStream {
    pub(crate) fn create(context: &Arc<CudaContext>) -> Result<Arc<Self>, DriverError> {
        context.bind_to_thread()?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: The context is current; the driver writes the new stream.
        unsafe { sys::cuStreamCreate(&raw mut stream, sys::CU_STREAM_NON_BLOCKING) }.result()?;
        Ok(Arc::new(Self {
            id: NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed),
            stream,
            context: Arc::clone(context),
        }))
    }

    pub(crate) fn id(&self) -> StreamId {
        self.id
    }

    /// The raw stream handle; do not destroy it.
    pub fn cu_stream(&self) -> sys::CUstream {
        self.stream
    }

    /// The stream's context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub(crate) fn bind(&self) -> Result<(), DriverError> {
        self.context.bind_to_thread()
    }

    /// Blocks until all work queued so far completes.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.bind()?;
        // SAFETY: The stream is valid for this object's lifetime.
        unsafe { sys::cuStreamSynchronize(self.stream) }.result()
    }

    /// A new event recording the work queued so far.
    pub fn record_event(
        &self,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let event = self.context.new_event(flags)?;
        event.record(self)?;
        Ok(event)
    }

    /// Makes later work wait for the work recorded in `event`.
    pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
        self.bind()?;
        self.wait_raw(event.cu_event())
    }

    pub(crate) fn wait_raw(&self, event: sys::CUevent) -> Result<(), DriverError> {
        // SAFETY: Both handles belong to this stream's context and are valid.
        unsafe {
            sys::cuStreamWaitEvent(
                self.stream,
                event,
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT.0,
            )
        }
        .result()
    }

    pub(crate) fn record_raw(&self, event: sys::CUevent) -> Result<(), DriverError> {
        // SAFETY: Both handles belong to this stream's context and are valid.
        unsafe { sys::cuEventRecord(event, self.stream) }.result()
    }

    /// Allocates `bytes` on this stream; empty allocations are null pointers on every device.
    pub(crate) fn alloc_raw(&self, bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
        self.bind()?;
        // `cuMemAllocAsync` returns null for zero bytes, but `cuMemAlloc` rejects them.
        if bytes == 0 {
            return Ok(0);
        }
        let mut pointer = 0;
        // SAFETY: The context is current; the driver writes the allocation.
        unsafe {
            if self.context.has_async_alloc() {
                let pool = self.context.mem_pool();
                if pool.is_null() {
                    sys::cuMemAllocAsync(&raw mut pointer, bytes, self.stream)
                } else {
                    sys::cuMemAllocFromPoolAsync(&raw mut pointer, bytes, pool, self.stream)
                }
            } else {
                sys::cuMemAlloc_v2(&raw mut pointer, bytes)
            }
        }
        .result()?;
        Ok(pointer)
    }

    pub(crate) fn free_raw(&self, pointer: sys::CUdeviceptr) -> Result<(), DriverError> {
        // Empty allocations allocated nothing.
        if pointer == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: Callers free an allocation of this context once, after its accesses were
        // ordered before this stream's queued work.
        unsafe {
            if self.context.has_async_alloc() {
                sys::cuMemFreeAsync(pointer, self.stream).result()
            } else {
                sys::cuStreamSynchronize(self.stream).result()?;
                sys::cuMemFree_v2(pointer).result()
            }
        }
    }

    pub(crate) fn memset_d8_raw(
        &self,
        pointer: sys::CUdeviceptr,
        value: u8,
        bytes: usize,
    ) -> Result<(), DriverError> {
        self.bind()?;
        // SAFETY: Callers pass a range of `bytes` writable device bytes ordered for this stream.
        unsafe { sys::cuMemsetD8Async(pointer, value, bytes, self.stream) }.result()
    }
}

/// Enqueues a host function on `stream`.
///
/// # Safety
///
/// `stream` must be valid, and `function` must be safe to call once with `argument` from a
/// driver thread, without making CUDA calls.
pub unsafe fn launch_host_function(
    stream: sys::CUstream,
    function: unsafe extern "C" fn(*mut c_void),
    argument: *mut c_void,
) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees.
    unsafe { sys::cuLaunchHostFunc(stream, Some(function), argument) }.result()
}

/// Enqueues `callback` on `stream` with the status of the preceding work.
///
/// After successful registration, the driver calls it exactly once, even after a device
/// error. Unlike a host function, it is not skipped on error, so it can reclaim resources
/// owned through `argument`.
///
/// # Safety
///
/// `stream` must be valid, and `callback` must be safe to call once with `argument` from a
/// driver thread, without making CUDA calls.
pub unsafe fn add_callback(
    stream: sys::CUstream,
    callback: unsafe extern "C" fn(sys::CUstream, CUresult, *mut c_void),
    argument: *mut c_void,
) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees; the flags are reserved and must be zero.
    unsafe { sys::cuStreamAddCallback(stream, Some(callback), argument, 0) }.result()
}
