// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streams and events.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::context::CudaContext;
use crate::result::DriverError;
use crate::sys;

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
    owned: bool,
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
        if self.owned {
            self.context.record_err(self.context.make_current());
            // SAFETY: Owned streams are destroyed exactly once; queued work still completes.
            self.context
                .record_err(unsafe { sys::cuStreamDestroy_v2(self.stream) }.result());
        }
    }
}

impl CudaStream {
    pub(crate) fn create(
        context: &Arc<CudaContext>,
        priority: Option<i32>,
    ) -> Result<Arc<Self>, DriverError> {
        context.bind_to_thread()?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: The context is current; the driver writes the new stream.
        unsafe {
            match priority {
                Some(priority) => sys::cuStreamCreateWithPriority(
                    &raw mut stream,
                    sys::CU_STREAM_NON_BLOCKING,
                    priority,
                ),
                None => sys::cuStreamCreate(&raw mut stream, sys::CU_STREAM_NON_BLOCKING),
            }
        }
        .result()?;
        Ok(Arc::new(Self {
            id: NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed),
            stream,
            context: Arc::clone(context),
            owned: true,
        }))
    }

    /// Wraps a stream owned by someone else, which is not destroyed on drop.
    ///
    /// # Safety
    ///
    /// `stream` must belong to `context` and outlive the returned wrapper.
    pub unsafe fn borrowed(context: &Arc<CudaContext>, stream: sys::CUstream) -> Arc<Self> {
        Arc::new(Self {
            id: NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed),
            stream,
            context: Arc::clone(context),
            owned: false,
        })
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

    /// Like [`bind`](Self::bind), leaving recorded errors in place.
    pub(crate) fn make_current(&self) -> Result<(), DriverError> {
        self.context.make_current()
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

    /// Makes later work wait for the work queued on `other` so far.
    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
    }

    /// A new stream whose work follows the work queued here so far.
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        let stream = Self::create(&self.context, None)?;
        stream.join(self)?;
        Ok(stream)
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

    pub(crate) fn alloc_raw(&self, bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
        self.bind()?;
        let mut pointer = 0;
        // SAFETY: The context is current; the driver writes the allocation.
        unsafe {
            if self.context.has_async_alloc() {
                sys::cuMemAllocAsync(&raw mut pointer, bytes, self.stream)
            } else {
                sys::cuMemAlloc_v2(&raw mut pointer, bytes)
            }
        }
        .result()?;
        Ok(pointer)
    }

    pub(crate) fn free_raw(&self, pointer: sys::CUdeviceptr) -> Result<(), DriverError> {
        self.make_current()?;
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
        self.context.record_err(self.context.make_current());
        // SAFETY: Owned events are destroyed once; pending work completes regardless.
        self.context
            .record_err(unsafe { sys::cuEventDestroy_v2(self.event) }.result());
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
    ///
    /// Unlike most operations, this leaves errors recorded elsewhere in the context in place,
    /// so that dropping a host buffer can wait for its own transfers despite them.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.context.make_current()?;
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
