// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The device's primary context, its event pool, and deferred errors.

use std::ffi::CStr;
use std::ffi::c_char;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use crate::module::CudaModule;
use crate::result;
use crate::result::DriverError;
use crate::stream::CudaEvent;
use crate::stream::CudaStream;
use crate::sys;
use crate::sys::CUresult;

/// The primary context of one device.
///
/// Every operation binds the context to the calling thread first, so a context can be used
/// from any thread.
pub struct CudaContext {
    device: sys::CUdevice,
    context: sys::CUcontext,
    ordinal: usize,
    has_async_alloc: bool,
    /// Whether the context is the device's primary context, released rather than destroyed.
    primary: bool,
    error_state: AtomicU32,
    /// Timing-disabled events returned by finished accesses, reused to avoid creating and
    /// destroying an event per access.
    events: Mutex<Vec<sys::CUevent>>,
}

// SAFETY: The handles are driver handles usable from any thread; the pool is synchronized.
unsafe impl Send for CudaContext {}
// SAFETY: As above.
unsafe impl Sync for CudaContext {}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

impl PartialEq for CudaContext {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.context == other.context
    }
}

impl Eq for CudaContext {}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // Errors here have nowhere to go: they would be recorded in the context being dropped.
        let _ = self.make_current();
        for event in std::mem::take(self.events.get_mut()) {
            // SAFETY: Pooled events are owned by the pool and no longer in use.
            let _ = unsafe { sys::cuEventDestroy_v2(event) };
        }
        // SAFETY: Releases the retain made by `new`, or destroys the context created by
        // `new_non_primary`, once.
        let _ = unsafe {
            if self.primary {
                sys::cuDevicePrimaryCtxRelease_v2(self.device)
            } else {
                sys::cuCtxDestroy_v2(self.context)
            }
        };
    }
}

impl CudaContext {
    /// Retains the primary context of the device with `ordinal`.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        let (device, has_async_alloc) = Self::device(ordinal)?;
        let mut context = std::ptr::null_mut();
        // SAFETY: `device` is valid; the driver writes the retained context.
        unsafe { sys::cuDevicePrimaryCtxRetain(&raw mut context, device) }.result()?;
        Self::wrap(device, has_async_alloc, context, ordinal, true)
    }

    /// Creates a context separate from the device's primary context, with `cuCtxCreate` flags.
    pub fn new_non_primary(ordinal: usize, flags: u32) -> Result<Arc<Self>, DriverError> {
        let (device, has_async_alloc) = Self::device(ordinal)?;
        let mut context = std::ptr::null_mut();
        // SAFETY: `device` is valid; the driver writes the created context.
        unsafe { sys::cuCtxCreate_v2(&raw mut context, flags, device) }.result()?;
        Self::wrap(device, has_async_alloc, context, ordinal, false)
    }

    /// The device with `ordinal`, and whether its allocations can be stream-ordered.
    ///
    /// Everything that can fail is queried here, before a context is retained or created, so
    /// that no failure leaves one behind.
    fn device(ordinal: usize) -> Result<(sys::CUdevice, bool), DriverError> {
        result::init()?;
        let device = result::device::get(
            i32::try_from(ordinal).map_err(|_| DriverError(CUresult::CUDA_ERROR_INVALID_VALUE))?,
        )?;
        // SAFETY: `device` is valid.
        let pools = unsafe {
            result::device::get_attribute(
                device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )
        }?;
        Ok((device, pools > 0))
    }

    fn wrap(
        device: sys::CUdevice,
        has_async_alloc: bool,
        context: sys::CUcontext,
        ordinal: usize,
        primary: bool,
    ) -> Result<Arc<Self>, DriverError> {
        let context = Arc::new(Self {
            device,
            context,
            ordinal,
            has_async_alloc,
            primary,
            error_state: AtomicU32::new(0),
            events: Mutex::new(Vec::new()),
        });
        // On failure, dropping the context releases or destroys it.
        context.bind_to_thread()?;
        Ok(context)
    }

    /// The number of devices.
    pub fn device_count() -> Result<i32, DriverError> {
        result::init()?;
        result::device::get_count()
    }

    /// The device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The raw context handle; do not release it.
    pub fn cu_ctx(&self) -> sys::CUcontext {
        self.context
    }

    /// Whether allocations are stream-ordered through the device's memory pool.
    pub fn has_async_alloc(&self) -> bool {
        self.has_async_alloc
    }

    /// The device's name.
    pub fn name(&self) -> Result<String, DriverError> {
        let mut name = [0 as c_char; 256];
        // SAFETY: The buffer holds 256 bytes, which the driver NUL-terminates.
        unsafe { sys::cuDeviceGetName(name.as_mut_ptr(), 256, self.device) }.result()?;
        // SAFETY: NUL-terminated by the driver.
        Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// A device attribute.
    pub fn attribute(&self, attribute: sys::CUdevice_attribute) -> Result<i32, DriverError> {
        // SAFETY: The device handle is valid for the context's lifetime.
        unsafe { result::device::get_attribute(self.device, attribute) }
    }

    /// The device's free and total memory in bytes.
    pub fn mem_get_info(&self) -> Result<(usize, usize), DriverError> {
        self.bind_to_thread()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: The driver writes both values.
        unsafe { sys::cuMemGetInfo_v2(&raw mut free, &raw mut total) }.result()?;
        Ok((free, total))
    }

    /// Makes this context current on the calling thread, if it is not already.
    ///
    /// First returns, and clears, any error recorded where none could be returned.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        if self.error_state.load(Ordering::Relaxed) != 0 {
            self.check_err()?;
        }
        self.make_current()
    }

    /// Makes this context current on the calling thread, leaving recorded errors in place.
    ///
    /// Cleanup and access tracking bind with this: an unrelated recorded error must not make
    /// them skip a free or a record, nor be consumed where it cannot be reported.
    pub(crate) fn make_current(&self) -> Result<(), DriverError> {
        let mut current = std::ptr::null_mut();
        // SAFETY: The driver writes the current context.
        unsafe { sys::cuCtxGetCurrent(&raw mut current) }.result()?;
        if current != self.context {
            // SAFETY: The context is retained for this object's lifetime.
            unsafe { sys::cuCtxSetCurrent(self.context) }.result()?;
        }
        Ok(())
    }

    /// Blocks until all work in the context completes.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.check_err()?;
        self.synchronize_in_place()
    }

    /// Like [`synchronize`](Self::synchronize), leaving recorded errors in place.
    pub(crate) fn synchronize_in_place(&self) -> Result<(), DriverError> {
        self.make_current()?;
        // SAFETY: The context is current.
        unsafe { sys::cuCtxSynchronize() }.result()
    }

    /// A new non-blocking stream.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        CudaStream::create(self)
    }

    /// A new event with `flags`, by default timing-disabled.
    pub fn new_event(
        self: &Arc<Self>,
        flags: Option<sys::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        CudaEvent::create(
            self,
            flags.unwrap_or(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING),
        )
    }

    /// Loads a module from a cubin, fatbin or PTX image.
    ///
    /// # Safety
    ///
    /// The driver takes no length and reads the image's own: `image` must be a complete cubin
    /// or fatbin, or PTX text followed by a NUL byte.
    pub unsafe fn load_module(
        self: &Arc<Self>,
        image: &[u8],
    ) -> Result<Arc<CudaModule>, DriverError> {
        // SAFETY: Under the caller's guarantees.
        unsafe { CudaModule::load(self, image) }
    }

    /// A timing-disabled event from the pool, or a new one.
    pub(crate) fn take_event(&self) -> Result<sys::CUevent, DriverError> {
        if let Some(event) = self.events.lock().pop() {
            return Ok(event);
        }
        self.make_current()?;
        let mut event = std::ptr::null_mut();
        // SAFETY: The context is current; the driver writes the new event.
        unsafe {
            sys::cuEventCreate(
                &raw mut event,
                sys::CUevent_flags::CU_EVENT_DISABLE_TIMING.0,
            )
        }
        .result()?;
        Ok(event)
    }

    /// Returns an event to the pool once no access will wait for it again.
    ///
    /// Waits already enqueued on it are unaffected by later records.
    pub(crate) fn give_event(&self, event: sys::CUevent) {
        self.events.lock().push(event);
    }

    /// The number of events waiting in the pool.
    #[cfg(test)]
    pub(crate) fn pooled_events(&self) -> usize {
        self.events.lock().len()
    }

    /// The first error recorded where none could be returned, clearing it.
    pub fn check_err(&self) -> Result<(), DriverError> {
        match self.error_state.swap(0, Ordering::Relaxed) {
            0 => Ok(()),
            error => Err(DriverError(CUresult(error))),
        }
    }

    /// Keeps an error from a call that cannot return it, such as in `Drop`.
    ///
    /// An earlier error that was not yet checked is kept instead, as it is likely the cause.
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(error) = result {
            // A failed exchange means an earlier error is pending, which takes precedence.
            let _ = self.error_state.compare_exchange(
                0,
                error.0.0,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }
}
