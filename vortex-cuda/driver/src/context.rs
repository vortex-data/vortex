// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The device's primary context and its memory and event pools.

use std::ffi::CStr;
use std::ffi::c_char;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::DriverError;
use crate::device;
use crate::error::log_error;
use crate::event::CudaEvent;
use crate::mem_pool;
use crate::module::CudaModule;
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
    /// Our fixed retaining pool, or null when using the device allocator as a fallback.
    mem_pool: sys::CUmemoryPool,
    /// Whether the context is the device's primary context, released rather than destroyed.
    primary: bool,
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
        if let Err(error) = self.bind_to_thread() {
            log_error("binding context for destruction", error);
        }
        for event in std::mem::take(self.events.get_mut()) {
            // SAFETY: Pooled events are owned by the pool and no longer in use.
            if let Err(error) = unsafe { sys::cuEventDestroy_v2(event) }.result() {
                log_error("destroying cached CUDA event", error);
            }
        }
        if !self.mem_pool.is_null() {
            // SAFETY: Created by `wrap` and destroyed once; allocations still live are released
            // by the driver once they are freed.
            if let Err(error) = unsafe { mem_pool::destroy(self.mem_pool) } {
                log_error("destroying CUDA memory pool", error);
            }
        }
        // SAFETY: Releases the retain made by `new`, or destroys the context created by
        // `new_non_primary`, once.
        let released = unsafe {
            if self.primary {
                sys::cuDevicePrimaryCtxRelease_v2(self.device)
            } else {
                sys::cuCtxDestroy_v2(self.context)
            }
        };
        if let Err(error) = released.result() {
            log_error("releasing CUDA context", error);
        }
    }
}

impl CudaContext {
    /// Retains the primary context of the device with `ordinal` and creates a retaining pool.
    ///
    /// Share the returned `Arc` to share that pool; separate wrappers own separate pools.
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
    /// Query capabilities before retaining or creating a context, so a query failure cannot
    /// leak it. After acquisition, `wrap` establishes ownership before binding can fail.
    fn device(ordinal: usize) -> Result<(sys::CUdevice, bool), DriverError> {
        device::init()?;
        let device = device::get(
            i32::try_from(ordinal).map_err(|_| DriverError(CUresult::CUDA_ERROR_INVALID_VALUE))?,
        )?;
        // SAFETY: `device` is valid.
        let pools = unsafe {
            device::get_attribute(
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
        let mem_pool = if has_async_alloc {
            Self::retaining_pool(device)
        } else {
            std::ptr::null_mut()
        };
        let context = Arc::new(Self {
            device,
            context,
            ordinal,
            has_async_alloc,
            mem_pool,
            primary,
            events: Mutex::new(Vec::new()),
        });
        // On failure, dropping the context releases or destroys it.
        context.bind_to_thread()?;
        Ok(context)
    }

    /// A pool that retains freed memory for reuse, falling back to the device's current pool if
    /// setup fails.
    ///
    /// CUDA's initial release threshold is zero, allowing unused pool memory to be released at
    /// synchronization. Retaining it avoids repeated device-memory mapping work between batches.
    /// This pool uses half the device's memory as its release threshold: a retention target, not a
    /// hard cap on reserved memory or live allocations. Separately constructed context wrappers each
    /// own a pool, so this is not a process-wide budget.
    ///
    /// CUDA can reclaim unused pool memory for unrelated CUDA allocations in the same process.
    /// Allocations in other processes do not trigger that reclamation, so retaining memory here can
    /// reduce what is available to them.
    fn retaining_pool(device: sys::CUdevice) -> sys::CUmemoryPool {
        let mut total = 0;
        // SAFETY: `device` is valid; the driver writes its memory size.
        if let Err(error) = unsafe { sys::cuDeviceTotalMem_v2(&raw mut total, device) }.result() {
            tracing::warn!(
                %error,
                device,
                operation = "querying device memory size",
                "CUDA retaining pool setup failed; using the device's current pool"
            );
            return std::ptr::null_mut();
        }
        // SAFETY: `device` is valid and supports memory pools.
        let pool = match unsafe { mem_pool::create(device) } {
            Ok(pool) => pool,
            Err(error) => {
                tracing::warn!(
                    %error,
                    device,
                    operation = "creating retaining memory pool",
                    "CUDA retaining pool setup failed; using the device's current pool"
                );
                return std::ptr::null_mut();
            }
        };
        let mut threshold = total as u64 / 2;
        // SAFETY: The release threshold is a u64.
        let retained = unsafe {
            mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&raw mut threshold).cast(),
            )
        };
        if let Err(error) = retained {
            tracing::warn!(
                %error,
                device,
                operation = "setting memory pool release threshold",
                "CUDA retaining pool setup failed; using the device's current pool"
            );
            // SAFETY: Created above and never used.
            if let Err(error) = unsafe { mem_pool::destroy(pool) } {
                log_error("destroying unconfigured CUDA memory pool", error);
            }
            return std::ptr::null_mut();
        }
        pool
    }

    /// The number of devices.
    pub fn device_count() -> Result<i32, DriverError> {
        device::init()?;
        device::get_count()
    }

    /// The device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The raw device handle.
    pub fn cu_device(&self) -> sys::CUdevice {
        self.device
    }

    /// The raw context handle; do not release it.
    pub fn cu_ctx(&self) -> sys::CUcontext {
        self.context
    }

    /// Whether allocations are stream-ordered through a memory pool; see [`Self::mem_pool`].
    pub fn has_async_alloc(&self) -> bool {
        self.has_async_alloc
    }

    /// The pool stream-ordered allocations come from, or null for the device's current pool.
    ///
    /// This is a borrowed handle; do not destroy it while this wrapper or its resources use it.
    pub fn mem_pool(&self) -> sys::CUmemoryPool {
        self.mem_pool
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
        unsafe { device::get_attribute(self.device, attribute) }
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
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
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
        self.bind_to_thread()?;
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
    /// `image` must contain a valid, complete cubin or fatbin, with all referenced data
    /// within the slice, or PTX text with a terminating NUL byte within the slice.
    /// The driver receives only a pointer, not the slice length.
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
        self.bind_to_thread()?;
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
}
