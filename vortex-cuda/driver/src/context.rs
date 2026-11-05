// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The device's primary context, its event pool, and deferred errors.

use std::ffi::CStr;
use std::ffi::c_char;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use crate::DriverError;
use crate::device;
use crate::event::CudaEvent;
use crate::mem_pool;
use crate::mem_pool::MemPool;
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
    /// Allocation policy is fixed at construction; only an owned pool is destroyed on drop.
    mem_pool: MemPool,
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
        if let MemPool::Owned(pool) = self.mem_pool {
            // SAFETY: Created by `wrap` and destroyed once; allocations still live are released
            // by the driver once they are freed.
            let _ = unsafe { mem_pool::destroy(pool) };
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
        Self::wrap(device, has_async_alloc, context, ordinal, true, None)
    }

    /// Retains the primary context with a borrowed pool instead of creating a retaining pool.
    ///
    /// The pool choice is fixed for this wrapper and does not affect other wrappers of the same
    /// primary context. Null selects the device's current pool. Without stream-ordered allocation
    /// support, allocations use the synchronous allocator as usual.
    ///
    /// # Safety
    ///
    /// A non-null `pool` must be a device memory pool of the device with `ordinal`. It must outlive
    /// this wrapper and all resources using it, including exported buffers, and remain valid until
    /// their queued frees complete. The caller remains responsible for destroying the pool.
    pub unsafe fn new_with_mem_pool(
        ordinal: usize,
        pool: sys::CUmemoryPool,
    ) -> Result<Arc<Self>, DriverError> {
        let (device, has_async_alloc) = Self::device(ordinal)?;
        let mut context = std::ptr::null_mut();
        // SAFETY: `device` is valid; the driver writes the retained context.
        unsafe { sys::cuDevicePrimaryCtxRetain(&raw mut context, device) }.result()?;
        Self::wrap(device, has_async_alloc, context, ordinal, true, Some(pool))
    }

    /// Creates a context separate from the device's primary context, with `cuCtxCreate` flags.
    pub fn new_non_primary(ordinal: usize, flags: u32) -> Result<Arc<Self>, DriverError> {
        let (device, has_async_alloc) = Self::device(ordinal)?;
        let mut context = std::ptr::null_mut();
        // SAFETY: `device` is valid; the driver writes the created context.
        unsafe { sys::cuCtxCreate_v2(&raw mut context, flags, device) }.result()?;
        Self::wrap(device, has_async_alloc, context, ordinal, false, None)
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
        pool: Option<sys::CUmemoryPool>,
    ) -> Result<Arc<Self>, DriverError> {
        let mem_pool = match pool {
            Some(pool) if !pool.is_null() => MemPool::Borrowed(pool),
            Some(_) => MemPool::Device,
            None if has_async_alloc => Self::retaining_pool(device),
            None => MemPool::Device,
        };
        let context = Arc::new(Self {
            device,
            context,
            ordinal,
            has_async_alloc,
            mem_pool,
            primary,
            error_state: AtomicU32::new(0),
            events: Mutex::new(Vec::new()),
        });
        // On failure, dropping the context releases or destroys it.
        context.bind_to_thread()?;
        Ok(context)
    }

    /// A pool that keeps freed memory reserved for reuse, up to half the device, or the device's
    /// pool if setup fails.
    ///
    /// The device's default pool returns freed memory to the driver at every synchronization, so
    /// scans that free and reallocate their batches between synchronizations pay for remapping
    /// device memory on every batch. At each synchronization this pool releases freed memory beyond
    /// half the device: the threshold bounds what it keeps idle, not what it can allocate. Each
    /// wrapper using this default policy owns its pool. Failures fall back to the device's pool
    /// rather than failing the context.
    fn retaining_pool(device: sys::CUdevice) -> MemPool {
        let mut total = 0;
        // SAFETY: `device` is valid; the driver writes its memory size.
        if unsafe { sys::cuDeviceTotalMem_v2(&raw mut total, device) }
            .result()
            .is_err()
        {
            return MemPool::Device;
        }
        // SAFETY: `device` is valid and supports memory pools.
        let Ok(pool) = (unsafe { mem_pool::create(device) }) else {
            return MemPool::Device;
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
        if retained.is_err() {
            // SAFETY: Created above and never used.
            let _ = unsafe { mem_pool::destroy(pool) };
            return MemPool::Device;
        }
        MemPool::Owned(pool)
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
        match self.mem_pool {
            MemPool::Owned(pool) | MemPool::Borrowed(pool) => pool,
            MemPool::Device => std::ptr::null_mut(),
        }
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
        CudaStream::create(self, None)
    }

    /// A new non-blocking stream with `priority`; lower values are higher priorities.
    pub fn new_stream_with_priority(
        self: &Arc<Self>,
        priority: i32,
    ) -> Result<Arc<CudaStream>, DriverError> {
        CudaStream::create(self, Some(priority))
    }

    /// The legacy default stream.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        // SAFETY: The null stream is valid for the context's lifetime.
        unsafe { CudaStream::borrowed(self, std::ptr::null_mut()) }
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
