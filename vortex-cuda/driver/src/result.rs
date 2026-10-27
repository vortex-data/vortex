// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Driver calls returning [`Result`]s, for callers that need the raw handles.

use std::ffi::CStr;
use std::ffi::c_uint;
use std::ffi::c_void;
use std::fmt;
use std::mem::MaybeUninit;

use crate::sys;
use crate::sys::CUresult;

/// A failed CUDA driver call.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DriverError(pub CUresult);

impl DriverError {
    /// The driver's name for the error, such as `CUDA_ERROR_INVALID_VALUE`.
    pub fn error_name(&self) -> Result<&'static CStr, DriverError> {
        // Errors are formatted even when the driver library is missing.
        if !driver_present() {
            return Err(*self);
        }
        let mut name = std::ptr::null();
        // SAFETY: The driver writes a pointer to a static string.
        unsafe { sys::cuGetErrorName(self.0, &raw mut name) }.result()?;
        // SAFETY: Successful calls return a NUL-terminated static string.
        Ok(unsafe { CStr::from_ptr(name) })
    }

    /// The driver's description of the error.
    pub fn error_string(&self) -> Result<&'static CStr, DriverError> {
        if !driver_present() {
            return Err(*self);
        }
        let mut description = std::ptr::null();
        // SAFETY: The driver writes a pointer to a static string.
        unsafe { sys::cuGetErrorString(self.0, &raw mut description) }.result()?;
        // SAFETY: Successful calls return a NUL-terminated static string.
        Ok(unsafe { CStr::from_ptr(description) })
    }
}

impl fmt::Debug for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.error_name() {
            Ok(name) => write!(f, "DriverError({}, {})", self.0.0, name.to_string_lossy()),
            Err(_) => write!(f, "DriverError({})", self.0.0),
        }
    }
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.error_name(), self.error_string()) {
            (Ok(name), Ok(description)) => write!(
                f,
                "{}: {}",
                name.to_string_lossy(),
                description.to_string_lossy()
            ),
            _ => write!(f, "CUDA driver error {}", self.0.0),
        }
    }
}

impl std::error::Error for DriverError {}

impl CUresult {
    /// `Ok` for [`CUresult::CUDA_SUCCESS`], otherwise the error.
    #[inline]
    pub fn result(self) -> Result<(), DriverError> {
        if self == CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(DriverError(self))
        }
    }
}

/// Initializes the driver; repeated calls are cheap.
///
/// Fails with `CUDA_ERROR_NOT_INITIALIZED` if the driver library cannot be loaded, where the raw
/// [`sys`] functions would panic, so that the safe API reports a host without CUDA as an error.
pub fn init() -> Result<(), DriverError> {
    if !driver_present() {
        return Err(DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED));
    }
    // SAFETY: cuInit has no preconditions.
    unsafe { sys::cuInit(0) }.result()
}

/// Whether the driver library is loaded, which the raw [`sys`] functions require.
fn driver_present() -> bool {
    // SAFETY: Loads the driver library, which any driver call would load as well.
    unsafe { sys::is_culib_present() }
}

/// Allocates `bytes` of page-locked host memory with the given `cuMemHostAlloc` flags.
///
/// # Safety
///
/// A CUDA context must be current; the memory is uninitialized.
pub unsafe fn malloc_host(bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
    let mut pointer = MaybeUninit::uninit();
    // SAFETY: The driver writes the allocation into `pointer` on success.
    unsafe { sys::cuMemHostAlloc(pointer.as_mut_ptr(), bytes, flags) }.result()?;
    // SAFETY: Initialized by the successful call.
    Ok(unsafe { pointer.assume_init() })
}

/// Frees memory allocated by [`malloc_host`].
///
/// # Safety
///
/// `pointer` must come from [`malloc_host`] and no pending copy may still use it.
pub unsafe fn free_host(pointer: *mut c_void) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees.
    unsafe { sys::cuMemFreeHost(pointer) }.result()
}

/// Synchronously copies `target.len()` elements from device memory at `source`.
///
/// # Safety
///
/// `source` must address at least `size_of_val(target)` readable device bytes.
pub unsafe fn memcpy_dtoh_sync<T>(
    target: &mut [T],
    source: sys::CUdeviceptr,
) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees; `target` is writable for its whole size.
    unsafe { sys::cuMemcpyDtoH_v2(target.as_mut_ptr().cast(), source, size_of_val(target)) }
        .result()
}

/// Device queries.
pub mod device {
    use super::*;

    /// The device with the given ordinal.
    pub fn get(ordinal: i32) -> Result<sys::CUdevice, DriverError> {
        let mut device = MaybeUninit::uninit();
        // SAFETY: The driver writes the device into `device` on success.
        unsafe { sys::cuDeviceGet(device.as_mut_ptr(), ordinal) }.result()?;
        // SAFETY: Initialized by the successful call.
        Ok(unsafe { device.assume_init() })
    }

    /// The number of devices.
    pub fn get_count() -> Result<i32, DriverError> {
        let mut count = 0;
        // SAFETY: The driver writes the count.
        unsafe { sys::cuDeviceGetCount(&raw mut count) }.result()?;
        Ok(count)
    }

    /// A device attribute.
    ///
    /// # Safety
    ///
    /// `device` must be a valid device handle.
    pub unsafe fn get_attribute(
        device: sys::CUdevice,
        attribute: sys::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut value = 0;
        // SAFETY: Under the caller's guarantees.
        unsafe { sys::cuDeviceGetAttribute(&raw mut value, attribute, device) }.result()?;
        Ok(value)
    }

    /// The device's current memory pool.
    ///
    /// # Safety
    ///
    /// `device` must be a valid device handle.
    pub unsafe fn get_mem_pool(device: sys::CUdevice) -> Result<sys::CUmemoryPool, DriverError> {
        let mut pool = MaybeUninit::uninit();
        // SAFETY: Under the caller's guarantees.
        unsafe { sys::cuDeviceGetMemPool(pool.as_mut_ptr(), device) }.result()?;
        // SAFETY: Initialized by the successful call.
        Ok(unsafe { pool.assume_init() })
    }
}

/// Memory pool attributes.
pub mod mem_pool {
    use super::*;

    /// Reads a pool attribute into `value`.
    ///
    /// # Safety
    ///
    /// `value` must point to storage of the attribute's type.
    pub unsafe fn get_attribute(
        pool: sys::CUmemoryPool,
        attribute: sys::CUmemPool_attribute,
        value: *mut c_void,
    ) -> Result<(), DriverError> {
        // SAFETY: Under the caller's guarantees.
        unsafe { sys::cuMemPoolGetAttribute(pool, attribute, value) }.result()
    }

    /// Sets a pool attribute from `value`.
    ///
    /// # Safety
    ///
    /// `value` must point to a value of the attribute's type.
    pub unsafe fn set_attribute(
        pool: sys::CUmemoryPool,
        attribute: sys::CUmemPool_attribute,
        value: *mut c_void,
    ) -> Result<(), DriverError> {
        // SAFETY: Under the caller's guarantees.
        unsafe { sys::cuMemPoolSetAttribute(pool, attribute, value) }.result()
    }
}

/// Stream operations on raw handles.
pub mod stream {
    use super::*;

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
}

/// Event operations on raw handles.
pub mod event {
    use super::*;

    /// Blocks until the work recorded in `event` completes.
    ///
    /// # Safety
    ///
    /// `event` must be a valid event handle.
    pub unsafe fn synchronize(event: sys::CUevent) -> Result<(), DriverError> {
        // SAFETY: Under the caller's guarantees.
        unsafe { sys::cuEventSynchronize(event) }.result()
    }
}
