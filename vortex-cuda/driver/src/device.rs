// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Driver initialization and device queries.

use std::mem::MaybeUninit;

use crate::DriverError;
use crate::sys;
use crate::sys::CUresult;

/// Initializes the driver; repeated calls are cheap.
///
/// Fails with `CUDA_ERROR_NOT_INITIALIZED` if the driver library cannot be loaded, where the raw
/// [`sys`] functions would panic, so that the safe API reports a host without CUDA as an error.
pub fn init() -> Result<(), DriverError> {
    if sys::driver_load_error().is_some() {
        return Err(DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED));
    }
    // SAFETY: cuInit has no preconditions.
    unsafe { sys::cuInit(0) }.result()
}

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
