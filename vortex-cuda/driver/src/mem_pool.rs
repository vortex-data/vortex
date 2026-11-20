// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA memory pool creation, destruction, and attributes.

use std::ffi::c_void;
use std::mem::MaybeUninit;

use crate::DriverError;
use crate::sys;

/// Creates a pool of pinned device memory on `device`.
///
/// # Safety
///
/// `device` must be a valid device handle that supports memory pools.
pub unsafe fn create(device: sys::CUdevice) -> Result<sys::CUmemoryPool, DriverError> {
    let props = sys::CUmemPoolProps {
        allocType: sys::CU_MEM_ALLOCATION_TYPE_PINNED,
        handleTypes: sys::CU_MEM_HANDLE_TYPE_NONE,
        location: sys::CUmemLocation {
            type_: sys::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32SecurityAttributes: std::ptr::null_mut(),
        reserved: [0; 64],
    };
    let mut pool = MaybeUninit::uninit();
    // SAFETY: Under the caller's guarantees; the driver writes the created pool.
    unsafe { sys::cuMemPoolCreate(pool.as_mut_ptr(), &raw const props) }.result()?;
    // SAFETY: Initialized by the successful call.
    Ok(unsafe { pool.assume_init() })
}

/// Destroys `pool`. Memory still allocated from it is released once freed.
///
/// # Safety
///
/// `pool` must be a pool created by [`create`], destroyed once, and no longer used to allocate.
pub unsafe fn destroy(pool: sys::CUmemoryPool) -> Result<(), DriverError> {
    // SAFETY: Under the caller's guarantees.
    unsafe { sys::cuMemPoolDestroy(pool) }.result()
}

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
