// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA memory pool attributes.

use std::ffi::c_void;

use crate::DriverError;
use crate::sys;

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
