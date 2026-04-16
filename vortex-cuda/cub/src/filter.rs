// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Safe wrappers around CUB DeviceSelect::Flagged for GPU filtering.
//!
//! This module provides type-safe Rust functions that wrap CUB filter
//! operations. The underlying FFI functions are loaded at runtime via
//! libloading.

use std::ffi::c_void;

use crate::cub_library;
use crate::error::CubError;
use crate::error::check_cuda_error;
pub use crate::sys::cudaStream_t;

/// Trait for types to support CUB filter operations.
pub trait CubFilterable: Copy + 'static {
    /// Get the temporary storage size required for filtering `num_items` elements.
    fn get_temp_size(num_items: i64) -> Result<usize, CubError>;

    /// Execute CUB DeviceSelect::Flagged with a byte mask (one byte per element).
    ///
    /// # Safety
    ///
    /// All device pointers must be valid and properly sized:
    /// - `d_temp` must have at least `temp_bytes` allocated
    /// - `d_in` must have at least `num_items` elements
    /// - `d_flags` must have at least `num_items` bytes (one per element, 0 or 1)
    /// - `d_out` must have enough space for selected elements
    /// - `d_num_selected` must point to a valid i64 on the device
    #[expect(clippy::too_many_arguments)]
    unsafe fn filter_bytemask(
        d_temp: *mut c_void,
        temp_bytes: usize,
        d_in: *const Self,
        d_flags: *const u8,
        d_out: *mut Self,
        d_num_selected: *mut i64,
        num_items: i64,
        stream: cudaStream_t,
    ) -> Result<(), CubError>;

    /// Execute CUB DeviceSelect::Flagged with a bit mask (one bit per element).
    ///
    /// This version accepts packed bits directly, avoiding the need to expand
    /// bits to bytes in a separate kernel. Uses CUB's TransformInputIterator
    /// internally to read bits on-the-fly during the filter operation.
    ///
    /// # Safety
    ///
    /// All device pointers must be valid and properly sized:
    /// - `d_temp` must have at least `temp_bytes` allocated
    /// - `d_in` must have at least `num_items` elements
    /// - `d_bitmask` must contain enough bytes to hold `bit_offset + num_items` bits
    /// - `d_out` must have enough space for selected elements
    /// - `d_num_selected` must point to a valid i64 on the device
    #[expect(clippy::too_many_arguments)]
    unsafe fn filter_bitmask(
        d_temp: *mut c_void,
        temp_bytes: usize,
        d_in: *const Self,
        d_bitmask: *const u8,
        bit_offset: u64,
        d_out: *mut Self,
        d_num_selected: *mut i64,
        num_items: i64,
        stream: cudaStream_t,
    ) -> Result<(), CubError>;
}

macro_rules! impl_filter {
    ($($suffix:ident => $ty:ty),* $(,)?) => {
        paste::paste! {
            $(
                impl CubFilterable for $ty {
                    fn get_temp_size(num_items: i64) -> Result<usize, CubError> {
                        [<filter_get_temp_size_ $suffix>](num_items)
                    }

                    unsafe fn filter_bytemask(
                        d_temp: *mut c_void,
                        temp_bytes: usize,
                        d_in: *const Self,
                        d_flags: *const u8,
                        d_out: *mut Self,
                        d_num_selected: *mut i64,
                        num_items: i64,
                        stream: cudaStream_t,
                    ) -> Result<(), CubError> {
                        // SAFETY: Caller ensures all pointers are valid.
                        unsafe {
                            [<filter_bytemask_ $suffix>](
                                d_temp, temp_bytes, d_in, d_flags, d_out, d_num_selected, num_items, stream,
                            )
                        }
                    }

                    unsafe fn filter_bitmask(
                        d_temp: *mut c_void,
                        temp_bytes: usize,
                        d_in: *const Self,
                        d_bitmask: *const u8,
                        bit_offset: u64,
                        d_out: *mut Self,
                        d_num_selected: *mut i64,
                        num_items: i64,
                        stream: cudaStream_t,
                    ) -> Result<(), CubError> {
                        // SAFETY: Caller ensures all pointers are valid.
                        unsafe {
                            [<filter_bitmask_ $suffix>](
                                d_temp, temp_bytes, d_in, d_bitmask, bit_offset, d_out, d_num_selected, num_items, stream,
                            )
                        }
                    }
                }

                #[doc = "Get the temporary storage size required for filtering elements."]
                pub fn [<filter_get_temp_size_ $suffix>](num_items: i64) -> Result<usize, CubError> {
                    let lib = cub_library()?;
                    let mut temp_bytes: usize = 0;
                    let err = unsafe { (lib.[<filter_temp_size_ $suffix>])(&raw mut temp_bytes, num_items) };
                    check_cuda_error(err, concat!("filter_temp_size_", stringify!($suffix)))?;
                    Ok(temp_bytes)
                }

                #[doc = "Filter elements using a byte mask (one byte per element)."]
                ///
                /// # Safety
                ///
                /// All device pointers must be valid and properly sized:
                /// - `d_temp` must have at least `temp_bytes` allocated
                /// - `d_in` must have at least `num_items` elements
                /// - `d_flags` must have at least `num_items` bytes (one per element, 0 or 1)
                /// - `d_out` must have enough space for selected elements
                /// - `d_num_selected` must point to a valid i64 on the device
                #[expect(clippy::too_many_arguments)]
                pub unsafe fn [<filter_bytemask_ $suffix>](
                    d_temp: *mut c_void,
                    temp_bytes: usize,
                    d_in: *const $ty,
                    d_flags: *const u8,
                    d_out: *mut $ty,
                    d_num_selected: *mut i64,
                    num_items: i64,
                    stream: cudaStream_t,
                ) -> Result<(), CubError> {
                    let lib = cub_library()?;
                    let err = unsafe {
                        (lib.[<filter_bytemask_ $suffix>])(
                            d_temp,
                            temp_bytes,
                            d_in,
                            d_flags,
                            d_out,
                            d_num_selected,
                            num_items,
                            stream,
                        )
                    };
                    check_cuda_error(err, concat!("filter_bytemask_", stringify!($suffix)))
                }

                #[doc = "Filter elements using a bit mask (one bit per element)."]
                ///
                /// This version accepts packed bits directly, avoiding the need to expand
                /// bits to bytes in a separate kernel.
                ///
                /// # Safety
                ///
                /// All device pointers must be valid and properly sized:
                /// - `d_temp` must have at least `temp_bytes` allocated
                /// - `d_in` must have at least `num_items` elements
                /// - `d_bitmask` must contain enough bytes to hold `bit_offset + num_items` bits
                /// - `d_out` must have enough space for selected elements
                /// - `d_num_selected` must point to a valid i64 on the device
                #[expect(clippy::too_many_arguments)]
                pub unsafe fn [<filter_bitmask_ $suffix>](
                    d_temp: *mut c_void,
                    temp_bytes: usize,
                    d_in: *const $ty,
                    d_bitmask: *const u8,
                    bit_offset: u64,
                    d_out: *mut $ty,
                    d_num_selected: *mut i64,
                    num_items: i64,
                    stream: cudaStream_t,
                ) -> Result<(), CubError> {
                    let lib = cub_library()?;
                    let err = unsafe {
                        (lib.[<filter_bitmask_ $suffix>])(
                            d_temp,
                            temp_bytes,
                            d_in,
                            d_bitmask,
                            bit_offset,
                            d_out,
                            d_num_selected,
                            num_items,
                            stream,
                        )
                    };
                    check_cuda_error(err, concat!("filter_bitmask_", stringify!($suffix)))
                }
            )*
        }
    };
}

impl_filter! {
    u8 => u8,
    i8 => i8,
    u16 => u16,
    i16 => i16,
    u32 => u32,
    i32 => i32,
    u64 => u64,
    i64 => i64,
    f32 => f32,
    f64 => f64,
    i128 => i128,
    i256 => vortex_array::dtype::i256,
}
