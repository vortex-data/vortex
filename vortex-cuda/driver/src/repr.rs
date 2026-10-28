// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Marker traits for values stored in device memory or passed to kernels.

/// A type with the same representation on the device, usable in device memory and as a kernel
/// argument.
///
/// # Safety
///
/// The type must be plain data with a layout the device code agrees on, usually `#[repr(C)]`.
pub unsafe trait DeviceRepr {}

/// A type whose all-zero bit pattern is a valid value.
///
/// # Safety
///
/// Zeroed memory must be a valid value of the type.
pub unsafe trait ValidAsZeroBits {}

macro_rules! impl_plain {
    ($($t:ty),*) => {
        $(
            // SAFETY: Primitive plain data.
            unsafe impl DeviceRepr for $t {}
            // SAFETY: Zero is a valid primitive value.
            unsafe impl ValidAsZeroBits for $t {}
        )*
    };
}

impl_plain!(
    bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);

#[cfg(feature = "f16")]
impl_plain!(half::f16, half::bf16);

#[cfg(feature = "i256")]
// SAFETY: i256 transparently wraps Arrow's pair of 128-bit integers, matching the CUDA layout.
unsafe impl DeviceRepr for vortex_array::dtype::i256 {}

// SAFETY: Arrays of device values are laid out contiguously.
unsafe impl<T: DeviceRepr, const N: usize> DeviceRepr for [T; N] {}
// SAFETY: Arrays of values valid as zeros are valid as zeros.
unsafe impl<T: ValidAsZeroBits, const N: usize> ValidAsZeroBits for [T; N] {}
