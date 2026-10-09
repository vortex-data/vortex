// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small byte-table take with a fallback for unsupported targets and inputs.

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_endian = "little"))] {
        mod neon;
    } else if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
        mod avx2;
    } else {
        mod fallback {
            use super::super::FixedWidthTakeValue;

            // SAFETY: u8 has no padding or uninitialized bytes.
            unsafe impl FixedWidthTakeValue for u8 {}

            // SAFETY: i8 has no padding or uninitialized bytes.
            unsafe impl FixedWidthTakeValue for i8 {}

            // SAFETY: Byte arrays have no padding and every byte is initialized.
            unsafe impl<const N: usize> FixedWidthTakeValue for [u8; N] {}
        }
    }
}
