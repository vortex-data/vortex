// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unlike other SIMD ISAs, NEON does not have a compress instruction.
//! Therefore, we actually use different logic depending on the bit-width of the data type.

// There's simply too many intrinsics to bother with this!
#![allow(unsafe_op_in_unsafe_fn)]

mod neon_u16;
mod neon_u32;
mod neon_u8;

use std::arch::is_aarch64_feature_detected;

use vortex_buffer::BitView;
use vortex_error::vortex_panic;

/// Benchmark wrapper for [`filter_neon`].
#[doc(hidden)]
#[cfg(feature = "bench")]
#[inline(never)]
pub fn bench_filter_neon<const NB: usize, T: Copy>(bit_view: &BitView<NB>, slice: &mut [T]) {
    if is_aarch64_feature_detected!("neon") {
        unsafe { filter_neon(slice, bit_view) }
    }
    vortex_panic!("NEON not detected on this CPU");
}

/// Filters the given slice of items in place according to the provided BitView using neon
/// (non-SIMD) code.
///
/// The caller *should* handle where the BitView has zero or full true counts to avoid unnecessary
/// work.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn filter_neon<const NB: usize, T: Copy>(slice: &mut [T], mask: &BitView<NB>) {
    assert_eq!(
        slice.len(),
        BitView::<NB>::N,
        "Slice length must match BitView length"
    );
    match size_of::<T>() {
        1 => neon_u8::filter_neon_u8(slice.as_mut_ptr() as *mut u8, mask),
        2 => neon_u16::filter_neon_u16(slice.as_mut_ptr() as *mut u16, mask),
        4 => neon_u32::filter_neon_u32(slice.as_mut_ptr() as *mut u32, mask),
        _ => {
            // Fallback to scalar for wider sizes
            super::scalar::filter_scalar(slice, mask)
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_u32() {
        const NB: usize = 128; // 1024 bits
        const N: usize = NB * 8;

        let view = BitView::<NB>::with_prefix(512);
        let mut data: Vec<u32> = (0..N).map(|i| i as u32).collect();

        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(
            &data[..view.true_count()],
            &(0..512).collect::<Vec<u32>>()[..]
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_sparse() {
        const NB: usize = 256; // 2048 bits
        const N: usize = NB * 8;

        let mut bits = [0u8; NB];
        // Set every 16th bit
        for i in (0..N).step_by(16) {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bits[byte_idx] |= 1 << bit_idx;
        }
        let view = BitView::<NB>::new(&bits);

        let mut data: Vec<u64> = (0..N).map(|i| i as u64).collect();
        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(
            &data[..view.true_count()],
            &(0..N as u64).step_by(16).collect::<Vec<u64>>()[..]
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_all_true() {
        const NB: usize = 64; // 512 bits
        const N: usize = NB * 8;

        let view = BitView::<NB>::all_true();
        let mut data: Vec<u16> = (0..N).map(|i| i as u16).collect();
        let original = data.clone();

        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(&data[..view.true_count()], &original[..]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_in_place_all_false() {
        const NB: usize = 64; // 512 bits
        const N: usize = NB * 8;

        let view = BitView::<NB>::all_false();
        let mut data: Vec<u8> = (0..N).map(|i| i as u8).collect();

        unsafe { filter_neon(&mut data, &view) };
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    fn test_filter_slice_wrapper() {
        const NB: usize = 128;
        const N: usize = NB * 8;

        let view = BitView::<NB>::with_prefix(256);
        let mut data: Vec<u32> = (0..N).map(|i| i as u32).collect();

        unsafe { filter_neon(&mut data, &view) };

        assert_eq!(&data[..256], &(0..256).collect::<Vec<u32>>()[..]);
    }
}
