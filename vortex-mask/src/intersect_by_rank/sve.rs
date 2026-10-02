// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SVE2 BDEP deposit and the kernels compiled with SVE2 enabled.

use vortex_buffer::BitBuffer;

use super::DepositBits;
use super::SelectBit;
use super::intersect_bit_buffer_by_rank_indices;
use super::intersect_bit_buffers;
use super::intersect_mask_driven;
use crate::Mask;

pub(super) struct Sve2Bdep;

impl DepositBits for Sve2Bdep {
    #[inline]
    fn deposit_bits(source: u64, mask: u64, _mask_count: usize) -> u64 {
        // SAFETY: callers only instantiate this implementation after checking SVE2 BITPERM support.
        unsafe { bdep_sve2(source, mask) }
    }
}

impl SelectBit for Sve2Bdep {
    #[inline]
    fn select_bit_position(word: u64, rank: usize) -> usize {
        debug_assert!(rank < word.count_ones() as usize);
        // SAFETY: callers only instantiate this implementation after checking SVE2 BITPERM support.
        unsafe { bdep_sve2(1u64 << rank, word) }.trailing_zeros() as usize
    }
}

/// SVE2 BDEP on the lowest lane, one chunk at a time as the rank reader yields them.
#[target_feature(enable = "sve2,sve2-bitperm")]
unsafe fn bdep_sve2(source: u64, mask: u64) -> u64 {
    let result: u64;
    // SAFETY: registers only, and the caller checked SVE2 BITPERM support.
    unsafe {
        std::arch::asm!(
            "fmov d0, {source}",
            "fmov d1, {mask}",
            "bdep z0.d, z0.d, z1.d",
            "fmov {result}, d0",
            source = in(reg) source,
            mask = in(reg) mask,
            result = lateout(reg) result,
            out("v0") _,
            out("v1") _,
            options(pure, nomem, nostack),
        );
    }
    result
}

/// The kernels with SVE2 enabled for the whole loop, so that BDEP inlines into it.
///
/// # Safety
/// Requires SVE2 and SVE2 BITPERM.
#[target_feature(enable = "sve2,sve2-bitperm")]
pub(super) unsafe fn intersect_bit_buffers_sve2(
    self_buffer: &BitBuffer,
    mask_buffer: &BitBuffer,
    true_count: usize,
) -> Mask {
    intersect_bit_buffers::<Sve2Bdep>(self_buffer, mask_buffer, true_count)
}

/// See [`intersect_bit_buffers_sve2`].
///
/// # Safety
/// Requires SVE2 and SVE2 BITPERM.
#[target_feature(enable = "sve2,sve2-bitperm")]
pub(super) unsafe fn intersect_bit_buffer_by_rank_indices_sve2(
    self_buffer: &BitBuffer,
    mask_indices: &[usize],
) -> Mask {
    intersect_bit_buffer_by_rank_indices::<Sve2Bdep>(self_buffer, mask_indices)
}

/// See [`intersect_bit_buffers_sve2`].
///
/// # Safety
/// Requires SVE2 and SVE2 BITPERM.
#[target_feature(enable = "sve2,sve2-bitperm")]
pub(super) unsafe fn intersect_mask_driven_sve2<I>(
    self_buffer: &BitBuffer,
    mask_indices: I,
    true_count: usize,
) -> Mask
where
    I: Iterator<Item = usize>,
{
    intersect_mask_driven::<Sve2Bdep, _>(self_buffer, mask_indices, true_count)
}
