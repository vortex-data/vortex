// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BMI2 PDEP deposit and the kernels compiled with BMI2 enabled.

use std::arch::x86_64::_pdep_u64;

use vortex_buffer::BitBuffer;

use super::DepositBits;
use super::SelectBit;
use super::intersect_bit_buffer_by_rank_indices;
use super::intersect_bit_buffers;
use super::intersect_mask_driven;
use crate::Mask;

pub(super) struct Bmi2;

impl DepositBits for Bmi2 {
    #[inline]
    fn deposit_bits(source: u64, mask: u64, _mask_count: usize) -> u64 {
        // SAFETY: callers only instantiate this implementation after checking BMI2 support.
        unsafe { pdep_bmi2(source, mask) }
    }
}

impl SelectBit for Bmi2 {
    #[inline]
    fn select_bit_position(word: u64, rank: usize) -> usize {
        debug_assert!(rank < word.count_ones() as usize);
        // PDEP places the rank-th bit of source into the rank-th set bit of mask, returning a
        // single bit at the desired position.
        // SAFETY: callers only instantiate this implementation after checking BMI2 support.
        unsafe { pdep_bmi2(1u64 << rank, word) }.trailing_zeros() as usize
    }
}

#[inline]
#[target_feature(enable = "bmi2")]
unsafe fn pdep_bmi2(source: u64, mask: u64) -> u64 {
    _pdep_u64(source, mask)
}

/// The kernels with BMI2 enabled for the whole loop, so that PDEP inlines into it. A generic
/// kernel without the feature calls an out-of-line `pdep_bmi2` once per word.
///
/// # Safety
/// Requires BMI2.
#[target_feature(enable = "bmi2")]
pub(super) unsafe fn intersect_bit_buffers_bmi2(
    self_buffer: &BitBuffer,
    mask_buffer: &BitBuffer,
    true_count: usize,
) -> Mask {
    intersect_bit_buffers::<Bmi2>(self_buffer, mask_buffer, true_count)
}

/// See [`intersect_bit_buffers_bmi2`].
///
/// # Safety
/// Requires BMI2.
#[target_feature(enable = "bmi2")]
pub(super) unsafe fn intersect_bit_buffer_by_rank_indices_bmi2(
    self_buffer: &BitBuffer,
    mask_indices: &[usize],
) -> Mask {
    intersect_bit_buffer_by_rank_indices::<Bmi2>(self_buffer, mask_indices)
}

/// See [`intersect_bit_buffers_bmi2`].
///
/// # Safety
/// Requires BMI2.
#[target_feature(enable = "bmi2")]
pub(super) unsafe fn intersect_mask_driven_bmi2<I>(
    self_buffer: &BitBuffer,
    mask_indices: I,
    true_count: usize,
) -> Mask
where
    I: Iterator<Item = usize>,
{
    intersect_mask_driven::<Bmi2, _>(self_buffer, mask_indices, true_count)
}
