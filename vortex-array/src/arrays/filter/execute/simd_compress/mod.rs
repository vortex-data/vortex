// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD compress kernels for fixed-width filtering.
//!
//! Architecture-specific kernels compact selected lanes a mask word at a time. Full-width stores
//! may write trailing garbage, so out-of-place outputs reserve one vector of slack and in-place
//! kernels never store beyond the source chunk already loaded.
//!
//! The table is the complete SIMD eligibility map after cached representations have declined the
//! mask. `d` is mask density; every entry also requires `len >= 64`. A dash means that no SIMD
//! kernel exists for that width.
//!
//! | Target features         | 1 byte       | 2 bytes      | 4 bytes     | 8 bytes            |
//! | ----------------------- | ------------ | ------------ | ----------- | ------------------ |
//! | x86 AVX-512 VBMI2       | `d >= 0.00`  | `d >= 0.15`  | `d >= 0.25` | `d >= 0.30`        |
//! | x86 AVX-512F (no VBMI2) | `d >= 0.15`* | `d >= 0.25`* | `d >= 0.25` | `d >= 0.30`        |
//! | x86 AVX2                | `d >= 0.15`  | `d >= 0.25`  | `d >= 0.25` | `d >= 0.45`        |
//! | aarch64 NEON            | `d >= 0.15`  | `d >= 0.15`  | `d >= 0.30` | `0.50 <= d < 0.80` |
//! | other                   | —            | —            | —           | —                  |
//!
//! \* The 1- and 2-byte fallbacks require AVX2.
//!
//! On x86, the widest available kernel is considered first for each width. When no entry matches,
//! control returns to [`buffer`](super::buffer) for scalar selection.

use std::ptr;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::MaskValues;

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod neon;
#[cfg(test)]
mod tests;
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod x86;

const MIN_LEN: usize = 64;

const SLACK_BYTES: usize = 64;

/// `dst == src` for in-place kernels.
type Kernel = unsafe fn(*const u8, *mut u8, &MaskValues) -> usize;

/// Filter a slice with a SIMD compress kernel, if one applies.
///
/// Returns `None` when the caller should use a scalar strategy.
pub(super) fn filter_slice_by_bitmap<T: Copy>(
    values: &[T],
    mask: &MaskValues,
) -> Option<Buffer<T>> {
    debug_assert_eq!(values.len(), mask.len());
    let kernel = select_kernel::<T, false>(mask)?;

    let true_count = mask.true_count();
    let mut out = BufferMut::<T>::with_capacity(true_count + SLACK_BYTES / size_of::<T>());
    // SAFETY: `select_kernel` probed the kernel's target features; `values` holds `mask.len()`
    // elements and the output has capacity for every selected element plus a full vector of
    // slack, so each unmasked store stays in bounds.
    let written = unsafe {
        kernel(
            values.as_ptr().cast(),
            out.spare_capacity_mut().as_mut_ptr().cast(),
            mask,
        )
    };
    debug_assert_eq!(written, true_count);
    // SAFETY: the kernel initialized the first `true_count` elements.
    unsafe { out.set_len(true_count) };
    Some(out.freeze())
}

/// In-place variant of [`filter_slice_by_bitmap`]: compact the selected elements to the front
/// of `values` and return the new length, if a SIMD kernel applies.
pub(super) fn filter_slice_mut_by_bitmap<T: Copy>(
    values: &mut [T],
    mask: &MaskValues,
) -> Option<usize> {
    debug_assert_eq!(values.len(), mask.len());
    let kernel = select_kernel::<T, true>(mask)?;

    let dst = values.as_mut_ptr().cast::<u8>();
    // SAFETY: `select_kernel` probed the kernel's target features; the in-place instantiation
    // compacts forward (stores never pass the positions it has already read) and keeps partial
    // tail chunks off the full-width store path, so all accesses stay inside `values`.
    let written = unsafe { kernel(dst.cast_const(), dst, mask) };
    debug_assert_eq!(written, mask.true_count());
    Some(written)
}

/// Choose the widest profitable kernel available for `T`.
fn select_kernel<T, const IN_PLACE: bool>(mask: &MaskValues) -> Option<Kernel> {
    if mask.len() < MIN_LEN {
        return None;
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        x86::select_kernel::<T, IN_PLACE>(mask)
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        neon::select_kernel::<T, IN_PLACE>(mask)
    }
    #[cfg(any(not(any(target_arch = "x86_64", target_arch = "aarch64")), miri))]
    {
        let _ = mask;
        None
    }
}

/// Build byte-shuffle rows that gather selected lanes to the front of a vector.
#[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri)))]
#[expect(
    clippy::cast_possible_truncation,
    reason = "byte indices are bounded by the 8- or 16-byte table width"
)]
const fn compress_lut<const ROWS: usize, const BYTES: usize>(
    lanes: usize,
    elem_size: usize,
) -> [[u8; BYTES]; ROWS] {
    assert!(ROWS == 1 << lanes);
    assert!(lanes * elem_size <= BYTES);

    let mut lut = [[0u8; BYTES]; ROWS];
    let mut m = 0;
    while m < ROWS {
        let mut out_lane = 0;
        let mut lane = 0;
        while lane < lanes {
            if m & (1 << lane) != 0 {
                let mut byte = 0;
                while byte < elem_size {
                    lut[m][out_lane * elem_size + byte] = (lane * elem_size + byte) as u8;
                    byte += 1;
                }
                out_lane += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    lut
}

/// Compact a partial vector without reading past the source.
///
/// # Safety
///
/// The pointer contract of [`filter_slice_by_bitmap`] / [`filter_slice_mut_by_bitmap`] must hold,
/// and `bits` must only select elements that are in bounds.
#[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri)))]
#[inline(always)]
unsafe fn compress_tail<const IN_PLACE: bool>(
    src: *const u8,
    dst: *mut u8,
    mut bits: u64,
    base: usize,
    mut write_pos: usize,
    elem_size: usize,
) -> usize {
    while bits != 0 {
        let index = base + bits.trailing_zeros() as usize;
        // SAFETY: `index` is in bounds per the contract above and stable compaction guarantees
        // `write_pos <= index`.
        unsafe { bulk_copy::<IN_PLACE>(src, dst, index, 1, write_pos, elem_size) };
        write_pos += 1;
        bits &= bits - 1;
    }
    write_pos
}

/// Copy `word_len` elements of `elem_size` bytes for a fully-set mask word.
///
/// # Safety
///
/// `word_len` source elements starting at `word_start` must be in bounds. Out-of-place, `dst`
/// must not overlap `src` and must have room at `write_pos`; in-place, forward compaction must
/// guarantee `write_pos <= word_start`.
#[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri)))]
#[inline(always)]
unsafe fn bulk_copy<const IN_PLACE: bool>(
    src: *const u8,
    dst: *mut u8,
    word_start: usize,
    word_len: usize,
    write_pos: usize,
    elem_size: usize,
) {
    // SAFETY: offsets are in bounds per the contract above.
    let src_bytes = unsafe { src.add(word_start * elem_size) };
    // SAFETY: see above.
    let dst_bytes = unsafe { dst.add(write_pos * elem_size) };
    if IN_PLACE {
        if write_pos != word_start {
            // SAFETY: source and destination may overlap while compacting forward.
            unsafe { ptr::copy(src_bytes, dst_bytes, word_len * elem_size) };
        }
    } else {
        // SAFETY: out-of-place buffers are disjoint allocations.
        unsafe { ptr::copy_nonoverlapping(src_bytes, dst_bytes, word_len * elem_size) };
    }
}
