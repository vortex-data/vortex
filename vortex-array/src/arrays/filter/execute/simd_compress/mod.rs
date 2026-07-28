// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD compress kernels for fixed-width filtering.
//!
//! Every architecture reduces the same problem — write the elements selected by a mask word to
//! the front of the output — to a per-(sub)word gather, then walks the mask with
//! [`for_each_mask_word`](super::slice::for_each_mask_word). What differs is how one chunk of
//! lanes is compacted:
//!
//! - AVX-512: `vpcompress[b/w/d/q]` consumes each mask (sub)word directly as the `k`-register
//!   (AVX-512F for 4/8-byte elements, additionally VBMI2 for 1/2-byte elements).
//! - AVX2, 4/8-byte elements: a per-mask-byte (or nibble) lane-index LUT feeding `vpermd`, a
//!   vectorized version of the scalar `BYTE_COMPRESS_LUT`.
//! - AVX2, 1/2-byte elements: a byte-index LUT feeding 128-bit `pshufb`. `pshufb` only shuffles
//!   within a 128-bit lane, but eight lanes of a 1- or 2-byte element need at most a 16-byte
//!   table, so one lane is all the table has to span.
//! - NEON: the same byte-index LUT construction feeding `tbl`, which unlike `pshufb` spans a
//!   whole register, so it also serves 4- and 8-byte elements.
//!
//! The byte-shuffle kernels (`pshufb`/`tbl`) share [`compress_lut`] and [`compress_tail`]; only
//! the load/shuffle/store intrinsics differ.
//!
//! x86 selects a kernel per buffer by runtime feature detection (`is_x86_feature_detected!`
//! caches per feature, so this stays cheap); NEON is part of the aarch64 baseline and needs no
//! probe. Unlike `vortex_buffer::bit::pack`'s `collect_bool_words_multiversioned`, there is no
//! caller-supplied closure for a `#[target_feature]` boundary to deoptimize — the per-word body
//! is a fixed gather — so a supported element width routes here whenever the mask is dense
//! enough to amortize the per-chunk work.
//!
//! Compressed vectors are written with full-width unmasked stores: the out-of-place output is
//! over-allocated by one vector of slack so trailing garbage lands in spare capacity, which
//! also sidesteps `vpcompressstoreu`'s microcoded slowness on Zen 4. The in-place variant
//! bounds every unmasked store by the source position it has already consumed, and handles
//! partial tail chunks without a full-width store at all.

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

/// Below one full mask word the scalar paths win; skip kernel selection entirely.
const MIN_LEN: usize = 64;

/// One full AVX-512 vector of output slack, so every out-of-place store can be unmasked.
const SLACK_BYTES: usize = 64;

/// A per-buffer compress kernel: `(src, dst, mask) -> written`, with `src`/`dst` pointing at
/// the first element. `dst == src` for the in-place instantiations.
type Kernel = unsafe fn(*const u8, *mut u8, &MaskValues) -> usize;

/// Filter a slice with a SIMD compress kernel, if one applies.
///
/// Returns `None` when no kernel applies (unsupported architecture or element width, missing
/// CPU features, or an input too short or too sparse to amortize the per-chunk work); the
/// caller then falls back to the scalar strategies.
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

/// Choose the widest kernel this build and CPU offer for `T`, if one beats the scalar walk for
/// this mask.
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

/// Build the byte-shuffle index rows for one element width: `lut[m]` gathers the lanes selected
/// by mask `m` into the front of the vector, leaving the trailing bytes as don't-care zeros.
///
/// `BYTES` is the table width, which may exceed the `lanes * elem_size` bytes the mask covers —
/// `pshufb` always indexes a full 16-byte register even when only its low half holds lanes. Both
/// `ROWS == 1 << lanes` and the table fit are checked at compile time by the `static`
/// initializers.
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

/// Compact the elements selected by `bits` (bit `i` selects element `base + i`) one at a time.
///
/// Used for the final partial chunk of a mask word, where a full-width vector load would read
/// past the end of the source. A mask has at most two such chunks — one for an unaligned leading
/// word and one for the trailing word — so this never runs in the hot loop.
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
