// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NEON `tbl` compress kernels.
//!
//! Byte-index lookup tables drive `tbl` shuffles for 1-, 2-, 4-, and 8-byte elements.

use core::arch::aarch64::vld1_u8;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vqtbl1q_u8;
use core::arch::aarch64::vst1_u8;
use core::arch::aarch64::vst1q_u8;
use core::arch::aarch64::vtbl1_u8;

use vortex_mask::MaskValues;

use super::super::slice::for_each_mask_word;
use super::super::slice::low_bits_mask;
use super::Kernel;
use super::bulk_copy;
use super::compress_lut;
use super::compress_tail;

/// Return the benchmarked density range where NEON beats the scalar strategies.
fn density_band<T>() -> Option<std::ops::Range<f64>> {
    match size_of::<T>() {
        1 | 2 => Some(0.15..f64::INFINITY),
        4 => Some(0.30..f64::INFINITY),
        8 => Some(0.50..0.80),
        _ => None,
    }
}

/// Choose the kernel for this element width, if one applies to this mask.
pub(super) fn select_kernel<T, const IN_PLACE: bool>(mask: &MaskValues) -> Option<Kernel> {
    if !density_band::<T>()?.contains(&mask.density()) {
        return None;
    }

    match size_of::<T>() {
        1 => Some(compress_neon_8::<IN_PLACE> as Kernel),
        2 => Some(compress_neon_16::<IN_PLACE> as Kernel),
        4 => Some(compress_neon_32::<IN_PLACE> as Kernel),
        8 => Some(compress_neon_64::<IN_PLACE> as Kernel),
        _ => None,
    }
}

static IDX_LUT_8: [[u8; 8]; 256] = compress_lut::<256, 8>(8, 1);
static IDX_LUT_16: [[u8; 16]; 256] = compress_lut::<256, 16>(8, 2);
static IDX_LUT_32: [[u8; 16]; 16] = compress_lut::<16, 16>(4, 4);
static IDX_LUT_64: [[u8; 16]; 4] = compress_lut::<4, 16>(2, 8);

/// Generate a NEON `tbl` kernel for one element width.
macro_rules! neon_compress_kernel {
    (
        $word_fn:ident,
        $walk_fn:ident,elem_size:
        $elem_size:literal,lanes:
        $lanes:literal,idx_lut:
        $idx_lut:ident,load:
        $load:ident,tbl:
        $tbl:ident,store:
        $store:ident
    ) => {
        /// # Safety
        ///
        /// The pointer contract of [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
        /// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deliberate submask narrowing"
        )]
        #[inline]
        unsafe fn $word_fn<const IN_PLACE: bool>(
            src: *const u8,
            dst: *mut u8,
            word: u64,
            word_start: usize,
            word_len: usize,
            mut write_pos: usize,
        ) -> usize {
            if word == 0 {
                return write_pos;
            }
            if word == low_bits_mask(word_len) {
                // SAFETY: forwarded from the caller contract.
                unsafe {
                    bulk_copy::<IN_PLACE>(src, dst, word_start, word_len, write_pos, $elem_size)
                };
                return write_pos + word_len;
            }

            // Empty chunks still store garbage that the next chunk overwrites; branching here
            // regresses masks near the density crossover.
            let mut sub = 0;
            while sub + $lanes <= word_len {
                let m = ((word >> sub) & low_bits_mask($lanes)) as usize;
                // SAFETY: the chunk holds `$lanes` in-bounds source elements, which is exactly
                // the table width.
                let chunk = unsafe { $load(src.add((word_start + sub) * $elem_size)) };
                // SAFETY: every LUT row is one table wide.
                let idx = unsafe { $load($idx_lut[m].as_ptr()) };
                // SAFETY: out-of-place output has vector slack. In-place, the store ends within
                // the source chunk already loaded, and later stores overwrite trailing garbage.
                unsafe { $store(dst.add(write_pos * $elem_size), $tbl(chunk, idx)) };
                write_pos += m.count_ones() as usize;
                sub += $lanes;
            }

            if sub < word_len {
                let bits = (word >> sub) & low_bits_mask(word_len - sub);
                // SAFETY: forwarded from the caller contract.
                write_pos = unsafe {
                    compress_tail::<IN_PLACE>(
                        src,
                        dst,
                        bits,
                        word_start + sub,
                        write_pos,
                        $elem_size,
                    )
                };
            }

            write_pos
        }

        /// # Safety
        ///
        /// The pointer contract of [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
        /// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
        pub(super) unsafe fn $walk_fn<const IN_PLACE: bool>(
            src: *const u8,
            dst: *mut u8,
            mask: &MaskValues,
        ) -> usize {
            let mut write_pos = 0;
            for_each_mask_word(mask, |word, word_start, word_len| {
                // SAFETY: forwarded from the caller contract.
                write_pos = unsafe {
                    $word_fn::<IN_PLACE>(src, dst, word, word_start, word_len, write_pos)
                };
            });
            write_pos
        }
    };
}

neon_compress_kernel!(
    compress_word_neon_8, compress_neon_8,
    elem_size: 1,
    lanes: 8,
    idx_lut: IDX_LUT_8,
    load: vld1_u8,
    tbl: vtbl1_u8,
    store: vst1_u8
);

neon_compress_kernel!(
    compress_word_neon_16, compress_neon_16,
    elem_size: 2,
    lanes: 8,
    idx_lut: IDX_LUT_16,
    load: vld1q_u8,
    tbl: vqtbl1q_u8,
    store: vst1q_u8
);

neon_compress_kernel!(
    compress_word_neon_32, compress_neon_32,
    elem_size: 4,
    lanes: 4,
    idx_lut: IDX_LUT_32,
    load: vld1q_u8,
    tbl: vqtbl1q_u8,
    store: vst1q_u8
);

neon_compress_kernel!(
    compress_word_neon_64, compress_neon_64,
    elem_size: 8,
    lanes: 2,
    idx_lut: IDX_LUT_64,
    load: vld1q_u8,
    tbl: vqtbl1q_u8,
    store: vst1q_u8
);
