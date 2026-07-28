// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NEON `tbl` compress kernels.
//!
//! `tbl` is an arbitrary byte shuffle over a register-sized table, so one instruction compacts
//! all the lanes of a single vector. Unlike AVX2's `vpermd` it is byte-granular, which means the
//! same construction serves every element width — the LUT just holds byte indices instead of
//! lane indices. What it does not have is AVX-512's `vpcompress`, so the permutation still comes
//! from a table indexed by the mask bits for one vector of lanes:
//!
//! | element | lanes per `tbl` | mask bits | LUT rows | LUT size |
//! | --- | --- | --- | --- | --- |
//! | 1 byte | 8 (`vtbl1_u8`, 64-bit) | 8 | 256 | 2 KiB |
//! | 2 bytes | 8 (`vqtbl1q_u8`) | 8 | 256 | 4 KiB |
//! | 4 bytes | 4 (`vqtbl1q_u8`) | 4 | 16 | 256 B |
//! | 8 bytes | 2 (`vqtbl1q_u8`) | 2 | 4 | 64 B |
//!
//! Every LUT fits comfortably in L1. The lanes-per-shuffle column sets both the payoff and the
//! density band each kernel is worth using over (see [`density_band`]), and is why wide elements
//! gain far less here than under `vpcompressq`: a `tbl` shuffles one register whatever the
//! element width, so the 8-byte kernel retires two elements per shuffle where AVX-512 retires
//! eight, leaving it worth ~1.1-1.6x over a narrow band while the 1/2-byte kernels reach 9x.
//!
//! See the [module docs](super) for how these fit the shared dispatch. NEON is part of the
//! aarch64 baseline, so there is nothing to probe and no `#[target_feature]` boundary to block
//! inlining.

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

/// The mask-density band in which the kernel for `T` beats both scalar strategies.
///
/// These kernels do a fixed amount of work per chunk, so their cost per element is flat in the
/// mask density; the bounds are just where the two scalar strategies cross that flat line.
///
/// Below the lower bound the set-bit walk wins, because it steps from one set bit to the next
/// instead of paying a shuffle and store for lanes it discards. The fewer lanes a chunk holds
/// the longer that takes to amortize, so the bound rises with element width.
///
/// Only 8-byte elements need an upper bound. A `tbl` shuffles one register whatever the element
/// width, so an 8-byte chunk is a mere two lanes and the kernel never gets far ahead; past ~0.8
/// the byte-LUT's all-set bulk copies overtake it. Returning `None` is what lets
/// [`filter_buffer`](super::super::buffer) fall through to them.
///
/// Bounds measured with `benches/filter_fixed_width.rs` on Apple M-series, at `LEN` 16384,
/// interleaved against the same binary with the kernels disabled. Speedups inside each band:
///
/// | element | band | at floor | at 0.5 | at 0.9 | at 0.99 |
/// | --- | --- | --- | --- | --- | --- |
/// | 1 byte | 0.15.. | 1.4x | 5.7x | 10.1x | 2.4x |
/// | 2 bytes | 0.15.. | 1.3x | 5.1x | 9.2x | 2.0x |
/// | 4 bytes | 0.30.. | 1.2x | 2.9x | 5.1x | 1.1x |
/// | 8 bytes | 0.50..0.80 | 1.3x | 1.3x | — | — |
///
/// The bounds sit just above the measured crossovers so a core with a different
/// shuffle-to-scalar balance still cannot regress.
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

/// Generate a NEON `tbl` kernel for one element width: a per-word compress (`$word_fn`) plus
/// the mask-word walk (`$walk_fn`) that drives it. Each chunk of `$lanes` mask bits indexes
/// `$idx_lut` for the byte permutation that compacts one vector of lanes.
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
        /// Compress the elements selected by one mask word.
        ///
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

            // Deliberately branchless: an empty chunk still shuffles and stores, it just
            // leaves `write_pos` where it was so the next chunk overwrites the garbage. A
            // `m != 0` guard here is a large *pessimization*, because the mask bits of one
            // chunk are exactly as random as the data — `P(m == 0) = (1 - density)^lanes`
            // passes 0.5 somewhere inside the useful density range for every width, so the
            // branch mispredicts near-maximally right where it would pay off. Measured on
            // Apple M-series with `benches/filter_fixed_width.rs`, the guarded 4-byte kernel
            // costs 1.95x at density 0.5 and the guarded 1-byte kernel 1.45x at 0.1.
            let mut sub = 0;
            while sub + $lanes <= word_len {
                let m = ((word >> sub) & low_bits_mask($lanes)) as usize;
                // SAFETY: the chunk holds `$lanes` in-bounds source elements, which is exactly
                // the table width.
                let chunk = unsafe { $load(src.add((word_start + sub) * $elem_size)) };
                // SAFETY: every LUT row is one table wide.
                let idx = unsafe { $load($idx_lut[m].as_ptr()) };
                // SAFETY: the store covers `$lanes` elements at `write_pos`, of which the
                // leading `m.count_ones()` are selected values and the rest are garbage that a
                // later store overwrites, since stores resume at the advanced `write_pos`.
                // Out-of-place, the trailing garbage of the final store lands in the vector of
                // slack past `true_count`. In-place, `write_pos <= word_start + sub`, so the
                // store ends at or before `word_start + sub + $lanes <= word_start + word_len
                // <= len`, and it can only overwrite source elements already loaded into
                // `chunk` or output positions not yet final.
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

        /// Walk the mask words of `mask`, compressing the selected elements.
        ///
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
