// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512 `vpcompress` and AVX2 `vpermd` compress kernels.
//!
//! See the [module docs](super) for how these fit the shared dispatch.

use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_maskload_epi32;
use std::arch::x86_64::_mm256_maskload_epi64;
use std::arch::x86_64::_mm256_maskstore_epi32;
use std::arch::x86_64::_mm256_maskstore_epi64;
use std::arch::x86_64::_mm256_permutevar8x32_epi32;
use std::arch::x86_64::_mm256_storeu_si256;
use std::arch::x86_64::_mm512_loadu_epi8;
use std::arch::x86_64::_mm512_loadu_epi16;
use std::arch::x86_64::_mm512_loadu_epi32;
use std::arch::x86_64::_mm512_loadu_epi64;
use std::arch::x86_64::_mm512_mask_storeu_epi8;
use std::arch::x86_64::_mm512_mask_storeu_epi16;
use std::arch::x86_64::_mm512_mask_storeu_epi32;
use std::arch::x86_64::_mm512_mask_storeu_epi64;
use std::arch::x86_64::_mm512_maskz_compress_epi8;
use std::arch::x86_64::_mm512_maskz_compress_epi16;
use std::arch::x86_64::_mm512_maskz_compress_epi32;
use std::arch::x86_64::_mm512_maskz_compress_epi64;
use std::arch::x86_64::_mm512_maskz_loadu_epi8;
use std::arch::x86_64::_mm512_maskz_loadu_epi16;
use std::arch::x86_64::_mm512_maskz_loadu_epi32;
use std::arch::x86_64::_mm512_maskz_loadu_epi64;
use std::arch::x86_64::_mm512_storeu_epi8;
use std::arch::x86_64::_mm512_storeu_epi16;
use std::arch::x86_64::_mm512_storeu_epi32;
use std::arch::x86_64::_mm512_storeu_epi64;

use vortex_mask::MaskValues;

use super::super::slice::for_each_mask_word;
use super::super::slice::low_bits_mask;
use super::Kernel;
use super::bulk_copy;

/// Choose the widest kernel the CPU supports for this element width, and the minimum mask
/// density at which it beats the scalar strategies.
///
/// Below that density the scalar set-bit walk wins, because it steps from one set bit to the
/// next instead of paying a compress and a full-width store for lanes it discards. How long that
/// takes to amortize is set by how many lanes one compress covers, so the floor falls as the
/// kernel gets wider: `vpcompressb` consumes a whole 64-bit mask word at once and never loses,
/// while AVX2's 4-lane `vpermd` over 8-byte elements needs a nearly half-set mask to pay off.
///
/// Measured with `benches/filter_fixed_width.rs` on Zen 5 (EPYC 9R05, AVX-512 VBMI2), at `LEN`
/// 16384, interleaved against the same binary with the kernels disabled. The AVX2 rows come from
/// the same machine with AVX-512 detection forced off:
///
/// | kernel | lanes per compress | element | floor | at floor | best in band |
/// | --- | --- | --- | --- | --- | --- |
/// | `vpcompressb` | 64 | 1 byte | none | 1.9x at 0.05 | 5.5x at 0.8 |
/// | `vpcompressw` | 32 | 2 bytes | 0.15 | 1.1x | 3.8x at 0.8 |
/// | `vpcompressd` | 16 | 4 bytes | 0.25 | 1.1x | 2.8x at 0.8 |
/// | `vpcompressq` | 8 | 8 bytes | 0.30 | 1.1x | 3.2x at 0.65 |
/// | `vpermd` | 8 | 4 bytes | 0.25 | 1.1x | 2.5x at 0.8 |
/// | `vpermd` | 4 | 8 bytes | 0.45 | 1.4x | 2.0x at 0.65 |
///
/// `is_x86_feature_detected!` caches per feature, so per-buffer selection stays cheap.
pub(super) fn select_kernel<T, const IN_PLACE: bool>(mask: &MaskValues) -> Option<Kernel> {
    let (kernel, min_density) = match size_of::<T>() {
        1 if avx512_vbmi2() => (compress_avx512_epi8::<IN_PLACE> as Kernel, 0.0),
        2 if avx512_vbmi2() => (compress_avx512_epi16::<IN_PLACE> as Kernel, 0.15),
        4 if avx512f() => (compress_avx512_epi32::<IN_PLACE> as Kernel, 0.25),
        8 if avx512f() => (compress_avx512_epi64::<IN_PLACE> as Kernel, 0.30),
        4 if avx2() => (compress_avx2_epi32::<IN_PLACE> as Kernel, 0.25),
        8 if avx2() => (compress_avx2_epi64::<IN_PLACE> as Kernel, 0.45),
        _ => return None,
    };

    (mask.density() >= min_density).then_some(kernel)
}

fn avx512f() -> bool {
    is_x86_feature_detected!("avx512f")
}

fn avx512_vbmi2() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi2")
}

fn avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Generate an AVX-512 `vpcompress` kernel for one element width: a per-word compress
/// (`$word_fn`) plus the mask-word walk (`$walk_fn`) that drives it. Each mask subword of
/// `$lanes` bits is used directly as the `k`-register of one compress.
macro_rules! avx512_compress_kernel {
    (
        $word_fn:ident,
        $walk_fn:ident,features:
        $features:literal,elem:
        $elem:ty,lanes:
        $lanes:literal,kmask:
        $kmask:ty,loadu:
        $loadu:ident,maskz_loadu:
        $maskz_loadu:ident,maskz_compress:
        $maskz_compress:ident,storeu:
        $storeu:ident,mask_storeu:
        $mask_storeu:ident
    ) => {
        /// Compress the elements selected by one mask word.
        ///
        /// # Safety
        ///
        /// The CPU must support the enabled target features and the pointer contract of
        /// [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
        /// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
        // `allow` rather than `expect`: the cast is only lossy for sub-`u64` k-masks.
        #[allow(clippy::cast_possible_truncation)]
        #[target_feature(enable = $features)]
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
                    bulk_copy::<IN_PLACE>(
                        src,
                        dst,
                        word_start,
                        word_len,
                        write_pos,
                        size_of::<$elem>(),
                    )
                };
                return write_pos + word_len;
            }

            let mut sub = 0;
            while sub < word_len {
                let count = (word_len - sub).min($lanes);
                let k = ((word >> sub) & low_bits_mask(count)) as $kmask;
                if k != 0 {
                    let src_ptr = unsafe { src.cast::<$elem>().add(word_start + sub) };
                    let chunk = if count == $lanes {
                        // SAFETY: `$lanes` source elements starting at
                        // `word_start + sub` are in bounds.
                        unsafe { $loadu(src_ptr) }
                    } else {
                        // SAFETY: the load mask enables only the `count` in-bounds lanes.
                        unsafe { $maskz_loadu(low_bits_mask(count) as $kmask, src_ptr) }
                    };
                    let packed = $maskz_compress(k, chunk);
                    let selected = k.count_ones() as usize;
                    let dst_ptr = unsafe { dst.cast::<$elem>().add(write_pos) };
                    if !IN_PLACE || count == $lanes {
                        // SAFETY: out-of-place output has a vector of slack past
                        // `true_count`; in-place, `write_pos <= word_start + sub`, so
                        // `write_pos + $lanes <= word_start + sub + $lanes <= len`.
                        unsafe { $storeu(dst_ptr, packed) };
                    } else {
                        // SAFETY: stores exactly the `selected` selected lanes, all
                        // within the `true_count` output positions.
                        unsafe { $mask_storeu(dst_ptr, low_bits_mask(selected) as $kmask, packed) };
                    }
                    write_pos += selected;
                }
                sub += $lanes;
            }
            write_pos
        }

        /// Walk the mask words of `mask`, compressing the selected elements.
        ///
        /// # Safety
        ///
        /// The CPU must support the enabled target features and the pointer contract of
        /// [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
        /// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
        #[target_feature(enable = $features)]
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

avx512_compress_kernel!(
    compress_word_avx512_epi8, compress_avx512_epi8,
    features: "avx512f,avx512bw,avx512vbmi2",
    elem: i8,
    lanes: 64,
    kmask: u64,
    loadu: _mm512_loadu_epi8,
    maskz_loadu: _mm512_maskz_loadu_epi8,
    maskz_compress: _mm512_maskz_compress_epi8,
    storeu: _mm512_storeu_epi8,
    mask_storeu: _mm512_mask_storeu_epi8
);

avx512_compress_kernel!(
    compress_word_avx512_epi16, compress_avx512_epi16,
    features: "avx512f,avx512bw,avx512vbmi2",
    elem: i16,
    lanes: 32,
    kmask: u32,
    loadu: _mm512_loadu_epi16,
    maskz_loadu: _mm512_maskz_loadu_epi16,
    maskz_compress: _mm512_maskz_compress_epi16,
    storeu: _mm512_storeu_epi16,
    mask_storeu: _mm512_mask_storeu_epi16
);

avx512_compress_kernel!(
    compress_word_avx512_epi32, compress_avx512_epi32,
    features: "avx512f",
    elem: i32,
    lanes: 16,
    kmask: u16,
    loadu: _mm512_loadu_epi32,
    maskz_loadu: _mm512_maskz_loadu_epi32,
    maskz_compress: _mm512_maskz_compress_epi32,
    storeu: _mm512_storeu_epi32,
    mask_storeu: _mm512_mask_storeu_epi32
);

avx512_compress_kernel!(
    compress_word_avx512_epi64, compress_avx512_epi64,
    features: "avx512f",
    elem: i64,
    lanes: 8,
    kmask: u8,
    loadu: _mm512_loadu_epi64,
    maskz_loadu: _mm512_maskz_loadu_epi64,
    maskz_compress: _mm512_maskz_compress_epi64,
    storeu: _mm512_storeu_epi64,
    mask_storeu: _mm512_mask_storeu_epi64
);

/// For each mask byte, `vpermd` lane indices compacting the selected 4-byte lanes to the front
/// (trailing lanes are don't-care).
static PERM_LUT_32: [[u32; 8]; 256] = {
    let mut lut = [[0u32; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let mut out_lane = 0;
        let mut bit = 0;
        while bit < 8 {
            if m & (1 << bit) != 0 {
                lut[m][out_lane] = bit as u32;
                out_lane += 1;
            }
            bit += 1;
        }
        m += 1;
    }
    lut
};

/// For each mask nibble, `vpermd` lane indices compacting the selected 8-byte lanes (as pairs
/// of 4-byte lanes) to the front.
static PERM_LUT_64: [[u32; 8]; 16] = {
    let mut lut = [[0u32; 8]; 16];
    let mut m = 0;
    while m < 16 {
        let mut out_lane = 0;
        let mut bit = 0;
        while bit < 4 {
            if m & (1 << bit) != 0 {
                lut[m][out_lane * 2] = (bit * 2) as u32;
                lut[m][out_lane * 2 + 1] = (bit * 2 + 1) as u32;
                out_lane += 1;
            }
            bit += 1;
        }
        m += 1;
    }
    lut
};

/// Lane-enable vectors for `vpmaskmov` loads/stores of the first `count` 4-byte lanes.
static LANE_MASK_32: [[i32; 8]; 9] = {
    let mut lut = [[0i32; 8]; 9];
    let mut count = 0;
    while count <= 8 {
        let mut lane = 0;
        while lane < count {
            lut[count][lane] = -1;
            lane += 1;
        }
        count += 1;
    }
    lut
};

/// Lane-enable vectors for `vpmaskmov` loads/stores of the first `count` 8-byte lanes.
static LANE_MASK_64: [[i64; 4]; 5] = {
    let mut lut = [[0i64; 4]; 5];
    let mut count = 0;
    while count <= 4 {
        let mut lane = 0;
        while lane < count {
            lut[count][lane] = -1;
            lane += 1;
        }
        count += 1;
    }
    lut
};

/// Generate an AVX2 permutation-LUT kernel for one element width, mirroring
/// [`avx512_compress_kernel`] with `vpermd` compaction per mask byte (or nibble).
macro_rules! avx2_compress_kernel {
    (
        $word_fn:ident,
        $walk_fn:ident,elem:
        $elem:ty,lanes:
        $lanes:literal,perm_lut:
        $perm_lut:ident,lane_masks:
        $lane_masks:ident,maskload:
        $maskload:ident,maskstore:
        $maskstore:ident
    ) => {
        /// Compress the elements selected by one mask word.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX2 and the pointer contract of
        /// [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
        /// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "deliberate submask narrowing"
        )]
        #[target_feature(enable = "avx2")]
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
                    bulk_copy::<IN_PLACE>(
                        src,
                        dst,
                        word_start,
                        word_len,
                        write_pos,
                        size_of::<$elem>(),
                    )
                };
                return write_pos + word_len;
            }

            let mut sub = 0;
            while sub < word_len {
                let count = (word_len - sub).min($lanes);
                let m = ((word >> sub) & low_bits_mask(count)) as usize;
                if m != 0 {
                    let src_ptr = unsafe { src.cast::<$elem>().add(word_start + sub) };
                    let chunk = if count == $lanes {
                        // SAFETY: `$lanes` source elements starting at
                        // `word_start + sub` are in bounds.
                        unsafe { _mm256_loadu_si256(src_ptr.cast()) }
                    } else {
                        // SAFETY: the lane mask enables only the `count` in-bounds lanes.
                        unsafe {
                            $maskload(
                                src_ptr,
                                _mm256_loadu_si256($lane_masks[count].as_ptr().cast()),
                            )
                        }
                    };
                    // SAFETY: LUT rows are 32 bytes.
                    let perm = unsafe { _mm256_loadu_si256($perm_lut[m].as_ptr().cast()) };
                    let packed = _mm256_permutevar8x32_epi32(chunk, perm);
                    let selected = m.count_ones() as usize;
                    let dst_ptr = unsafe { dst.cast::<$elem>().add(write_pos) };
                    if !IN_PLACE || count == $lanes {
                        // SAFETY: out-of-place output has a vector of slack past
                        // `true_count`; in-place, `write_pos <= word_start + sub`, so
                        // `write_pos + $lanes <= word_start + sub + $lanes <= len`.
                        unsafe { _mm256_storeu_si256(dst_ptr.cast(), packed) };
                    } else {
                        // SAFETY: stores exactly the `selected` selected lanes, all
                        // within the `true_count` output positions.
                        unsafe {
                            $maskstore(
                                dst_ptr,
                                _mm256_loadu_si256($lane_masks[selected].as_ptr().cast()),
                                packed,
                            )
                        };
                    }
                    write_pos += selected;
                }
                sub += $lanes;
            }
            write_pos
        }

        /// Walk the mask words of `mask`, compressing the selected elements.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX2 and the pointer contract of
        /// [`filter_slice_by_bitmap`](super::filter_slice_by_bitmap) /
        /// [`filter_slice_mut_by_bitmap`](super::filter_slice_mut_by_bitmap) must hold.
        #[target_feature(enable = "avx2")]
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

avx2_compress_kernel!(
    compress_word_avx2_epi32, compress_avx2_epi32,
    elem: i32,
    lanes: 8,
    perm_lut: PERM_LUT_32,
    lane_masks: LANE_MASK_32,
    maskload: _mm256_maskload_epi32,
    maskstore: _mm256_maskstore_epi32
);

avx2_compress_kernel!(
    compress_word_avx2_epi64, compress_avx2_epi64,
    elem: i64,
    lanes: 4,
    perm_lut: PERM_LUT_64,
    lane_masks: LANE_MASK_64,
    maskload: _mm256_maskload_epi64,
    maskstore: _mm256_maskstore_epi64
);
