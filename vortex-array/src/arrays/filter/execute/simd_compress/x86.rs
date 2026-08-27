// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512 `vpcompress`, AVX2 `vpermd`, and 128-bit `pshufb` compress kernels.
//!
//! See the [module docs](super) for how these fit the shared dispatch.

use std::arch::x86_64::__m128i;
use std::arch::x86_64::_mm_loadl_epi64;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm_shuffle_epi8;
use std::arch::x86_64::_mm_storel_epi64;
use std::arch::x86_64::_mm_storeu_si128;
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
use super::compress_lut;
use super::compress_tail;

/// Choose the widest available kernel above its benchmarked density crossover.
///
/// Sparse masks stay on the scalar set-bit walk. See `benches/filter_fixed_width.rs` when
/// changing these thresholds.
pub(super) fn select_kernel<T, const IN_PLACE: bool>(mask: &MaskValues) -> Option<Kernel> {
    let (kernel, min_density) = match size_of::<T>() {
        1 if avx512_vbmi2() => (compress_avx512_epi8::<IN_PLACE> as Kernel, 0.0),
        2 if avx512_vbmi2() => (compress_avx512_epi16::<IN_PLACE> as Kernel, 0.15),
        4 if avx512f() => (compress_avx512_epi32::<IN_PLACE> as Kernel, 0.25),
        8 if avx512f() => (compress_avx512_epi64::<IN_PLACE> as Kernel, 0.30),
        // AVX-512F without VBMI2 (e.g. Skylake-X) falls through to these too.
        1 if avx2() => (compress_pshufb_epi8::<IN_PLACE> as Kernel, 0.15),
        2 if avx2() => (compress_pshufb_epi16::<IN_PLACE> as Kernel, 0.25),
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

/// Generate an AVX-512 kernel for one element width.
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

/// Generate an AVX2 permutation-LUT kernel for one element width.
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

/// Byte-index rows for `pshufb`, which always indexes a full 16-byte register even though only
/// the low 8 (1-byte elements) or all 16 (2-byte elements) bytes hold lanes.
static SHUF_LUT_8: [[u8; 16]; 256] = compress_lut::<256, 16>(8, 1);
static SHUF_LUT_16: [[u8; 16]; 256] = compress_lut::<256, 16>(8, 2);

/// Generate an AVX2 `pshufb` kernel for 1- or 2-byte elements.
macro_rules! pshufb_compress_kernel {
    (
        $word_fn:ident,
        $walk_fn:ident,elem_size:
        $elem_size:literal,idx_lut:
        $idx_lut:ident,load:
        $load:ident,store:
        $store:ident
    ) => {
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
                    bulk_copy::<IN_PLACE>(src, dst, word_start, word_len, write_pos, $elem_size)
                };
                return write_pos + word_len;
            }

            // Empty chunks still store garbage that the next chunk overwrites; branching here
            // regresses masks near the density crossover.
            let mut sub = 0;
            while sub + 8 <= word_len {
                let m = ((word >> sub) & low_bits_mask(8)) as usize;
                // SAFETY: the chunk holds 8 in-bounds source elements.
                let chunk = unsafe { $load(src.add((word_start + sub) * $elem_size).cast()) };
                // SAFETY: every LUT row is 16 bytes.
                let idx = unsafe { _mm_loadu_si128($idx_lut[m].as_ptr().cast()) };
                // SAFETY: out-of-place output has vector slack. In-place, the store ends within
                // the source chunk already loaded, and later stores overwrite trailing garbage.
                unsafe {
                    $store(
                        dst.add(write_pos * $elem_size).cast::<__m128i>(),
                        _mm_shuffle_epi8(chunk, idx),
                    )
                };
                write_pos += m.count_ones() as usize;
                sub += 8;
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

pshufb_compress_kernel!(
    compress_word_pshufb_epi8, compress_pshufb_epi8,
    elem_size: 1,
    idx_lut: SHUF_LUT_8,
    load: _mm_loadl_epi64,
    store: _mm_storel_epi64
);

pshufb_compress_kernel!(
    compress_word_pshufb_epi16, compress_pshufb_epi16,
    elem_size: 2,
    idx_lut: SHUF_LUT_16,
    load: _mm_loadu_si128,
    store: _mm_storeu_si128
);
