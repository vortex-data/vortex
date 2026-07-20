// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runtime-dispatched SIMD compress kernels for fixed-width filtering.
//!
//! This mirrors the multiversioning pattern of `vortex_buffer::bit::pack`
//! (`collect_bool_words_multiversioned`): a shared per-mask-word walk instantiated behind
//! per-feature-level `#[target_feature]` wrappers, selected once per buffer by runtime feature
//! detection. Unlike `collect_bool`, there is no caller-supplied closure for the
//! `#[target_feature]` boundary to deoptimize — the per-word body is a fixed gather — so every
//! supported element width routes here unconditionally and density thresholds only matter for
//! the scalar fallbacks.
//!
//! Two kernel tiers, dispatched on element width:
//!
//! - AVX-512: `vpcompress[b/w/d/q]` consumes each mask (sub)word directly as the `k`-register
//!   (AVX-512F for 4/8-byte elements, additionally VBMI2 for 1/2-byte elements).
//! - AVX2 (4/8-byte elements only): a per-mask-byte (or nibble) permutation LUT feeding
//!   `vpermd`, a vectorized version of the scalar `BYTE_COMPRESS_LUT`. AVX2 has no cross-lane
//!   byte shuffle, so 1/2-byte elements stay on the scalar byte-LUT below AVX-512.
//!
//! Compressed vectors are written with full-width unmasked stores: the out-of-place output is
//! over-allocated by one vector of slack so trailing garbage lands in spare capacity, which
//! also sidesteps `vpcompressstoreu`'s microcoded slowness on Zen 4. The in-place variant
//! bounds every unmasked store by the source position it has already consumed and masks the
//! stores of partial tail chunks instead.

use vortex_buffer::Buffer;
use vortex_mask::MaskValues;

/// Filter a slice with a runtime-detected SIMD compress kernel, if one applies.
///
/// Returns `None` when no kernel applies (non-x86 target, unsupported element width, missing
/// CPU features, or an input too short to amortize dispatch); the caller then falls back to
/// the scalar strategies.
pub(super) fn filter_slice_by_bitmap<T: Copy>(
    values: &[T],
    mask: &MaskValues,
) -> Option<Buffer<T>> {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        return x86::filter(values, mask);
    }
    #[allow(unreachable_code)]
    {
        let _ = (values, mask);
        None
    }
}

/// In-place variant of [`filter_slice_by_bitmap`]: compact the selected elements to the front
/// of `values` and return the new length, if a SIMD kernel applies.
pub(super) fn filter_slice_mut_by_bitmap<T: Copy>(
    values: &mut [T],
    mask: &MaskValues,
) -> Option<usize> {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        return x86::filter_mut(values, mask);
    }
    #[allow(unreachable_code)]
    {
        let _ = (values, mask);
        None
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod x86 {
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
    use std::ptr;

    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_mask::MaskValues;

    use super::super::slice::for_each_mask_word;
    use super::super::slice::low_bits_mask;

    /// Below one full mask word the scalar paths win; skip feature detection entirely.
    const MIN_LEN: usize = 64;

    /// Below this mask density the scalar set-bit walk beats vector compress for multi-byte
    /// elements: with under ~3 set bits per mask word, the per-subword loop and full-width
    /// stores are pure overhead. Single-byte elements consume a whole mask word per compress
    /// and win at every density. Covered by `benches/filter_fixed_width.rs` (0.01 vs 0.5).
    const MIN_DENSITY_WIDE: f64 = 0.05;

    /// One full vector of output slack so every out-of-place store can be unmasked.
    const SLACK_BYTES: usize = 64;

    /// A per-buffer compress kernel: `(src, dst, mask) -> written`, with `src`/`dst` pointing
    /// at the first element. `dst == src` for the in-place instantiations.
    type Kernel = unsafe fn(*const u8, *mut u8, &MaskValues) -> usize;

    pub(super) fn filter<T: Copy>(values: &[T], mask: &MaskValues) -> Option<Buffer<T>> {
        debug_assert_eq!(values.len(), mask.len());
        if values.len() < MIN_LEN || too_sparse::<T>(mask) {
            return None;
        }
        let kernel = select_kernel::<T, false>()?;

        let true_count = mask.true_count();
        let mut out = BufferMut::<T>::with_capacity(true_count + SLACK_BYTES / size_of::<T>());
        // SAFETY: `select_kernel` probed the kernel's target features; `values` holds
        // `mask.len()` elements and the output has capacity for every selected element plus a
        // full vector of slack, so each unmasked store stays in bounds.
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

    pub(super) fn filter_mut<T: Copy>(values: &mut [T], mask: &MaskValues) -> Option<usize> {
        debug_assert_eq!(values.len(), mask.len());
        if values.len() < MIN_LEN || too_sparse::<T>(mask) {
            return None;
        }
        let kernel = select_kernel::<T, true>()?;

        let dst = values.as_mut_ptr().cast::<u8>();
        // SAFETY: `select_kernel` probed the kernel's target features; the in-place
        // instantiation compacts forward (stores never pass the positions it has already
        // read) and masks partial tail stores, so all accesses stay inside `values`.
        let written = unsafe { kernel(dst.cast_const(), dst, mask) };
        debug_assert_eq!(written, mask.true_count());
        Some(written)
    }

    /// True when the mask is sparse enough that the scalar walk wins for this element width.
    fn too_sparse<T>(mask: &MaskValues) -> bool {
        size_of::<T>() > 1 && mask.density() < MIN_DENSITY_WIDE
    }

    /// Choose the widest kernel the CPU supports for this element width, if any.
    ///
    /// `is_x86_feature_detected!` caches per feature, so per-buffer selection stays cheap.
    fn select_kernel<T, const IN_PLACE: bool>() -> Option<Kernel> {
        match size_of::<T>() {
            1 if avx512_vbmi2() => Some(compress_avx512_epi8::<IN_PLACE> as Kernel),
            2 if avx512_vbmi2() => Some(compress_avx512_epi16::<IN_PLACE> as Kernel),
            4 if avx512f() => Some(compress_avx512_epi32::<IN_PLACE> as Kernel),
            8 if avx512f() => Some(compress_avx512_epi64::<IN_PLACE> as Kernel),
            4 if avx2() => Some(compress_avx2_epi32::<IN_PLACE> as Kernel),
            8 if avx2() => Some(compress_avx2_epi64::<IN_PLACE> as Kernel),
            _ => None,
        }
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

    /// Copy `word_len` elements of `elem_size` bytes for a fully-set mask word.
    ///
    /// # Safety
    ///
    /// `word_len` source elements starting at `word_start` must be in bounds. Out-of-place,
    /// `dst` must not overlap `src` and must have room at `write_pos`; in-place, forward
    /// compaction must guarantee `write_pos <= word_start`.
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
            /// [`filter`]/[`filter_mut`] must hold.
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
                            unsafe {
                                $mask_storeu(dst_ptr, low_bits_mask(selected) as $kmask, packed)
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
            /// The CPU must support the enabled target features and the pointer contract of
            /// [`filter`]/[`filter_mut`] must hold.
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

    /// For each mask byte, `vpermd` lane indices compacting the selected 4-byte lanes to the
    /// front (trailing lanes are don't-care).
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

    /// For each mask nibble, `vpermd` lane indices compacting the selected 8-byte lanes
    /// (as pairs of 4-byte lanes) to the front.
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
            /// The CPU must support AVX2 and the pointer contract of [`filter`]/[`filter_mut`]
            /// must hold.
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
            /// The CPU must support AVX2 and the pointer contract of [`filter`]/[`filter_mut`]
            /// must hold.
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
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_mask::Mask;

    use super::super::slice;
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    use super::x86;
    use super::*;

    /// `None` when the mask normalized to `AllTrue`/`AllFalse`, which `filter_buffer` never
    /// sees (the filter fast paths intercept those before buffer-level dispatch).
    fn mask_values(mask: &Mask) -> Option<&MaskValues> {
        match mask {
            Mask::Values(values) => Some(values.as_ref()),
            _ => None,
        }
    }

    fn make_mask(len: usize, offset: usize, pattern: impl Fn(usize) -> bool) -> Mask {
        let backing =
            BitBuffer::from_iter(std::iter::repeat_n(false, offset).chain((0..len).map(pattern)));
        Mask::from_buffer(BitBuffer::new_with_offset(
            backing.inner().clone(),
            len,
            offset,
        ))
    }

    type Pattern = fn(usize) -> bool;

    fn patterns() -> Vec<(&'static str, Pattern)> {
        vec![
            ("all_but_first", |i| i != 0),
            ("only_first", |i| i == 0),
            ("sparse", |i| i % 97 == 0),
            ("mid", |i| i % 3 == 0),
            ("dense", |i| i % 16 != 0),
            ("alternating", |i| i % 2 == 0),
            ("word_blocks", |i| (i / 64) % 2 == 0),
            ("edges", |i| i < 3 || i % 61 == 60),
        ]
    }

    fn check<T: Copy + PartialEq + std::fmt::Debug>(values: &[T], mask: &Mask) {
        let Some(mask) = mask_values(mask) else {
            return;
        };
        let expected = slice::filter_slice_by_bitmap(values, mask);

        if let Some(actual) = filter_slice_by_bitmap(values, mask) {
            assert_eq!(actual.as_slice(), expected.as_slice());
        }

        let mut compacted = values.to_vec();
        if let Some(new_len) = filter_slice_mut_by_bitmap(&mut compacted, mask) {
            assert_eq!(&compacted[..new_len], expected.as_slice());
        }
    }

    #[rstest]
    fn simd_matches_scalar(
        #[values(0, 3, 5)] offset: usize,
        #[values(64, 100, 151, 1000, 1024)] len: usize,
    ) {
        for (name, pattern) in patterns() {
            let mask = make_mask(len, offset, pattern);
            let mask_debug = format!("pattern={name} len={len} offset={offset}");

            let u8_values: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let u16_values: Vec<u16> = (0..len).map(|i| i as u16).collect();
            let u32_values: Vec<u32> = (0..len).map(|i| i as u32).collect();
            let u64_values: Vec<u64> = (0..len).map(|i| i as u64).collect();

            println!("checking {mask_debug}");
            check(&u8_values, &mask);
            check(&u16_values, &mask);
            check(&u32_values, &mask);
            check(&u64_values, &mask);
        }
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn engages_on_supported_cpus() {
        let values: Vec<u32> = (0..256).collect();
        let mask = make_mask(256, 0, |i| i % 2 == 0);
        let mask = mask_values(&mask).expect("alternating mask is mixed");
        if is_x86_feature_detected!("avx2") {
            assert!(filter_slice_by_bitmap(&values, mask).is_some());
        }
    }

    /// On AVX-512 machines the dispatcher never selects the AVX2 tier for 4/8-byte elements,
    /// so exercise those kernels directly.
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn avx2_kernels_match_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        fn check_kernel<T: Copy + PartialEq + std::fmt::Debug + Default>(
            kernel_out_of_place: unsafe fn(*const u8, *mut u8, &MaskValues) -> usize,
            kernel_in_place: unsafe fn(*const u8, *mut u8, &MaskValues) -> usize,
            values: &[T],
            mask: &MaskValues,
        ) {
            let expected = slice::filter_slice_by_bitmap(values, mask);

            // One vector of slack, mirroring the allocation in `x86::filter`.
            let mut out = vec![T::default(); mask.true_count() + 64 / size_of::<T>()];
            // SAFETY: AVX2 was detected above and the output has a vector of slack.
            let written = unsafe {
                kernel_out_of_place(values.as_ptr().cast(), out.as_mut_ptr().cast(), mask)
            };
            assert_eq!(written, mask.true_count());
            assert_eq!(&out[..written], expected.as_slice());

            let mut compacted = values.to_vec();
            let ptr = compacted.as_mut_ptr().cast::<u8>();
            // SAFETY: AVX2 was detected above; in-place compaction stays within the slice.
            let written = unsafe { kernel_in_place(ptr.cast_const(), ptr, mask) };
            assert_eq!(written, mask.true_count());
            assert_eq!(&compacted[..written], expected.as_slice());
        }

        for (_, pattern) in patterns() {
            for len in [64, 100, 151, 1000] {
                for offset in [0, 5] {
                    let mask = make_mask(len, offset, pattern);
                    let Some(mask) = mask_values(&mask) else {
                        continue;
                    };
                    let u32_values: Vec<u32> = (0..len as u32).collect();
                    let u64_values: Vec<u64> = (0..len as u64).collect();
                    check_kernel(
                        x86::compress_avx2_epi32::<false>,
                        x86::compress_avx2_epi32::<true>,
                        &u32_values,
                        mask,
                    );
                    check_kernel(
                        x86::compress_avx2_epi64::<false>,
                        x86::compress_avx2_epi64::<true>,
                        &u64_values,
                        mask,
                    );
                }
            }
        }
    }
}
