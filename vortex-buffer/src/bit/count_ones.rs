// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::prelude::*;
use fearless_simd::u8x64;
use fearless_simd::u64x8;
use fearless_simd_macros::simd;
use vortex_error::VortexExpect;

/// Below this many bits, counting inline beats dispatching to a SIMD kernel, provided the
/// build already has a hardware popcount.
const SIMD_MIN_BITS: usize = 256;

/// Whether `count_ones` on primitive integers compiles to a hardware instruction without any
/// runtime dispatch. x86 only gets `popcnt` with SSE4.2-era target features, which the default
/// targets do not enable.
const STATIC_HW_POPCNT: bool =
    !cfg!(any(target_arch = "x86", target_arch = "x86_64")) || cfg!(target_feature = "popcnt");

#[inline]
pub fn count_ones(bytes: &[u8], offset: usize, len: usize) -> usize {
    if bytes.is_empty() {
        return 0;
    }

    // Without static POPCNT, every range is counted inside a dispatched SIMD level, so even
    // short ranges and the partial head and tail bytes use the `popcnt` (SSE4.2 and up) or
    // vector popcount the level enables.
    if STATIC_HW_POPCNT && len < SIMD_MIN_BITS {
        return count_ones_scalar(bytes, offset, len);
    }

    fearless_simd::dispatch!(Level::new(), simd => count_ones_simd(simd, bytes, offset, len))
}

#[inline]
pub(super) fn align_offset_len(
    bytes: &[u8],
    offset: usize,
    len: usize,
) -> (Option<u8>, &[u8], Option<u8>) {
    let start_byte = offset / 8;
    let start_bit = offset % 8;
    let end_bit = offset + len;
    let end_byte = end_bit / 8;
    let head = (start_bit != 0).then(|| {
        let start_len = (8 - start_bit).min(len);
        mask_byte(bytes[start_byte], start_bit, start_len)
    });

    let middle_start = start_byte + usize::from(start_bit != 0);
    let middle_end = end_byte;
    let middle = if middle_start < middle_end {
        &bytes[middle_start..middle_end]
    } else {
        &[]
    };

    let consumed = if start_bit != 0 {
        (8 - start_bit).min(len)
    } else {
        0
    } + middle.len() * 8;
    let tail_len = len - consumed;
    let tail = (tail_len != 0).then(|| mask_byte(bytes[middle_end], 0, tail_len));

    (head, middle, tail)
}

#[inline]
fn mask_byte(byte: u8, bit_offset: usize, bit_len: usize) -> u8 {
    debug_assert!(bit_offset < 8);
    debug_assert!(bit_len <= 8 - bit_offset);

    let shifted = byte >> bit_offset;
    let mask = if bit_len == 8 {
        u8::MAX
    } else {
        (1u8 << bit_len) - 1
    };

    shifted & mask
}

#[inline]
fn count_ones_scalar(bytes: &[u8], offset: usize, len: usize) -> usize {
    let (head, middle, tail) = align_offset_len(bytes, offset, len);
    count_byte(head) + count_ones_aligned_scalar(middle) + count_byte(tail)
}

// Always inlined so the calls in `count_ones_simd` compile with the SIMD level's `popcnt`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn count_byte(byte: Option<u8>) -> usize {
    byte.map_or(0, |v| v.count_ones() as usize)
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn count_ones_aligned_scalar(bytes: &[u8]) -> usize {
    let (words, tail) = bytes.as_chunks::<8>();
    let count = words
        .iter()
        .map(|word| u64::from_le_bytes(*word).count_ones() as usize)
        .sum::<usize>();

    count
        + tail
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum::<usize>()
}

/// Lane-wise `u64` popcount over 64-byte vectors, with the level's target features also
/// applying to the scalar head, tail and remainder. The total is independent of lane byte order.
#[simd]
fn count_ones_simd<S: Simd>(simd: S, bytes: &[u8], offset: usize, len: usize) -> usize {
    let (head, middle, tail) = align_offset_len(bytes, offset, len);
    let (chunks, remainder) = middle.as_chunks::<64>();
    let mut accum = u64x8::splat(simd, 0);
    for chunk in chunks {
        let words: u64x8<S> = u8x64::from_slice(simd, chunk).bitcast();
        accum += words.count_ones();
    }

    count_byte(head)
        + usize::try_from(accum.reduce_sum()).vortex_expect("true_count doesn't fit in usize")
        + count_ones_aligned_scalar(remainder)
        + count_byte(tail)
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use fearless_simd::Level;
    use hegel::TestCase;
    use hegel::generators as gs;
    #[cfg(target_arch = "x86_64")]
    use rstest::rstest;

    #[cfg(target_arch = "x86_64")]
    use super::count_ones_scalar;
    #[cfg(target_arch = "x86_64")]
    use super::count_ones_simd;
    use crate::BitBuffer;

    const MAX_LEN: usize = 4096;

    fn draw_bits(tc: &TestCase, len: usize) -> Vec<bool> {
        match tc.draw(gs::integers::<u8>().max_value(2)) {
            0 => vec![true; len],
            1 => vec![false; len],
            _ => tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len)),
        }
    }

    #[hegel::test]
    #[cfg_attr(miri, ignore)]
    fn true_count(tc: TestCase) {
        let len = tc.draw(gs::integers::<usize>().max_value(MAX_LEN));
        let bits = draw_bits(&tc, len);
        let buf = BitBuffer::from(bits.clone());
        let end = tc.draw(gs::integers::<usize>().max_value(len));
        let start = tc.draw(gs::integers::<usize>().max_value(end));

        let sliced = buf.slice(start..end);
        let expected = bits[start..end].iter().filter(|&&b| b).count();

        assert_eq!(sliced.true_count(), expected);
    }

    /// Every SIMD level the host supports must match the scalar count, not only the one
    /// `Level::new` selects.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[rstest]
    fn test_count_ones_x86_levels_match_scalar(
        #[values(0usize, 3, 8)] offset: usize,
        #[values(0usize, 7, 64, 255, 256, 511, 4097, 65_537)] len: usize,
    ) {
        let buf = BitBuffer::collect_bool(offset + len, |i| (i % 3 == 0) ^ (i % 7 == 0));
        let bytes = buf.inner().as_slice();
        let expected = count_ones_scalar(bytes, offset, len);
        let level = Level::new();

        if let Some(sse2) = level.as_sse2() {
            assert_eq!(count_ones_simd(sse2, bytes, offset, len), expected);
        }
        if let Some(sse4_2) = level.as_sse4_2() {
            assert_eq!(count_ones_simd(sse4_2, bytes, offset, len), expected);
        }
        if let Some(avx2) = level.as_avx2() {
            assert_eq!(count_ones_simd(avx2, bytes, offset, len), expected);
        }
        if let Some(avx512) = level.as_avx512() {
            assert_eq!(count_ones_simd(avx512, bytes, offset, len), expected);
        }
    }
}
