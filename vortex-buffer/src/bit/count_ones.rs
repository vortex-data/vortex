// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::prelude::*;
use fearless_simd::u8x64;
use fearless_simd::u64x8;
use fearless_simd_macros::simd;
use vortex_error::VortexExpect;

#[inline]
pub fn count_ones(bytes: &[u8], offset: usize, len: usize) -> usize {
    if bytes.is_empty() {
        return 0;
    }

    let (head, middle, tail) = align_offset_len(bytes, offset, len);

    let mut count = head.map_or(0, |v| v.count_ones() as usize);

    if !middle.is_empty() {
        count += count_ones_aligned(middle);
    }

    count + tail.map_or(0, |v| v.count_ones() as usize)
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
fn count_ones_aligned(bytes: &[u8]) -> usize {
    // SIMD kernels only pay off from 32 bytes. Below that, call the scalar kernel
    // directly: it stays inlinable and skips the dispatch, which would otherwise
    // dominate the couple of word popcounts.
    if bytes.len() < 32 {
        return count_ones_aligned_scalar(bytes);
    }

    fearless_simd::dispatch!(Level::new(), simd => count_ones_aligned_simd(simd, bytes))
}

#[inline]
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

/// Lane-wise `u64` popcount over 64-byte vectors. The total is independent of lane byte order.
#[simd]
fn count_ones_aligned_simd<S: Simd>(simd: S, bytes: &[u8]) -> usize {
    let (chunks, tail) = bytes.as_chunks::<64>();
    let mut accum = u64x8::splat(simd, 0);
    for chunk in chunks {
        let words: u64x8<S> = u8x64::from_slice(simd, chunk).bitcast();
        accum += words.count_ones();
    }

    usize::try_from(accum.reduce_sum()).vortex_expect("true_count doesn't fit in usize")
        + count_ones_aligned_scalar(tail)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::BitBuffer;

    #[cfg_attr(miri, ignore)]
    #[rstest]
    fn test_count_ones_matches_iteration_for_slices(
        #[values(
            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30
        )]
        offset: usize,
        #[values(
            0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257, 513
        )]
        slice_len: usize,
    ) {
        let len = 513;
        let buf = BitBuffer::collect_bool(len + 31, |i| (i % 3 == 0) ^ (i % 11 == 0));

        if offset + slice_len > buf.len() {
            return;
        }

        let sliced = buf.slice(offset..offset + slice_len);
        let expected = sliced.iter().filter(|bit| *bit).count();

        assert_eq!(
            sliced.true_count(),
            expected,
            "offset={offset} len={slice_len}"
        );
    }

    /// Lengths spanning many 64-byte SIMD chunks plus a scalar tail.
    #[cfg_attr(miri, ignore)]
    #[rstest]
    fn test_count_ones_matches_iteration_multi_chunk(
        #[values(0usize, 5)] offset: usize,
        #[values(1024usize, 4096, 4097, 65_537)] len: usize,
    ) {
        let buf = BitBuffer::collect_bool(offset + len, |i| (i % 3 == 0) ^ (i % 7 == 0));
        let sliced = buf.slice(offset..offset + len);
        let expected = sliced.iter().filter(|bit| *bit).count();

        assert_eq!(sliced.true_count(), expected, "offset={offset} len={len}");
    }
}
