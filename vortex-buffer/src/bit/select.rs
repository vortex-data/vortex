// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::count_ones::align_offset_len;

/// Returns the position of the `nth` set bit (0-indexed) within the logical range
/// `[offset, offset + len)` of the given byte slice.
///
/// The returned position is relative to the logical start (i.e., 0-indexed from `offset`).
/// Returns `None` if `nth` is out of bounds.
///
/// Uses architecture-specific optimizations:
/// - **aarch64**: NEON `vcnt`-based popcount for the 64-byte chunk scan.
/// - **x86_64 + AVX-512 VPOPCNTDQ**: 64-byte chunk scan.
/// - **x86_64 + AVX-512 VBMI2**: byte-lane compress for the final in-word select.
/// - **x86_64 + BMI2**: `pdep` + `tzcnt` for the final in-word select.
/// - **Scalar fallback**: 4× unrolled word scan with `count_ones`, byte-level narrowing.
#[inline]
pub fn bit_select(bytes: &[u8], offset: usize, len: usize, nth: usize) -> Option<usize> {
    let (head, middle, tail) = align_offset_len(bytes, offset, len);
    let mut remaining = nth;
    let mut pos = 0usize;

    // ── partial first byte ──────────────────────────────────────────────
    if let Some(head) = head {
        let count = head.count_ones() as usize;
        if remaining < count {
            return Some(select_in_byte(head, remaining));
        }
        remaining -= count;
        let start_bit = offset % 8;
        pos = (8 - start_bit).min(len);
    }

    // ── aligned middle bytes ────────────────────────────────────────────
    if !middle.is_empty() {
        let (chunks, tail_bytes) = middle.as_chunks::<64>();

        let (rem, new_pos, chunk_idx) = scan_chunks(chunks, remaining, pos);
        remaining = rem;
        pos = new_pos;

        if chunk_idx < chunks.len() {
            return Some(pos + select_in_chunk(&chunks[chunk_idx], remaining));
        }

        let (words, tail_bytes) = tail_bytes.as_chunks::<8>();

        let (rem, new_pos, word_idx) = scan_words(words, remaining, pos);
        remaining = rem;
        pos = new_pos;

        if word_idx < words.len() {
            let word = u64::from_le_bytes(words[word_idx]);
            return Some(pos + select_in_word(word, remaining));
        }

        // Remaining aligned bytes that don't fill a full u64.
        for &byte in tail_bytes {
            let count = byte.count_ones() as usize;
            if remaining < count {
                return Some(pos + select_in_byte(byte, remaining));
            }
            remaining -= count;
            pos += 8;
        }
    }

    // ── partial last byte ───────────────────────────────────────────────
    if let Some(tail) = tail
        && remaining < tail.count_ones() as usize
    {
        return Some(pos + select_in_byte(tail, remaining));
    }

    None
}

// ── 64-byte chunk scan ──────────────────────────────────────────────────

/// Scan `chunks` accumulating popcounts. Returns `(remaining, position, chunk_index)`.
///
/// If `chunk_index < chunks.len()`, the target bit is inside that chunk and `remaining`
/// is the rank *within* that chunk. Otherwise all chunks were consumed.
#[inline]
fn scan_chunks(chunks: &[[u8; 64]], remaining: usize, pos: usize) -> (usize, usize, usize) {
    scan_chunks_impl(chunks, remaining, pos)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn scan_chunks_impl(chunks: &[[u8; 64]], remaining: usize, pos: usize) -> (usize, usize, usize) {
    scan_chunks_scalar(chunks, remaining, pos)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn scan_chunks_impl(chunks: &[[u8; 64]], remaining: usize, pos: usize) -> (usize, usize, usize) {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
        // SAFETY: runtime detection guarantees the required target features.
        return unsafe { scan_chunks_avx512_vpopcnt(chunks, remaining, pos) };
    }

    scan_chunks_scalar(chunks, remaining, pos)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::cast_possible_truncation)] // u64 → usize is lossless on aarch64 (64-bit)
#[inline]
fn scan_chunks_impl(
    chunks: &[[u8; 64]],
    mut remaining: usize,
    mut pos: usize,
) -> (usize, usize, usize) {
    use std::arch::aarch64::vcntq_u8;
    use std::arch::aarch64::vgetq_lane_u64;
    use std::arch::aarch64::vld1q_u8;
    use std::arch::aarch64::vpaddlq_u8;
    use std::arch::aarch64::vpaddlq_u16;
    use std::arch::aarch64::vpaddlq_u32;

    for (idx, chunk) in chunks.iter().enumerate() {
        let ptr = chunk.as_ptr();
        // SAFETY: chunk is exactly 64 bytes split across four 128-bit NEON loads.
        // NEON vld1q_u8 supports unaligned access.
        let total = unsafe {
            let pop_0 = vcntq_u8(vld1q_u8(ptr));
            let pop_1 = vcntq_u8(vld1q_u8(ptr.add(16)));
            let pop_2 = vcntq_u8(vld1q_u8(ptr.add(32)));
            let pop_3 = vcntq_u8(vld1q_u8(ptr.add(48)));
            let sums_0 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_0)));
            let sums_1 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_1)));
            let sums_2 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_2)));
            let sums_3 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(pop_3)));

            (vgetq_lane_u64::<0>(sums_0)
                + vgetq_lane_u64::<1>(sums_0)
                + vgetq_lane_u64::<0>(sums_1)
                + vgetq_lane_u64::<1>(sums_1)
                + vgetq_lane_u64::<0>(sums_2)
                + vgetq_lane_u64::<1>(sums_2)
                + vgetq_lane_u64::<0>(sums_3)
                + vgetq_lane_u64::<1>(sums_3)) as usize
        };

        if remaining < total {
            return (remaining, pos, idx);
        }

        remaining -= total;
        pos += 512;
    }

    (remaining, pos, chunks.len())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn scan_chunks_avx512_vpopcnt(
    chunks: &[[u8; 64]],
    mut remaining: usize,
    mut pos: usize,
) -> (usize, usize, usize) {
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_popcnt_epi64;
    use std::arch::x86_64::_mm512_reduce_add_epi64;

    use vortex_error::VortexExpect;

    for (idx, chunk) in chunks.iter().enumerate() {
        // SAFETY: chunk is exactly 64 bytes. `_mm512_loadu_si512` supports unaligned access.
        let block = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
        let counts = _mm512_popcnt_epi64(block);
        let total =
            usize::try_from(_mm512_reduce_add_epi64(counts)).vortex_expect("must fit in usize");

        if remaining < total {
            return (remaining, pos, idx);
        }

        remaining -= total;
        pos += 512;
    }

    (remaining, pos, chunks.len())
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn scan_chunks_scalar(
    chunks: &[[u8; 64]],
    mut remaining: usize,
    mut pos: usize,
) -> (usize, usize, usize) {
    for (idx, chunk) in chunks.iter().enumerate() {
        let total = count_ones_chunk(chunk);
        if remaining < total {
            return (remaining, pos, idx);
        }

        remaining -= total;
        pos += 512;
    }

    (remaining, pos, chunks.len())
}

// ── Word-level scan ─────────────────────────────────────────────────────

/// Scan `words` accumulating popcounts. Returns `(remaining, position, word_index)`.
///
/// If `word_index < words.len()`, the target bit is inside that word and `remaining`
/// is the rank *within* that word. Otherwise all words were consumed.
#[inline]
fn scan_words(words: &[[u8; 8]], remaining: usize, pos: usize) -> (usize, usize, usize) {
    scan_words_impl(words, remaining, pos)
}

// ── Scalar word scan ────────────────────────────────────────────────────

#[inline]
fn scan_words_impl(words: &[[u8; 8]], remaining: usize, pos: usize) -> (usize, usize, usize) {
    scan_words_scalar(words, remaining, pos)
}

#[inline]
fn scan_words_scalar(
    words: &[[u8; 8]],
    mut remaining: usize,
    mut pos: usize,
) -> (usize, usize, usize) {
    let mut idx = 0;

    // 4× unrolled: the four independent `count_ones` calls pipeline well.
    while idx + 4 <= words.len() {
        let count_0 = u64::from_le_bytes(words[idx]).count_ones() as usize;
        let count_1 = u64::from_le_bytes(words[idx + 1]).count_ones() as usize;
        let count_2 = u64::from_le_bytes(words[idx + 2]).count_ones() as usize;
        let count_3 = u64::from_le_bytes(words[idx + 3]).count_ones() as usize;
        let total = count_0 + count_1 + count_2 + count_3;

        if remaining >= total {
            remaining -= total;
            pos += 256;
            idx += 4;
            continue;
        }

        if remaining < count_0 {
            return (remaining, pos, idx);
        }
        remaining -= count_0;
        pos += 64;
        if remaining < count_1 {
            return (remaining, pos, idx + 1);
        }
        remaining -= count_1;
        pos += 64;
        if remaining < count_2 {
            return (remaining, pos, idx + 2);
        }
        remaining -= count_2;
        pos += 64;
        return (remaining, pos, idx + 3);
    }

    while idx < words.len() {
        let word = u64::from_le_bytes(words[idx]);
        let count = word.count_ones() as usize;
        if remaining < count {
            return (remaining, pos, idx);
        }
        remaining -= count;
        pos += 64;
        idx += 1;
    }

    (remaining, pos, idx)
}

// ── In-chunk select ─────────────────────────────────────────────────────

/// Position of the `nth` set bit inside a 64-byte chunk (0-indexed).
#[inline]
fn select_in_chunk(chunk: &[u8; 64], nth: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vpopcntdq")
            && is_x86_feature_detected!("avx512vbmi2")
        {
            // SAFETY: runtime detection guarantees the required target features.
            return unsafe { select_in_chunk_vbmi2(chunk, nth) };
        }
    }

    select_in_chunk_scalar(chunk, nth)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq,avx512vbmi2")]
unsafe fn select_in_chunk_vbmi2(chunk: &[u8; 64], mut nth: usize) -> usize {
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_popcnt_epi64;
    use std::arch::x86_64::_mm512_storeu_epi64;

    use vortex_error::VortexExpect;

    let words = chunk.as_chunks::<8>().0;

    // SAFETY: chunk is exactly 64 bytes. `_mm512_loadu_si512` supports unaligned access.
    let block = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
    let counts = _mm512_popcnt_epi64(block);
    let mut lane_counts = [0_i64; 8];

    // SAFETY: `lane_counts` has room for all eight i64 lanes.
    unsafe { _mm512_storeu_epi64(lane_counts.as_mut_ptr(), counts) };

    for (idx, count) in lane_counts.into_iter().enumerate() {
        let count = usize::try_from(count).vortex_expect("must fit in usize");
        if nth < count {
            return idx * 64 + select_in_word(u64::from_le_bytes(words[idx]), nth);
        }
        nth -= count;
    }

    unreachable!("select_in_chunk: nth exceeds popcount")
}

#[inline]
fn select_in_chunk_scalar(chunk: &[u8; 64], mut nth: usize) -> usize {
    let words = chunk.as_chunks::<8>().0;

    for (idx, word) in words.iter().enumerate() {
        let word = u64::from_le_bytes(*word);
        let count = word.count_ones() as usize;
        if nth < count {
            return idx * 64 + select_in_word(word, nth);
        }
        nth -= count;
    }

    unreachable!("select_in_chunk: nth exceeds popcount")
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn count_ones_chunk(chunk: &[u8; 64]) -> usize {
    let words = chunk.as_chunks::<8>().0;
    u64::from_le_bytes(words[0]).count_ones() as usize
        + u64::from_le_bytes(words[1]).count_ones() as usize
        + u64::from_le_bytes(words[2]).count_ones() as usize
        + u64::from_le_bytes(words[3]).count_ones() as usize
        + u64::from_le_bytes(words[4]).count_ones() as usize
        + u64::from_le_bytes(words[5]).count_ones() as usize
        + u64::from_le_bytes(words[6]).count_ones() as usize
        + u64::from_le_bytes(words[7]).count_ones() as usize
}

// ── In-word select ──────────────────────────────────────────────────────

/// Position of the `nth` set bit inside a u64 (0-indexed, little-endian bit order).
#[inline]
fn select_in_word(word: u64, nth: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            // SAFETY: runtime detection guarantees the required target feature.
            return unsafe { select_in_word_bmi2(word, nth) };
        }
    }
    select_in_word_scalar(word, nth)
}

/// BMI2: deposit a single bit at the nth set-bit position, then count trailing zeros.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn select_in_word_bmi2(word: u64, nth: usize) -> usize {
    use std::arch::x86_64::_pdep_u64;
    use std::arch::x86_64::_tzcnt_u64;

    use vortex_error::VortexExpect;

    usize::try_from(unsafe { _tzcnt_u64(_pdep_u64(1u64 << nth, word)) })
        .vortex_expect("safe to convert tzcnt result to usize")
}

/// Scalar: narrow to the correct byte, then clear `nth` lowest set bits and trailing-zeros.
#[inline]
fn select_in_word_scalar(word: u64, mut nth: usize) -> usize {
    let bytes = word.to_le_bytes();
    let mut bit_offset = 0usize;
    for &byte in &bytes {
        let count = byte.count_ones() as usize;
        if nth < count {
            return bit_offset + select_in_byte(byte, nth);
        }
        nth -= count;
        bit_offset += 8;
    }
    unreachable!("select_in_word: nth exceeds popcount")
}

// ── In-byte select ──────────────────────────────────────────────────────

/// Position of the `nth` set bit inside a byte (0-indexed, LSB-first).
///
/// Clears the lowest `nth` set bits, then uses `trailing_zeros`.
#[inline]
fn select_in_byte(byte: u8, nth: usize) -> usize {
    debug_assert!(nth < byte.count_ones() as usize);
    let mut bits = u32::from(byte);
    for _ in 0..nth {
        bits &= bits - 1; // clear lowest set bit
    }
    bits.trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use rstest::rstest;

    use super::*;

    #[test]
    fn test_select_all_set() {
        // Every bit is set — select(n) == n.
        let buf = [0xFFu8; 16]; // 128 bits, all set
        for nth in 0..128 {
            assert_eq!(bit_select(&buf, 0, 128, nth), Some(nth), "nth={nth}");
        }
    }

    #[test]
    fn test_select_every_other() {
        // 0b01010101 repeated: bits 0,2,4,6 of each byte are set.
        let buf = [0x55u8; 16]; // 128 bits, 64 set
        for nth in 0..64 {
            assert_eq!(bit_select(&buf, 0, 128, nth), Some(nth * 2), "nth={nth}");
        }
    }

    #[test]
    fn test_select_single_bit() {
        // Only bit 42 is set.
        let mut buf = [0u8; 16];
        buf[42 / 8] |= 1 << (42 % 8);
        assert_eq!(bit_select(&buf, 0, 128, 0), Some(42));
    }

    #[test]
    fn test_select_out_of_bounds_returns_none() {
        let buf = [0b0001_0100u8];
        assert_eq!(bit_select(&buf, 0, 8, 0), Some(2));
        assert_eq!(bit_select(&buf, 0, 8, 1), Some(4));
        assert_eq!(bit_select(&buf, 0, 8, 2), None);
    }

    #[rstest]
    #[case(0, 128)]
    #[case(3, 100)]
    #[case(7, 50)]
    #[case(1, 7)]
    #[case(5, 5)]
    #[case(0, 1)]
    #[case(0, 64)]
    #[case(1, 64)]
    #[case(0, 65)]
    #[case(3, 256)]
    #[case(0, 512)]
    #[case(0, 513)]
    #[case(5, 1024)]
    fn test_select_agrees_with_naive(#[case] offset: usize, #[case] len: usize) {
        let total_bits = offset + len;
        let total_bytes = total_bits.div_ceil(8);
        // Deterministic pattern with moderate density.
        let buf: Vec<u8> = (0..total_bytes)
            .map(|i| ((i.wrapping_mul(0x9E) ^ 0xA5) & 0xFF) as u8)
            .collect();

        // Collect set-bit positions naively.
        let expected: Vec<usize> = (0..len)
            .filter(|&i| {
                let phys = offset + i;
                (buf[phys / 8] >> (phys % 8)) & 1 == 1
            })
            .collect();

        for (nth, &expected_pos) in expected.iter().enumerate() {
            assert_eq!(
                bit_select(&buf, offset, len, nth),
                Some(expected_pos),
                "offset={offset} len={len} nth={nth}"
            );
        }
    }

    #[test]
    fn test_select_large_buffer() {
        // ~64 KB buffer, ~50% density.
        let len = 65_536 * 8;
        let buf: Vec<u8> = (0u32..65_536)
            .map(|i| ((i.wrapping_mul(0x37) ^ 0xBC) & 0xFF) as u8)
            .collect();

        let true_count = buf.iter().map(|b| b.count_ones() as usize).sum::<usize>();

        // Spot-check a few positions.
        let first = bit_select(&buf, 0, len, 0);
        let last = bit_select(&buf, 0, len, true_count - 1);
        let first = first.expect("buffer has at least one set bit");
        let last = last.expect("true_count - 1 is in bounds");
        assert!(first < len);
        assert!(last < len);
        assert!(first <= last);

        // Verify the found positions are actually set.
        assert_ne!(buf[first / 8] & (1 << (first % 8)), 0);
        assert_ne!(buf[last / 8] & (1 << (last % 8)), 0);
    }

    #[test]
    fn test_select_in_word_basic() {
        // 0b1010_1010 = 0xAA — bits 1,3,5,7 are set.
        let word = 0x00000000_000000AAu64;
        assert_eq!(select_in_word(word, 0), 1);
        assert_eq!(select_in_word(word, 1), 3);
        assert_eq!(select_in_word(word, 2), 5);
        assert_eq!(select_in_word(word, 3), 7);
    }

    #[test]
    fn test_select_in_word_all_set() {
        let word = u64::MAX;
        for nth in 0..64 {
            assert_eq!(select_in_word(word, nth), nth, "nth={nth}");
        }
    }

    #[test]
    fn test_select_in_byte_basic() {
        assert_eq!(select_in_byte(0b1010_1010, 0), 1);
        assert_eq!(select_in_byte(0b1010_1010, 1), 3);
        assert_eq!(select_in_byte(0b1010_1010, 2), 5);
        assert_eq!(select_in_byte(0b1010_1010, 3), 7);
        assert_eq!(select_in_byte(0b0000_0001, 0), 0);
        assert_eq!(select_in_byte(0b1000_0000, 0), 7);
        assert_eq!(select_in_byte(0xFF, 7), 7);
    }
}
