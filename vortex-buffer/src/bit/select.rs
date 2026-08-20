// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::count_ones::align_offset_len;
use crate::dispatch::CpuKernel;

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
    bit_select_impl(bytes, offset, len, nth, false)
}

/// Returns the position of the `nth` *unset* bit (0-indexed) within the logical range
/// `[offset, offset + len)` of the given byte slice.
///
/// The complement of [`bit_select`], and the same walk: every tier below is shared, because a
/// fully-valid region of `width` bits holds `width - popcount` zeros. That identity is exact, so
/// no vector load has to be complemented — only the running totals change, and the complement
/// itself happens at the final scalar narrowing step.
#[inline]
pub fn bit_select_zero(bytes: &[u8], offset: usize, len: usize, nth: usize) -> Option<usize> {
    bit_select_impl(bytes, offset, len, nth, true)
}

/// Shared implementation of [`bit_select`] (`invert == false`) and [`bit_select_zero`]
/// (`invert == true`).
///
/// `invert` is loop-invariant at every level, so it costs nothing in the steady state: the
/// caller passes a constant, and each scan loop unswitches on it. It is a runtime parameter
/// rather than a const generic because the tiered kernels declare their `CpuKernel` statics
/// inside their own function bodies, and an item in a function body cannot name that
/// function's generic parameters (E0401).
#[inline]
fn bit_select_impl(
    bytes: &[u8],
    offset: usize,
    len: usize,
    nth: usize,
    invert: bool,
) -> Option<usize> {
    let (head, middle, tail) = align_offset_len(bytes, offset, len);
    let mut remaining = nth;
    let mut pos = 0usize;

    // ── partial first byte ──────────────────────────────────────────────
    if let Some(head) = head {
        // `align_offset_len` hands back the head already shifted down and masked to its valid
        // width, so the bits above that width read as zero. Counting *ones* is therefore correct
        // as-is; counting zeros would also count those padding bits, which is why the zero path
        // needs the valid width in order to re-mask after complementing.
        let start_len = (8 - offset % 8).min(len);
        let head = selectable_byte(head, start_len, invert);
        let count = head.count_ones() as usize;
        if remaining < count {
            return Some(select_in_byte(head, remaining));
        }
        remaining -= count;
        pos = start_len;
    }

    // ── aligned middle bytes ────────────────────────────────────────────
    if !middle.is_empty() {
        let (chunks, tail_bytes) = middle.as_chunks::<64>();

        let (rem, new_pos, chunk_idx) = scan_chunks(chunks, remaining, pos, invert);
        remaining = rem;
        pos = new_pos;

        if chunk_idx < chunks.len() {
            return Some(pos + select_in_chunk(&chunks[chunk_idx], remaining, invert));
        }

        let (words, tail_bytes) = tail_bytes.as_chunks::<8>();

        let (rem, new_pos, word_idx) = scan_words(words, remaining, pos, invert);
        remaining = rem;
        pos = new_pos;

        if word_idx < words.len() {
            let word = selectable_word(u64::from_le_bytes(words[word_idx]), invert);
            return Some(pos + select_in_word(word, remaining));
        }

        // Remaining aligned bytes that don't fill a full u64.
        for &byte in tail_bytes {
            let byte = selectable_byte(byte, 8, invert);
            let count = byte.count_ones() as usize;
            if remaining < count {
                return Some(pos + select_in_byte(byte, remaining));
            }
            remaining -= count;
            pos += 8;
        }
    }

    // ── partial last byte ───────────────────────────────────────────────
    // `pos` has now consumed the head plus every aligned middle byte — exactly the `consumed`
    // that `align_offset_len` subtracted from `len` to size the tail — so `len - pos` is the
    // tail's valid width.
    if let Some(tail) = tail {
        let tail = selectable_byte(tail, len - pos, invert);
        if remaining < tail.count_ones() as usize {
            return Some(pos + select_in_byte(tail, remaining));
        }
    }

    None
}

/// Narrow a byte down to the bits the select should walk.
///
/// On the ones path this is the identity. On the zeros path the byte is complemented, so its set
/// bits are the input's unset bits, and then re-masked to `valid_len` bits so the padding above
/// the valid range does not become phantom zeros.
#[inline]
fn selectable_byte(byte: u8, valid_len: usize, invert: bool) -> u8 {
    if !invert {
        return byte;
    }
    let mask = if valid_len >= 8 {
        u8::MAX
    } else {
        (1u8 << valid_len) - 1
    };
    !byte & mask
}

/// [`selectable_byte`] for a fully-valid word: no mask is needed, every bit counts.
#[inline]
fn selectable_word(word: u64, invert: bool) -> u64 {
    if invert { !word } else { word }
}

/// How many bits a fully-valid `width`-bit region contributes to the select, given its popcount.
#[inline]
fn selectable_count(width: usize, ones: usize, invert: bool) -> usize {
    if invert { width - ones } else { ones }
}

// ── 64-byte chunk scan ──────────────────────────────────────────────────

/// Scan `chunks` accumulating popcounts. Returns `(remaining, position, chunk_index)`.
///
/// If `chunk_index < chunks.len()`, the target bit is inside that chunk and `remaining`
/// is the rank *within* that chunk. Otherwise all chunks were consumed.
type ScanChunks = unsafe fn(&[[u8; 64]], usize, usize, bool) -> (usize, usize, usize);

#[inline]
fn scan_chunks(
    chunks: &[[u8; 64]],
    remaining: usize,
    pos: usize,
    invert: bool,
) -> (usize, usize, usize) {
    // Scans of a couple of chunks don't amortize the dispatch indirection: call the
    // per-architecture unconditional kernel directly so it stays inlinable (see the
    // size-gating note in the CpuKernel docs).
    if chunks.len() <= 2 {
        #[cfg(target_arch = "aarch64")]
        return scan_chunks_neon(chunks, remaining, pos, invert);
        #[allow(unreachable_code)]
        {
            return scan_chunks_scalar(chunks, remaining, pos, invert);
        }
    }

    static KERNEL: CpuKernel<ScanChunks> = CpuKernel::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
                return scan_chunks_avx512_vpopcnt;
            }
        }
        #[cfg(target_arch = "aarch64")]
        return scan_chunks_neon;
        // The aarch64 arm above returns unconditionally (NEON needs no probe), making
        // this portable default unreachable there.
        #[allow(unreachable_code)]
        {
            scan_chunks_scalar
        }
    });
    // SAFETY: the selector only returns kernels that are safe or whose required CPU
    // features were probed before selection.
    unsafe { KERNEL.get()(chunks, remaining, pos, invert) }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::cast_possible_truncation)] // u64 → usize is lossless on aarch64 (64-bit)
#[inline]
fn scan_chunks_neon(
    chunks: &[[u8; 64]],
    mut remaining: usize,
    mut pos: usize,
    invert: bool,
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
        let ones = unsafe {
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
        let total = selectable_count(512, ones, invert);

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
    invert: bool,
) -> (usize, usize, usize) {
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_popcnt_epi64;
    use std::arch::x86_64::_mm512_reduce_add_epi64;

    use vortex_error::VortexExpect;

    for (idx, chunk) in chunks.iter().enumerate() {
        // SAFETY: chunk is exactly 64 bytes. `_mm512_loadu_si512` supports unaligned access.
        let block = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
        let counts = _mm512_popcnt_epi64(block);
        let ones =
            usize::try_from(_mm512_reduce_add_epi64(counts)).vortex_expect("must fit in usize");
        let total = selectable_count(512, ones, invert);

        if remaining < total {
            return (remaining, pos, idx);
        }

        remaining -= total;
        pos += 512;
    }

    (remaining, pos, chunks.len())
}

#[inline]
fn scan_chunks_scalar(
    chunks: &[[u8; 64]],
    mut remaining: usize,
    mut pos: usize,
    invert: bool,
) -> (usize, usize, usize) {
    for (idx, chunk) in chunks.iter().enumerate() {
        let total = selectable_count(512, count_ones_chunk(chunk), invert);
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
fn scan_words(
    words: &[[u8; 8]],
    remaining: usize,
    pos: usize,
    invert: bool,
) -> (usize, usize, usize) {
    scan_words_impl(words, remaining, pos, invert)
}

// ── Scalar word scan ────────────────────────────────────────────────────

#[inline]
fn scan_words_impl(
    words: &[[u8; 8]],
    remaining: usize,
    pos: usize,
    invert: bool,
) -> (usize, usize, usize) {
    scan_words_scalar(words, remaining, pos, invert)
}

#[inline]
fn scan_words_scalar(
    words: &[[u8; 8]],
    mut remaining: usize,
    mut pos: usize,
    invert: bool,
) -> (usize, usize, usize) {
    let mut idx = 0;
    let count_at = |idx: usize| {
        selectable_count(
            64,
            u64::from_le_bytes(words[idx]).count_ones() as usize,
            invert,
        )
    };

    // 4× unrolled: the four independent `count_ones` calls pipeline well.
    while idx + 4 <= words.len() {
        let count_0 = count_at(idx);
        let count_1 = count_at(idx + 1);
        let count_2 = count_at(idx + 2);
        let count_3 = count_at(idx + 3);
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
        let count = count_at(idx);
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

type SelectInChunk = unsafe fn(&[u8; 64], usize, bool) -> usize;

/// Position of the `nth` set (or, when `invert`, unset) bit inside a 64-byte chunk (0-indexed).
#[inline]
fn select_in_chunk(chunk: &[u8; 64], nth: usize, invert: bool) -> usize {
    static KERNEL: CpuKernel<SelectInChunk> = CpuKernel::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vpopcntdq")
                && is_x86_feature_detected!("avx512vbmi2")
            {
                return select_in_chunk_vbmi2;
            }
        }
        select_in_chunk_scalar
    });
    // SAFETY: the selector only returns kernels that are safe or whose required CPU
    // features were probed before selection.
    unsafe { KERNEL.get()(chunk, nth, invert) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq,avx512vbmi2")]
unsafe fn select_in_chunk_vbmi2(chunk: &[u8; 64], mut nth: usize, invert: bool) -> usize {
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

    for (idx, ones) in lane_counts.into_iter().enumerate() {
        let ones = usize::try_from(ones).vortex_expect("must fit in usize");
        let count = selectable_count(64, ones, invert);
        if nth < count {
            let word = selectable_word(u64::from_le_bytes(words[idx]), invert);
            return idx * 64 + select_in_word(word, nth);
        }
        nth -= count;
    }

    unreachable!("select_in_chunk: nth exceeds popcount")
}

#[inline]
fn select_in_chunk_scalar(chunk: &[u8; 64], mut nth: usize, invert: bool) -> usize {
    let words = chunk.as_chunks::<8>().0;

    for (idx, word) in words.iter().enumerate() {
        let word = selectable_word(u64::from_le_bytes(*word), invert);
        let count = word.count_ones() as usize;
        if nth < count {
            return idx * 64 + select_in_word(word, nth);
        }
        nth -= count;
    }

    unreachable!("select_in_chunk: nth exceeds popcount")
}

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

type SelectInWord = unsafe fn(u64, usize) -> usize;

/// Position of the `nth` set bit inside a u64 (0-indexed, little-endian bit order).
#[inline]
fn select_in_word(word: u64, nth: usize) -> usize {
    static KERNEL: CpuKernel<SelectInWord> = CpuKernel::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi2") {
                return select_in_word_bmi2;
            }
        }
        select_in_word_scalar
    });
    // SAFETY: the selector only returns kernels that are safe or whose required CPU
    // features were probed before selection.
    unsafe { KERNEL.get()(word, nth) }
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

    /// Deterministic ~50% density filler, matching the pattern the original tests used.
    fn mixed_bytes(total_bytes: usize) -> Vec<u8> {
        (0..total_bytes)
            .map(|i| ((i.wrapping_mul(0x9E) ^ 0xA5) & 0xFF) as u8)
            .collect()
    }

    /// Every position in `[offset, offset + len)` whose bit equals `want`, in ascending order.
    fn naive_positions(buf: &[u8], offset: usize, len: usize, want: bool) -> Vec<usize> {
        (0..len)
            .filter(|&i| {
                let phys = offset + i;
                ((buf[phys / 8] >> (phys % 8)) & 1 == 1) == want
            })
            .collect()
    }

    /// Both select variants must agree with a bit-at-a-time reference over the whole rank range,
    /// and both must report `None` one past the last rank.
    fn check_against_naive(buf: &[u8], offset: usize, len: usize) {
        for (want, select) in [
            (
                true,
                bit_select as fn(&[u8], usize, usize, usize) -> Option<usize>,
            ),
            (
                false,
                bit_select_zero as fn(&[u8], usize, usize, usize) -> Option<usize>,
            ),
        ] {
            let expected = naive_positions(buf, offset, len, want);
            for (nth, &expected_pos) in expected.iter().enumerate() {
                assert_eq!(
                    select(buf, offset, len, nth),
                    Some(expected_pos),
                    "want={want} offset={offset} len={len} nth={nth}"
                );
            }
            assert_eq!(
                select(buf, offset, len, expected.len()),
                None,
                "want={want} offset={offset} len={len} past-the-end rank"
            );
        }
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
    // Head-only windows: an offset inside the first byte with a length that never leaves it, so
    // `align_offset_len` produces a head and nothing else. The zero path has to know the head's
    // valid width here, or it counts the padding bits above it.
    #[case(1, 1)]
    #[case(1, 6)]
    #[case(4, 3)]
    #[case(7, 1)]
    // Windows whose last byte is partial, exercising the tail's valid width.
    #[case(0, 9)]
    #[case(0, 63)]
    #[case(2, 71)]
    #[case(6, 130)]
    #[case(3, 517)]
    fn test_select_agrees_with_naive(#[case] offset: usize, #[case] len: usize) {
        check_against_naive(&mixed_bytes((offset + len).div_ceil(8)), offset, len);
    }

    /// The mixed pattern above sits near 50% density, which is exactly where confusing ones with
    /// zeros is least visible. These degenerate densities make it obvious.
    #[rstest]
    #[case::all_zero(0x00)]
    #[case::all_one(0xFF)]
    #[case::sparse(0x01)]
    #[case::dense(0xFE)]
    fn test_select_uniform_density(#[case] fill: u8) {
        for (offset, len) in [(0usize, 8usize), (0, 128), (3, 5), (5, 130), (1, 517)] {
            let buf = vec![fill; (offset + len).div_ceil(8)];
            check_against_naive(&buf, offset, len);
        }
    }

    #[test]
    fn test_select_zero_degenerate_buffers() {
        // All ones: no zero to find, at any rank.
        let ones = [0xFFu8; 16];
        assert_eq!(bit_select_zero(&ones, 0, 128, 0), None);

        // All zeros: the nth zero is at position n, and the nth one does not exist.
        let zeros = [0x00u8; 16];
        for nth in 0..128 {
            assert_eq!(bit_select_zero(&zeros, 0, 128, nth), Some(nth), "nth={nth}");
        }
        assert_eq!(bit_select_zero(&zeros, 0, 128, 128), None);
        assert_eq!(bit_select(&zeros, 0, 128, 0), None);
    }

    /// Cross-check the zero select against the already-public counting entry points: the number of
    /// zeros in a window is `len - count_ones`, and that is the first rank that must miss.
    #[test]
    fn test_select_zero_count_agrees_with_count_ones() {
        let buf = mixed_bytes(300);
        for (offset, len) in [(0usize, 2400usize), (5, 2000), (7, 1), (3, 519)] {
            let zeros = len - super::super::count_ones::count_ones(&buf, offset, len);
            if let Some(last) = zeros.checked_sub(1) {
                assert!(bit_select_zero(&buf, offset, len, last).is_some());
            }
            assert_eq!(bit_select_zero(&buf, offset, len, zeros), None);
        }
    }

    #[test]
    fn test_select_zero_large_buffer() {
        // ~64 KB buffer, spanning many 64-byte chunks so the chunk-scan tier runs.
        let len = 65_536 * 8;
        let buf = mixed_bytes(65_536);
        let zeros = len - super::super::count_ones::count_ones(&buf, 0, len);

        for nth in [0usize, 1, 1000, zeros / 2, zeros - 1] {
            let pos = bit_select_zero(&buf, 0, len, nth).expect("rank is in bounds");
            assert_eq!(buf[pos / 8] & (1 << (pos % 8)), 0, "nth={nth} pos={pos}");
            assert_eq!(
                super::super::count_ones::count_ones(&buf, 0, pos),
                pos - nth,
                "nth={nth}: rank1(select0(nth)) must be select0(nth) - nth"
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
