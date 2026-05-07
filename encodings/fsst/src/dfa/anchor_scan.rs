// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Global anchor scan: build a packed bitset over `all_bytes` whose bit `i` is
//! set iff `all_bytes[i]` is one of the DFA's "progressing" code bytes.
//!
//! Used by [`super::folded_contains::FoldedContainsDfa::scan_to_bitbuf`] when
//! the per-string skip strategy would be a 4+ byte set: per-string `memchr3`
//! cannot be used and a per-code bitmap probe inside the DFA's inner loop is
//! more expensive than a streaming AVX2 scan over the entire `all_bytes`
//! buffer once. Materializing the dense bitset turns "find next progressing
//! code in this string" into a few `u64` AND/range-mask operations.
//!
//! ## Algorithm
//!
//! We support sets of up to 8 progressing codes via the SIMD-in-a-Register
//! "PSHUFB Mula" technique:
//!
//! 1. Each set member is assigned a unique bit (1..=8). Build two 16-byte
//!    nibble tables: `lo_table[i]` ORs the bits of all set members whose low
//!    nibble equals `i`; `hi_table[i]` does the same for high nibbles.
//! 2. For each 32-byte block of input, two `vpshufb` lookups produce a 32-byte
//!    "lo bits" vector and a 32-byte "hi bits" vector. A bytewise AND then
//!    intersects per-byte: bit `b` survives iff the input byte's lo nibble
//!    selected `b` AND its hi nibble selected `b` — that is, the input byte
//!    equals set member `b`.
//! 3. A `vpcmpgtb` vs zero followed by `vpmovmskb` collapses 32 bytes into a
//!    32-bit hit mask, which is splatted into the output bitset.
//!
//! The same scheme works in pure scalar by replacing the AVX2 lookups with
//! 16-entry `[u8; 16]` table indexing.
//!
//! Throughput on a typical x86_64 part is bounded by load + a couple of vector
//! lookups per 32 input bytes, putting the scan well into memchr-class
//! territory.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128i;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m256i;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm_loadu_si128;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_and_si256;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_broadcastsi128_si256;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_cmpgt_epi8;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_loadu_si256;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_movemask_epi8;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_setzero_si256;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_shuffle_epi8;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_srli_epi64;

/// Maximum number of progressing codes the SIMD/scalar nibble table scheme
/// can encode in a single pass. The PSHUFB Mula trick uses one bit per set
/// member.
pub(super) const MAX_SET_BYTES: usize = 8;

/// Precomputed lookup tables for the nibble-table membership check.
struct NibbleTables {
    lo: [u8; 16],
    hi: [u8; 16],
}

impl NibbleTables {
    /// Build the lookup tables for the given progressing codes.
    ///
    /// Returns `None` if the set has more than [`MAX_SET_BYTES`] entries.
    fn build(codes: &[u8]) -> Option<Self> {
        if codes.len() > MAX_SET_BYTES {
            return None;
        }
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for (i, &b) in codes.iter().enumerate() {
            let bit = 1u8 << i;
            lo[usize::from(b & 0x0F)] |= bit;
            hi[usize::from(b >> 4)] |= bit;
        }
        Some(Self { lo, hi })
    }
}

/// Build a packed bitset of length `all_bytes.len()` whose bit `i` is set iff
/// `all_bytes[i]` is in `progressing_codes`.
///
/// Returns `None` when the set has more than [`MAX_SET_BYTES`] entries (the
/// caller must fall back to a different scan strategy in that case).
///
/// The output `Vec<u64>` is sized to fit `all_bytes.len()` bits, padded up to
/// the next 64-bit boundary.
pub(super) fn build_progressing_bitset(
    all_bytes: &[u8],
    progressing_codes: &[u8],
) -> Option<Vec<u64>> {
    let tables = NibbleTables::build(progressing_codes)?;
    let n_words = all_bytes.len().div_ceil(64);
    let mut out = vec![0u64; n_words];

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature was just detected at runtime.
            unsafe { fill_bitset_avx2(all_bytes, &tables, &mut out) };
            return Some(out);
        }
    }

    fill_bitset_scalar(all_bytes, &tables, &mut out);
    Some(out)
}

/// Scalar fallback: fill `out` with the progressing-code bitset using two
/// 16-entry nibble tables.
fn fill_bitset_scalar(all_bytes: &[u8], tables: &NibbleTables, out: &mut [u64]) {
    let mut word: u64 = 0;
    let mut bit_in_word: u64 = 0;
    let mut word_idx: usize = 0;

    for &b in all_bytes {
        let lo_bits = tables.lo[usize::from(b & 0x0F)];
        let hi_bits = tables.hi[usize::from(b >> 4)];
        let hit = (lo_bits & hi_bits) != 0;
        word |= u64::from(hit) << bit_in_word;
        bit_in_word += 1;
        if bit_in_word == 64 {
            out[word_idx] = word;
            word_idx += 1;
            word = 0;
            bit_in_word = 0;
        }
    }
    if bit_in_word > 0 {
        out[word_idx] = word;
    }
}

/// AVX2 implementation: 32 input bytes per iteration, producing a 32-bit hit
/// mask via PSHUFB-Mula nibble lookups. The mask is splatted into the output
/// `u64` bitset; tail bytes (< 32) are handled with a scalar pass.
///
/// # Safety
///
/// Requires AVX2 to be available at runtime. Caller must check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn fill_bitset_avx2(all_bytes: &[u8], tables: &NibbleTables, out: &mut [u64]) {
    use core::arch::x86_64::_mm256_or_si256;
    use core::arch::x86_64::_mm256_set1_epi8;

    let len = all_bytes.len();
    let main_len = len & !31; // round down to multiple of 32

    // Broadcast the 16-byte nibble tables to both 128-bit lanes of a 256-bit
    // register (vpshufb operates per-lane).
    let lo_table =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.lo.as_ptr() as *const __m128i));
    let hi_table =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.hi.as_ptr() as *const __m128i));
    let zero = _mm256_setzero_si256();

    // PSHUFB index handling: each PSHUFB index byte's high bit (0x80), when
    // set, forces the output byte to 0. So before either lookup we must mask
    // the index bytes to the relevant 4 bits.
    //
    // High-nibble extraction: `_mm256_srli_epi64` shifts at 64-bit granularity
    // (bits leak across byte boundaries within each 64-bit lane), so after
    // shifting the high nibble of each byte ends up in its low 4 bits but the
    // high 4 bits are contaminated. We mask to recover the clean high nibble.
    let nibble_mask = _mm256_set1_epi8(0x0F);

    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    while i < main_len {
        // SAFETY: `i < main_len <= len - 31`, so a 32-byte load is in bounds.
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);

        // lo_bits = pshufb(lo_table, v & 0x0F). Masking with 0x0F clears the
        // PSHUFB high-bit-zeroes-output behavior for any input byte ≥ 0x80.
        let v_lo = _mm256_and_si256(v, nibble_mask);
        let lo_bits = _mm256_shuffle_epi8(lo_table, v_lo);

        // hi_bits = pshufb(hi_table, (v >> 4) & 0x0F).
        let v_hi = _mm256_and_si256(_mm256_srli_epi64(v, 4), nibble_mask);
        let hi_bits = _mm256_shuffle_epi8(hi_table, v_hi);

        let merged = _mm256_and_si256(lo_bits, hi_bits);
        // Per-byte: any non-zero bit means "this byte is a member". Member #7
        // has bit 0x80, which is negative as `i8`, so a single signed
        // `_mm256_cmpgt_epi8(merged, 0)` would miss it. OR the "> 0" and
        // "< 0" comparisons to cover all non-zero bytes.
        let pos = _mm256_cmpgt_epi8(merged, zero);
        let neg = _mm256_cmpgt_epi8(zero, merged);
        let hit = _mm256_or_si256(pos, neg);
        let mask = _mm256_movemask_epi8(hit) as u32;

        // Splat `mask` (32 bits) into the output bitset at bit position `i`.
        // Since `i` is a multiple of 32, the 32-bit mask aligns with the lower
        // half (bit_off=0) or upper half (bit_off=32) of one `u64`, never
        // straddling.
        let word_idx = i >> 6;
        let bit_off = (i & 63) as u64;
        let m64 = u64::from(mask);
        // SAFETY: `i + 31 < len <= n_words * 64`, so `word_idx < n_words`.
        *out.get_unchecked_mut(word_idx) |= m64 << bit_off;

        i += 32;
    }

    // Tail: scalar.
    if i < len {
        let mut bit_in_word = (i & 63) as u64;
        let mut word_idx = i >> 6;
        let mut word = out[word_idx];
        for &b in &all_bytes[i..] {
            let lo_bits = tables.lo[usize::from(b & 0x0F)];
            let hi_bits = tables.hi[usize::from(b >> 4)];
            let hit = (lo_bits & hi_bits) != 0;
            word |= u64::from(hit) << bit_in_word;
            bit_in_word += 1;
            if bit_in_word == 64 {
                out[word_idx] = word;
                word_idx += 1;
                word = 0;
                bit_in_word = 0;
            }
        }
        if bit_in_word > 0 {
            out[word_idx] = word;
        }
    }
}

/// Find the next set bit in `bitset` whose absolute position is in
/// `[start, end)`. Returns `None` if no such bit exists.
///
/// Uses `tzcnt`-style word probing: for each `u64` word, mask out bits below
/// `start` (in the first word) or above `end-1` (in the last word), then
/// count trailing zeros. Skips entire 64-bit zero words in one cycle.
#[inline]
pub(super) fn next_set_in_range(bitset: &[u64], start: usize, end: usize) -> Option<usize> {
    if start >= end {
        return None;
    }
    let last_word = (end - 1) >> 6;
    let mut word_idx = start >> 6;
    let first_off = (start & 63) as u64;

    // Head: bits >= first_off in the first word.
    // SAFETY: word_idx <= last_word < bitset.len() (caller invariant).
    let mut w = unsafe { *bitset.get_unchecked(word_idx) } & (!0u64 << first_off);

    while w == 0 {
        if word_idx >= last_word {
            return None;
        }
        word_idx += 1;
        // SAFETY: word_idx <= last_word < bitset.len().
        w = unsafe { *bitset.get_unchecked(word_idx) };
    }

    let bit = w.trailing_zeros() as usize;
    let pos = (word_idx << 6) | bit;
    (pos < end).then_some(pos)
}

/// Probe the bitset for any set bit in the byte range `[start, end)`. Used by
/// the streaming-merge phase to decide whether to dispatch a per-string DFA
/// run or write `false` (or `negated`) directly.
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn range_has_hit(bitset: &[u64], start: usize, end: usize) -> bool {
    if start >= end {
        return false;
    }
    let first_word = start >> 6;
    let last_word = (end - 1) >> 6;
    let first_off = (start & 63) as u64;
    // Bits to keep in the last word: bits 0..=(end-1)&63 inclusive.
    let last_off = ((end - 1) & 63) as u64;

    if first_word == last_word {
        // Build a mask covering bits [first_off..=last_off].
        let width = last_off - first_off + 1;
        let mask: u64 = if width == 64 {
            u64::MAX
        } else {
            ((1u64 << width) - 1) << first_off
        };
        // SAFETY: caller guarantees bitset is large enough.
        return (unsafe { *bitset.get_unchecked(first_word) } & mask) != 0;
    }

    // Multi-word range.
    let head_mask: u64 = !0u64 << first_off;
    if (unsafe { *bitset.get_unchecked(first_word) } & head_mask) != 0 {
        return true;
    }
    for w in (first_word + 1)..last_word {
        if unsafe { *bitset.get_unchecked(w) } != 0 {
            return true;
        }
    }
    let tail_mask: u64 = if last_off == 63 {
        u64::MAX
    } else {
        (1u64 << (last_off + 1)) - 1
    };
    (unsafe { *bitset.get_unchecked(last_word) } & tail_mask) != 0
}

/// Collect the progressing codes from a 256-entry transition row, returning
/// `None` if there are more than [`MAX_SET_BYTES`] of them.
///
/// Mirrors the criterion used by [`super::skip::SkipStrategy`]: a code is
/// "progressing" if `transition_row[code] != start_state` or `code` is the
/// FSST escape code.
pub(super) fn collect_progressing_codes(transition_row: &[u8], start_state: u8) -> Option<Vec<u8>> {
    debug_assert!(transition_row.len() >= 256);
    let mut codes: Vec<u8> = Vec::with_capacity(MAX_SET_BYTES);
    for code in 0..=255u8 {
        if transition_row[usize::from(code)] != start_state || code == fsst::ESCAPE_CODE {
            if codes.len() >= MAX_SET_BYTES {
                return None;
            }
            codes.push(code);
        }
    }
    Some(codes)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn naive_bitset(all_bytes: &[u8], codes: &[u8]) -> Vec<u64> {
        let mut out = vec![0u64; all_bytes.len().div_ceil(64)];
        for (i, &b) in all_bytes.iter().enumerate() {
            if codes.contains(&b) {
                out[i >> 6] |= 1u64 << (i & 63);
            }
        }
        out
    }

    #[rstest]
    #[case(&[1, 2, 3], 0)]
    #[case(&[1, 2, 3], 7)]
    #[case(&[1, 2, 3], 31)]
    #[case(&[1, 2, 3], 32)]
    #[case(&[1, 2, 3], 33)]
    #[case(&[1, 2, 3], 63)]
    #[case(&[1, 2, 3], 64)]
    #[case(&[1, 2, 3], 65)]
    #[case(&[1, 2, 3], 127)]
    #[case(&[1, 2, 3], 128)]
    #[case(&[1, 2, 3], 1000)]
    #[case(&[0xFF, 0x80, 0x42, 0x00], 257)]
    #[case(&[0xFF], 200)]
    #[case(&[0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70], 4096)]
    fn bitset_matches_naive(#[case] codes: &[u8], #[case] len: usize) {
        // Build a deterministic input that exercises every byte value.
        let bytes: Vec<u8> = (0..len)
            .map(|i| u8::try_from(i & 0xFF).unwrap().wrapping_mul(31))
            .collect();
        let got = build_progressing_bitset(&bytes, codes).expect("set fits");
        let expected = naive_bitset(&bytes, codes);
        assert_eq!(got, expected, "mismatch for len={len}, codes={codes:?}");
    }

    #[test]
    fn rejects_too_many_codes() {
        let codes: Vec<u8> =
            (0..u8::try_from(MAX_SET_BYTES).unwrap() + 1).collect();
        let bytes = vec![0u8; 100];
        assert!(build_progressing_bitset(&bytes, &codes).is_none());
    }

    #[test]
    fn next_set_in_range_basic() {
        // bits 5, 70, 130 set across 192 bits (3 words). Caller must size the
        // bitset to cover at least `(end-1) >> 6` words.
        let bitset = vec![1u64 << 5, 1u64 << 6, 1u64 << 2];
        assert_eq!(next_set_in_range(&bitset, 0, 192), Some(5));
        assert_eq!(next_set_in_range(&bitset, 5, 192), Some(5));
        assert_eq!(next_set_in_range(&bitset, 6, 192), Some(70));
        assert_eq!(next_set_in_range(&bitset, 71, 192), Some(130));
        assert_eq!(next_set_in_range(&bitset, 131, 192), None);
        assert_eq!(next_set_in_range(&bitset, 0, 5), None);
        assert_eq!(next_set_in_range(&bitset, 0, 6), Some(5));
        assert_eq!(next_set_in_range(&bitset, 6, 70), None);
        assert_eq!(next_set_in_range(&bitset, 6, 71), Some(70));
    }

    #[test]
    fn range_has_hit_basic() {
        // bits 5, 70, 130 set
        let bitset = vec![1u64 << 5, 1u64 << 6, 1u64 << 2];
        assert!(range_has_hit(&bitset, 5, 6));
        assert!(!range_has_hit(&bitset, 6, 70));
        assert!(range_has_hit(&bitset, 6, 71));
        assert!(range_has_hit(&bitset, 70, 71));
        assert!(!range_has_hit(&bitset, 71, 130));
        assert!(range_has_hit(&bitset, 0, 200));
        assert!(!range_has_hit(&bitset, 10, 10));
    }
}
