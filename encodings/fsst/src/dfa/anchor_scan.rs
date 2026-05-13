// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// The streaming Teddy pass functions all share a fixed signature that
// threads `ssa_tables`, `offsets`, `bits`, and `verify_at` through every
// architecture variant; splitting them up would obscure the SIMD code
// without making it more readable.
#![allow(clippy::too_many_arguments)]
// The AVX2/AVX-512 inner loops fuse tzcnt-driven candidate peeling with
// inline verifier dispatch; clippy's cognitive-complexity heuristic
// flags them, but splitting the hot loop is what the comments and
// design notes explicitly avoid.
#![allow(clippy::cognitive_complexity)]
// Existing single-letter loop variables (`i`, `j`, `n`, etc.) are
// idiomatic for the byte-position arithmetic.
#![allow(clippy::many_single_char_names)]

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

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::uint8x16_t;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vaddv_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vandq_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vcgtq_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vdupq_n_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vget_high_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vget_low_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vld1q_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vqtbl1q_u8;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::vshrq_n_u8;
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

#[cfg(target_arch = "aarch64")]
const NEON_MOVEMASK_BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

#[cfg(target_arch = "aarch64")]
#[expect(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn neon_nibble_lookup(
    lo_table: uint8x16_t,
    hi_table: uint8x16_t,
    bytes: uint8x16_t,
    nibble_mask: uint8x16_t,
) -> uint8x16_t {
    let lo_idx = vandq_u8(bytes, nibble_mask);
    let hi_idx = vandq_u8(vshrq_n_u8::<4>(bytes), nibble_mask);
    vandq_u8(vqtbl1q_u8(lo_table, lo_idx), vqtbl1q_u8(hi_table, hi_idx))
}

#[cfg(target_arch = "aarch64")]
#[expect(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn neon_nonzero_mask(bytes: uint8x16_t, zero: uint8x16_t, lane_bits: uint8x16_t) -> u16 {
    let nonzero = vcgtq_u8(bytes, zero);
    let weighted = vandq_u8(nonzero, lane_bits);
    u16::from(vaddv_u8(vget_low_u8(weighted))) | (u16::from(vaddv_u8(vget_high_u8(weighted))) << 8)
}

/// Build a packed bitset of length `all_bytes.len()` whose bit `i` is set
/// iff `all_bytes[i] ∈ c1_codes` AND `all_bytes[i+1] ∈ c2_codes`. The last
/// bit (`i == all_bytes.len() - 1`) is forced to 0 — there is no
/// successor byte for the c2 lookup.
///
/// Kept for the `fsst_prefilter_compare` bench, which A/B-tests the
/// legacy Cartesian path against [`build_bucketed_pair_bitset`]. Returns
/// `None` when either union exceeds [`MAX_SET_BYTES`] — which is the
/// case on real FSST-trained dictionaries, the historical reason this
/// path rarely fired before bucketing.
#[cfg(any(test, feature = "_test-harness"))]
pub(super) fn build_pair_bitset(
    all_bytes: &[u8],
    c1_codes: &[u8],
    c2_codes: &[u8],
) -> Option<Vec<u64>> {
    let c1_tables = NibbleTables::build(c1_codes)?;
    let c2_tables = NibbleTables::build(c2_codes)?;
    let n_words = all_bytes.len().div_ceil(64);
    if n_words == 0 {
        return Some(Vec::new());
    }
    let mut c1 = vec![0u64; n_words];
    let mut c2 = vec![0u64; n_words];
    fill_two_bitsets(all_bytes, &c1_tables, &c2_tables, &mut c1, &mut c2);

    // Combine: pair[i] = c1[i] AND c2[i+1]. In u64 word-space:
    //   out[w] = c1[w] & ((c2[w] >> 1) | (c2[w+1] << 63))
    let mut out = c1;
    for w in 0..n_words.saturating_sub(1) {
        // SAFETY: w < n_words - 1 ≤ c2.len() - 1.
        let lo = unsafe { *c2.get_unchecked(w) };
        let hi = unsafe { *c2.get_unchecked(w + 1) };
        let shifted = (lo >> 1) | (hi << 63);
        // SAFETY: w < n_words = out.len().
        unsafe { *out.get_unchecked_mut(w) &= shifted };
    }
    let last = n_words - 1;
    // SAFETY: last < n_words = c2.len() = out.len().
    unsafe {
        *out.get_unchecked_mut(last) &= *c2.get_unchecked(last) >> 1;
    }
    // Force-clear the bit at `all_bytes.len() - 1`: no successor for c2.
    let last_bit = all_bytes.len() - 1;
    let last_word = last_bit >> 6;
    let last_off = last_bit & 63;
    // SAFETY: last_word < n_words = out.len().
    unsafe { *out.get_unchecked_mut(last_word) &= !(1u64 << last_off) };

    Some(out)
}

/// Fused fill of two bitsets in a single walk over `all_bytes`. Used
/// only by [`build_pair_bitset`] — the bench-only legacy Cartesian
/// path.
#[cfg(any(test, feature = "_test-harness"))]
fn fill_two_bitsets(
    all_bytes: &[u8],
    c1_tables: &NibbleTables,
    c2_tables: &NibbleTables,
    c1_out: &mut [u64],
    c2_out: &mut [u64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature was just detected at runtime.
            unsafe {
                fill_two_bitsets_avx2(all_bytes, c1_tables, c2_tables, c1_out, c2_out);
            }
            return;
        }
    }
    fill_bitset_scalar(all_bytes, c1_tables, c1_out);
    fill_bitset_scalar(all_bytes, c2_tables, c2_out);
}

#[cfg(all(target_arch = "x86_64", any(test, feature = "_test-harness")))]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn fill_two_bitsets_avx2(
    all_bytes: &[u8],
    c1_tables: &NibbleTables,
    c2_tables: &NibbleTables,
    c1_out: &mut [u64],
    c2_out: &mut [u64],
) {
    use core::arch::x86_64::_mm256_or_si256;
    use core::arch::x86_64::_mm256_set1_epi8;

    let len = all_bytes.len();
    let main_len = len & !31;

    let c1_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(c1_tables.lo.as_ptr() as *const __m128i));
    let c1_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(c1_tables.hi.as_ptr() as *const __m128i));
    let c2_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(c2_tables.lo.as_ptr() as *const __m128i));
    let c2_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(c2_tables.hi.as_ptr() as *const __m128i));
    let zero = _mm256_setzero_si256();
    let nibble_mask = _mm256_set1_epi8(0x0F);

    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    while i < main_len {
        // SAFETY: `i + 31 < main_len <= len`.
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let v_lo_idx = _mm256_and_si256(v, nibble_mask);
        let v_hi_idx = _mm256_and_si256(_mm256_srli_epi64(v, 4), nibble_mask);

        // c1 membership.
        let c1_lo_b = _mm256_shuffle_epi8(c1_lo, v_lo_idx);
        let c1_hi_b = _mm256_shuffle_epi8(c1_hi, v_hi_idx);
        let c1_merged = _mm256_and_si256(c1_lo_b, c1_hi_b);
        let c1_pos = _mm256_cmpgt_epi8(c1_merged, zero);
        let c1_neg = _mm256_cmpgt_epi8(zero, c1_merged);
        let c1_hit = _mm256_or_si256(c1_pos, c1_neg);
        let c1_mask = _mm256_movemask_epi8(c1_hit) as u32;

        // c2 membership.
        let c2_lo_b = _mm256_shuffle_epi8(c2_lo, v_lo_idx);
        let c2_hi_b = _mm256_shuffle_epi8(c2_hi, v_hi_idx);
        let c2_merged = _mm256_and_si256(c2_lo_b, c2_hi_b);
        let c2_pos = _mm256_cmpgt_epi8(c2_merged, zero);
        let c2_neg = _mm256_cmpgt_epi8(zero, c2_merged);
        let c2_hit = _mm256_or_si256(c2_pos, c2_neg);
        let c2_mask = _mm256_movemask_epi8(c2_hit) as u32;

        let word_idx = i >> 6;
        let bit_off = (i & 63) as u64;
        // SAFETY: i + 31 < len ≤ n_words * 64, so word_idx < n_words.
        *c1_out.get_unchecked_mut(word_idx) |= u64::from(c1_mask) << bit_off;
        *c2_out.get_unchecked_mut(word_idx) |= u64::from(c2_mask) << bit_off;

        i += 32;
    }

    // Tail: scalar.
    if i < len {
        let mut bit_in_word = (i & 63) as u64;
        let mut word_idx = i >> 6;
        let mut c1_word = c1_out[word_idx];
        let mut c2_word = c2_out[word_idx];
        for &b in &all_bytes[i..] {
            let c1_lo_bits = c1_tables.lo[usize::from(b & 0x0F)];
            let c1_hi_bits = c1_tables.hi[usize::from(b >> 4)];
            let c2_lo_bits = c2_tables.lo[usize::from(b & 0x0F)];
            let c2_hi_bits = c2_tables.hi[usize::from(b >> 4)];
            c1_word |= u64::from((c1_lo_bits & c1_hi_bits) != 0) << bit_in_word;
            c2_word |= u64::from((c2_lo_bits & c2_hi_bits) != 0) << bit_in_word;
            bit_in_word += 1;
            if bit_in_word == 64 {
                c1_out[word_idx] = c1_word;
                c2_out[word_idx] = c2_word;
                word_idx += 1;
                c1_word = 0;
                c2_word = 0;
                bit_in_word = 0;
            }
        }
        if bit_in_word > 0 {
            c1_out[word_idx] = c1_word;
            c2_out[word_idx] = c2_word;
        }
    }
}

/// Like [`build_progressing_bitset`], but supports an arbitrary-size code
/// set via multi-pass PSHUFB-Mula OR-merge — chunks of up to
/// [`MAX_SET_BYTES`] codes are scanned in separate passes and OR'd
/// together. Cost scales linearly with `ceil(codes.len() / MAX_SET_BYTES)`
/// passes over `all_bytes`. Always returns `Some(_)` when the input is
/// non-empty.
///
/// Used by the folded-contains scan path on corpora where the
/// state-0 progressing set exceeds the single-pass nibble-table limit
/// (typical for FSST-encoded URL data with rich symbol tables).
pub(super) fn build_progressing_bitset_unbounded(
    all_bytes: &[u8],
    progressing_codes: &[u8],
) -> Vec<u64> {
    let n_words = all_bytes.len().div_ceil(64);
    let mut out = vec![0u64; n_words];
    if progressing_codes.is_empty() || all_bytes.is_empty() {
        return out;
    }
    if progressing_codes.len() <= MAX_SET_BYTES {
        let tables = match NibbleTables::build(progressing_codes) {
            Some(tables) => tables,
            None => unreachable!("progressing_codes length already checked"),
        };
        fill_bitset(all_bytes, &tables, &mut out);
        return out;
    }
    // Multi-pass: chunk the codes, build a per-chunk bitset, OR-merge
    // into `out`. Reuse a scratch buffer across chunks to amortize
    // allocation.
    let mut scratch = vec![0u64; n_words];
    for chunk in progressing_codes.chunks(MAX_SET_BYTES) {
        let tables = match NibbleTables::build(chunk) {
            Some(tables) => tables,
            None => unreachable!("chunk length bounded by MAX_SET_BYTES"),
        };
        scratch.iter_mut().for_each(|w| *w = 0);
        fill_bitset(all_bytes, &tables, &mut scratch);
        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
            *dst |= *src;
        }
    }
    out
}

/// Internal: dispatch to AVX2 fill when available, scalar otherwise.
fn fill_bitset(all_bytes: &[u8], tables: &NibbleTables, out: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature was just detected at runtime.
            unsafe { fill_bitset_avx2(all_bytes, tables, out) };
            return;
        }
    }
    fill_bitset_scalar(all_bytes, tables, out);
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

/// Like [`collect_progressing_codes`], but never returns `None` — collects
/// the full set of progressing codes regardless of cardinality. Pair with
/// [`build_progressing_bitset_unbounded`] which scans in `ceil(N / 8)`
/// PSHUFB passes when `N > MAX_SET_BYTES`.
pub(super) fn collect_progressing_codes_unbounded(
    transition_row: &[u8],
    start_state: u8,
) -> Vec<u8> {
    debug_assert!(transition_row.len() >= 256);
    let mut codes: Vec<u8> = Vec::new();
    for code in 0..=255u8 {
        if transition_row[usize::from(code)] != start_state || code == fsst::ESCAPE_CODE {
            codes.push(code);
        }
    }
    codes
}

/// A bucketed view of pair-eligible codes: one bucket per distinct c1, with
/// the per-c1 set of strictly-advancing-or-escape c2 codes. Used by the
/// shared-c1 bucketed Teddy pair-bitset path: bucket `b` holds the Cartesian
/// sub-product `({c1_b}, c2_set_b)`, so OR-ing the buckets eliminates
/// cross-bucket false-positive pairs that pure Cartesian (`c1_union ×
/// c2_union`) admits.
pub(super) type BucketedPairCodes = Vec<(u8, Vec<u8>)>;

/// Compute shared-c1 buckets for the bucketed pair-bitset scan. Mirrors the
/// per-c1 logic of [`collect_pair_codes`] but keeps the c2 partition
/// per-c1 instead of flattening into a union.
///
/// Returns `None` when no c1 is pair-eligible or when `accept_state < 2`.
pub(super) fn collect_bucketed_pair_codes(
    transitions: &[u8],
    c1_codes: &[u8],
    accept_state: u8,
) -> Option<BucketedPairCodes> {
    if accept_state < 2 {
        return None;
    }
    debug_assert!(transitions.len() >= 256);
    let mut buckets: BucketedPairCodes = Vec::new();
    for &c1 in c1_codes {
        let s1 = transitions[usize::from(c1)];
        if s1 == 0 || s1 == accept_state {
            continue;
        }
        let row = usize::from(s1) * 256;
        let s1_is_escape = s1 > accept_state;
        let mut c2_set: Vec<u8> = Vec::new();
        for c2 in 0..=u8::MAX {
            let next = transitions[row + usize::from(c2)];
            let advances = if s1_is_escape { next != 0 } else { next > s1 };
            let escape = c2 == fsst::ESCAPE_CODE;
            if advances || escape {
                c2_set.push(c2);
            }
        }
        if !c2_set.is_empty() {
            buckets.push((c1, c2_set));
        }
    }
    if buckets.is_empty() {
        None
    } else {
        Some(buckets)
    }
}

/// Build a packed bitset of length `all_bytes.len()` whose bit `i` is set
/// iff `(all_bytes[i], all_bytes[i+1])` is approximated as a pair in
/// `buckets`: there exists a bucket `b` such that `all_bytes[i] == c1_b`
/// AND `all_bytes[i+1]` matches the bucket's c2 nibble tables (a small
/// nibble-cross over-approximation of `c2_set_b` for bucket sizes > 1,
/// admitting within-bucket FPs but never cross-bucket FPs). The last
/// bit (`i == all_bytes.len() - 1`) is forced to 0.
///
/// Single PSHUFB-Mula pass when `buckets.len() ≤ MAX_SET_BYTES`. Larger
/// bucket counts are processed in chunks of `MAX_SET_BYTES` and OR-merged
/// into the output.
///
/// Compared to [`build_pair_bitset`] (pure Cartesian `c1_union × c2_union`),
/// this preserves the per-c1 partition. On real FSST-trained URL data,
/// the c1 set typically has more than one element only when several
/// symbols re-encode the same anchor byte — in which case the bucketed
/// path drops the Cartesian's cross-product FPs (e.g. matching
/// `(c1_a, c2_b)` when only `(c1_a, c2_a)` and `(c1_b, c2_b)` are
/// valid).
pub(super) fn build_bucketed_pair_bitset(all_bytes: &[u8], buckets: &[(u8, Vec<u8>)]) -> Vec<u64> {
    let trace = std::env::var_os("VORTEX_FSST_BUCKET_BUILD_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let total_t = trace.then(std::time::Instant::now);
    let n_words = all_bytes.len().div_ceil(64);
    let alloc_t = trace.then(std::time::Instant::now);
    let mut out = vec![0u64; n_words];
    let alloc_us = alloc_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if buckets.is_empty() || all_bytes.len() < 2 {
        if let Some(total_t) = total_t {
            eprintln!(
                "[fsst::bucket_build] path=empty bytes={} buckets={} words={} alloc_us={:.3} fill_us=0.000 merge_us=0.000 total_us={:.3}",
                all_bytes.len(),
                buckets.len(),
                n_words,
                alloc_us,
                total_t.elapsed().as_secs_f64() * 1e6,
            );
        }
        return out;
    }
    if buckets.len() <= MAX_SET_BYTES {
        let fill_t = trace.then(std::time::Instant::now);
        fill_bucketed_pair(all_bytes, buckets, &mut out);
        let fill_us = fill_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        if let Some(total_t) = total_t {
            eprintln!(
                "[fsst::bucket_build] path=single bytes={} buckets={} words={} alloc_us={:.3} fill_us={:.3} merge_us=0.000 total_us={:.3}",
                all_bytes.len(),
                buckets.len(),
                n_words,
                alloc_us,
                fill_us,
                total_t.elapsed().as_secs_f64() * 1e6,
            );
        }
        return out;
    }
    // Multi-pass: OR-merge per-chunk pair bitsets.
    let scratch_alloc_t = trace.then(std::time::Instant::now);
    let mut scratch = vec![0u64; n_words];
    let scratch_alloc_us = scratch_alloc_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    let mut fill_us = 0f64;
    let mut merge_us = 0f64;
    for chunk in buckets.chunks(MAX_SET_BYTES) {
        scratch.iter_mut().for_each(|w| *w = 0);
        let fill_t = trace.then(std::time::Instant::now);
        fill_bucketed_pair(all_bytes, chunk, &mut scratch);
        fill_us += fill_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        let merge_t = trace.then(std::time::Instant::now);
        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
            *dst |= *src;
        }
        merge_us += merge_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
    }
    if let Some(total_t) = total_t {
        eprintln!(
            "[fsst::bucket_build] path=multipass bytes={} buckets={} words={} alloc_us={:.3} scratch_alloc_us={:.3} fill_us={:.3} merge_us={:.3} total_us={:.3}",
            all_bytes.len(),
            buckets.len(),
            n_words,
            alloc_us,
            scratch_alloc_us,
            fill_us,
            merge_us,
            total_t.elapsed().as_secs_f64() * 1e6,
        );
    }
    out
}

/// Internal: bucket tables for one bucketed pair-bitset pass. `c1_tables`
/// has bit `b` set at the lo/hi nibbles of bucket `b`'s c1. `c2_tables`
/// has bit `b` set at the lo/hi nibbles of any c2 in bucket `b`'s c2 set.
struct BucketTables {
    c1: NibbleTables,
    c2: NibbleTables,
}

impl BucketTables {
    fn build(buckets: &[(u8, Vec<u8>)]) -> Self {
        debug_assert!(buckets.len() <= MAX_SET_BYTES);
        let mut c1_lo = [0u8; 16];
        let mut c1_hi = [0u8; 16];
        let mut c2_lo = [0u8; 16];
        let mut c2_hi = [0u8; 16];
        for (b, (c1, c2_set)) in buckets.iter().enumerate() {
            let bit = 1u8 << b;
            c1_lo[usize::from(c1 & 0x0F)] |= bit;
            c1_hi[usize::from(c1 >> 4)] |= bit;
            for &c2 in c2_set {
                c2_lo[usize::from(c2 & 0x0F)] |= bit;
                c2_hi[usize::from(c2 >> 4)] |= bit;
            }
        }
        Self {
            c1: NibbleTables {
                lo: c1_lo,
                hi: c1_hi,
            },
            c2: NibbleTables {
                lo: c2_lo,
                hi: c2_hi,
            },
        }
    }
}

/// Dispatch to AVX2 bucketed-pair fill when available, scalar otherwise.
fn fill_bucketed_pair(all_bytes: &[u8], buckets: &[(u8, Vec<u8>)], out: &mut [u64]) {
    let tables = BucketTables::build(buckets);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 just detected.
            unsafe { fill_bucketed_pair_avx2(all_bytes, &tables, out) };
            return;
        }
    }
    fill_bucketed_pair_scalar(all_bytes, &tables, out);
}

/// Scalar bucketed-pair fill. Equivalent to the AVX2 path on a byte-by-byte
/// basis; bit `i` is set iff `(c1_mask[i] & c2_mask[i+1]) != 0` for the
/// nibble-table masks. Position `len - 1` is never set (no successor).
fn fill_bucketed_pair_scalar(all_bytes: &[u8], tables: &BucketTables, out: &mut [u64]) {
    let len = all_bytes.len();
    if len < 2 {
        return;
    }
    for i in 0..len - 1 {
        let b1 = all_bytes[i];
        let b2 = all_bytes[i + 1];
        let c1_bits = tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
        let c2_bits = tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
        if (c1_bits & c2_bits) != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
        }
    }
}

/// AVX2 bucketed-pair fill: per 32-byte chunk, compute c1 bucket bits from a
/// load at offset `i` and c2 bucket bits from a load at offset `i + 1`,
/// AND them per-byte, and movemask to a 32-bit candidate mask. Tail
/// positions are handled scalar.
///
/// # Safety
///
/// Requires AVX2 to be available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn fill_bucketed_pair_avx2(all_bytes: &[u8], tables: &BucketTables, out: &mut [u64]) {
    use core::arch::x86_64::_mm256_or_si256;
    use core::arch::x86_64::_mm256_set1_epi8;

    let len = all_bytes.len();
    if len < 2 {
        return;
    }
    // Largest multiple of 32 such that the unaligned load at offset `i + 1`
    // for `i = main_len - 32` is in bounds: `i + 32 + 1 ≤ len` ⇒
    // `main_len ≤ len - 1` ⇒ rounded down to a multiple of 32.
    let main_len = ((len - 1) >> 5) << 5;

    let c1_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.lo.as_ptr() as *const __m128i));
    let c1_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.hi.as_ptr() as *const __m128i));
    let c2_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.lo.as_ptr() as *const __m128i));
    let c2_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.hi.as_ptr() as *const __m128i));
    let zero = _mm256_setzero_si256();
    let nibble_mask = _mm256_set1_epi8(0x0F);

    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    while i < main_len {
        // v1 = bytes[i..i+32] for the c1 lane; v2 = bytes[i+1..i+33] for c2.
        // SAFETY: `i + 32 ≤ main_len ≤ len - 1` so `i + 32 < len`, both
        // 32-byte loads are in bounds.
        let v1 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let v2 = _mm256_loadu_si256(ptr.add(i + 1) as *const __m256i);

        let v1_lo = _mm256_and_si256(v1, nibble_mask);
        let v1_hi = _mm256_and_si256(_mm256_srli_epi64(v1, 4), nibble_mask);
        let v2_lo = _mm256_and_si256(v2, nibble_mask);
        let v2_hi = _mm256_and_si256(_mm256_srli_epi64(v2, 4), nibble_mask);

        let c1_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c1_lo, v1_lo),
            _mm256_shuffle_epi8(c1_hi, v1_hi),
        );
        let c2_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c2_lo, v2_lo),
            _mm256_shuffle_epi8(c2_hi, v2_hi),
        );
        // Per-byte AND preserves the per-bucket bit: surviving bit `b` means
        // the c1 lane matched bucket `b`'s c1 *and* the c2 lane (at offset
        // i+1) matched bucket `b`'s c2 set. Cross-bucket combinations
        // (bit `a` from c1, bit `b` from c2 with `a ≠ b`) AND to zero.
        let pair = _mm256_and_si256(c1_bits, c2_bits);

        // Collapse 32 bytes → 32-bit "any bit set" mask. Bytes whose pair
        // value has bit 7 set are negative as `i8`; cover them with the
        // signed "< 0" comparison.
        let pos = _mm256_cmpgt_epi8(pair, zero);
        let neg = _mm256_cmpgt_epi8(zero, pair);
        let hit = _mm256_or_si256(pos, neg);
        let mask = _mm256_movemask_epi8(hit) as u32;

        // i is a multiple of 32 ⇒ bit_off is 0 or 32; the 32-bit mask
        // never crosses a u64 boundary.
        let word_idx = i >> 6;
        let bit_off = (i & 63) as u64;
        // SAFETY: i + 31 < len, so word_idx < n_words.
        *out.get_unchecked_mut(word_idx) |= u64::from(mask) << bit_off;
        i += 32;
    }

    // Tail: scalar from i to len-2 (inclusive). Position len-1 has no
    // successor and must remain 0.
    for j in i..len - 1 {
        let b1 = *all_bytes.get_unchecked(j);
        let b2 = *all_bytes.get_unchecked(j + 1);
        let c1_bits_b = tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
        let c2_bits_b = tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
        if (c1_bits_b & c2_bits_b) != 0 {
            let word_idx = j >> 6;
            let bit_off = (j & 63) as u64;
            *out.get_unchecked_mut(word_idx) |= 1u64 << bit_off;
        }
    }
}

// ---------------------------------------------------------------------------
// Bucketed Teddy-3 (3-byte fingerprint pair scan).
//
// Same shared-c1 bucket scheme as Teddy-2, extended with a c3 set: bucket
// `b` holds `(c1_b, c2_set_b, c3_set_b)` and the bit is set at position `i`
// iff `all_bytes[i] == c1_b` AND `all_bytes[i+1] ∈ c2_set_b` AND
// `all_bytes[i+2] ∈ c3_set_b` (modulo nibble-cross within-bucket FPs in c2
// and c3). Selectivity scales roughly as `(|c1| · |c2| · |c3|) / 256³`
// per position vs Teddy-2's `(|c1| · |c2|) / 256²`. On dense alphabets
// like ClickBench URLs this typically yields 100–1000× fewer candidate
// hits than Teddy-2, at the cost of one additional PSHUFB-Mula pair and
// one extra unaligned 32-byte load per chunk.
// ---------------------------------------------------------------------------

/// One bucket per distinct c1, with the per-c1 advancing c2 set AND the
/// union (across that c1's c2 set) of strictly-advancing c3 codes. Used
/// by [`build_bucketed_triple_bitset`]. `None` when the needle is too
/// short to admit a 3-byte fingerprint (`accept_state < 3`) or no
/// progressing c1 has a non-empty c2 × c3 advancement chain.
pub(super) type BucketedTripleCodes = Vec<(u8, Vec<u8>, Vec<u8>)>;

/// Compute shared-c1 buckets for the bucketed Teddy-3 scan. Mirrors
/// [`collect_bucketed_pair_codes`] but walks one DFA step further:
/// for each pair-eligible `(c1, c2)` we collect the strictly-advancing
/// c3 codes from state `T[s1][c2]`, then union across c2's of the same
/// bucket.
///
/// We deliberately *skip* escape-state c1's here: under FSST escapes
/// the third byte is a literal byte fed through the byte-level table,
/// and the resulting c3 admission set blows up. The Teddy-2 path
/// continues to cover the escape case.
pub(super) fn collect_bucketed_triple_codes(
    transitions: &[u8],
    c1_codes: &[u8],
    accept_state: u8,
) -> Option<BucketedTripleCodes> {
    if accept_state < 3 {
        return None;
    }
    debug_assert!(transitions.len() >= 256);
    let mut buckets: BucketedTripleCodes = Vec::new();
    for &c1 in c1_codes {
        let s1 = transitions[usize::from(c1)];
        // Skip non-progressing, single-step-accept, and escape c1's. Escape
        // c1's would produce c3 admission sets approaching all literal
        // bytes — the Teddy-2 path handles them better.
        if s1 == 0 || s1 == accept_state || s1 > accept_state {
            continue;
        }
        let row_s1 = usize::from(s1) * 256;
        let mut c2_set: Vec<u8> = Vec::new();
        let mut c3_seen = [false; 256];
        let mut c3_set: Vec<u8> = Vec::new();
        for c2 in 0..=u8::MAX {
            let s2 = transitions[row_s1 + usize::from(c2)];
            // c2 must strictly advance from s1 AND s2 must be a *normal*
            // intermediate state (≠ accept, ≠ escape) so there is room
            // for c3 to advance once more before accept. (Single-step
            // accepts from s1 are already 2-byte matches and would be
            // missed by a 3-byte pair predicate — but they are caught
            // by the Teddy-2 path, which the cascade falls back to.)
            if s2 <= s1 || s2 == accept_state || s2 > accept_state {
                continue;
            }
            c2_set.push(c2);
            let row_s2 = usize::from(s2) * 256;
            for c3 in 0..=u8::MAX {
                let s3 = transitions[row_s2 + usize::from(c3)];
                // c3 strictly advances OR is the escape code (safety).
                let advances = s3 > s2;
                let escape = c3 == fsst::ESCAPE_CODE;
                if (advances || escape) && !c3_seen[usize::from(c3)] {
                    c3_seen[usize::from(c3)] = true;
                    c3_set.push(c3);
                }
            }
        }
        if !c2_set.is_empty() && !c3_set.is_empty() {
            buckets.push((c1, c2_set, c3_set));
        }
    }
    if buckets.is_empty() {
        None
    } else {
        Some(buckets)
    }
}

/// Derive the Teddy-2 remainder that must still run after Teddy-3.
///
/// Teddy-3 keeps only c2 values that advance from `s1` into a normal,
/// non-accept intermediate state with room for one more advancing step. That
/// intentionally excludes valid Teddy-2 cases such as 2-code accepts and
/// escape continuations. This helper subtracts the triple-covered c2 set for
/// each shared `c1`, leaving only the pair-only buckets that preserve
/// correctness when Teddy-3 is active.
pub(super) fn collect_pair_fallback_after_triple(
    pair_buckets: &[(u8, Vec<u8>)],
    triple_buckets: &[(u8, Vec<u8>, Vec<u8>)],
) -> Option<BucketedPairCodes> {
    let mut fallback: BucketedPairCodes = Vec::new();
    for (c1, c2_set) in pair_buckets {
        let Some((_, triple_c2_set, _)) = triple_buckets
            .iter()
            .find(|(triple_c1, ..)| c1 == triple_c1)
        else {
            fallback.push((*c1, c2_set.clone()));
            continue;
        };

        let mut covered = [false; 256];
        for &c2 in triple_c2_set {
            covered[usize::from(c2)] = true;
        }
        let remainder: Vec<u8> = c2_set
            .iter()
            .copied()
            .filter(|&c2| !covered[usize::from(c2)])
            .collect();
        if !remainder.is_empty() {
            fallback.push((*c1, remainder));
        }
    }

    if fallback.is_empty() {
        None
    } else {
        Some(fallback)
    }
}

/// Build a packed bitset of length `all_bytes.len()` whose bit `i` is set
/// iff `(all_bytes[i], all_bytes[i+1], all_bytes[i+2])` is in some
/// bucket's `(c1, c2_set, c3_set)` triple (with the same within-bucket
/// nibble-cross over-approximation as Teddy-2 for the c2 and c3 sets).
/// The last two bits (`i ≥ len - 2`) are forced to 0.
///
/// Single PSHUFB-Mula pass when `buckets.len() ≤ MAX_SET_BYTES`; larger
/// bucket counts are processed in chunks of `MAX_SET_BYTES` and
/// OR-merged into the output.
pub(super) fn build_bucketed_triple_bitset(
    all_bytes: &[u8],
    buckets: &[(u8, Vec<u8>, Vec<u8>)],
) -> Vec<u64> {
    let trace = std::env::var_os("VORTEX_FSST_BUCKET_BUILD_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let total_t = trace.then(std::time::Instant::now);
    let n_words = all_bytes.len().div_ceil(64);
    let alloc_t = trace.then(std::time::Instant::now);
    let mut out = vec![0u64; n_words];
    let alloc_us = alloc_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if buckets.is_empty() || all_bytes.len() < 3 {
        if let Some(total_t) = total_t {
            eprintln!(
                "[fsst::triple_build] path=empty bytes={} buckets={} words={} alloc_us={:.3} fill_us=0.000 merge_us=0.000 total_us={:.3}",
                all_bytes.len(),
                buckets.len(),
                n_words,
                alloc_us,
                total_t.elapsed().as_secs_f64() * 1e6,
            );
        }
        return out;
    }
    if buckets.len() <= MAX_SET_BYTES {
        let fill_t = trace.then(std::time::Instant::now);
        fill_bucketed_triple(all_bytes, buckets, &mut out);
        let fill_us = fill_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        if let Some(total_t) = total_t {
            eprintln!(
                "[fsst::triple_build] path=single bytes={} buckets={} words={} alloc_us={:.3} fill_us={:.3} merge_us=0.000 total_us={:.3}",
                all_bytes.len(),
                buckets.len(),
                n_words,
                alloc_us,
                fill_us,
                total_t.elapsed().as_secs_f64() * 1e6,
            );
        }
        return out;
    }
    let scratch_alloc_t = trace.then(std::time::Instant::now);
    let mut scratch = vec![0u64; n_words];
    let scratch_alloc_us = scratch_alloc_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    let mut fill_us = 0f64;
    let mut merge_us = 0f64;
    for chunk in buckets.chunks(MAX_SET_BYTES) {
        scratch.iter_mut().for_each(|w| *w = 0);
        let fill_t = trace.then(std::time::Instant::now);
        fill_bucketed_triple(all_bytes, chunk, &mut scratch);
        fill_us += fill_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        let merge_t = trace.then(std::time::Instant::now);
        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
            *dst |= *src;
        }
        merge_us += merge_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
    }
    if let Some(total_t) = total_t {
        eprintln!(
            "[fsst::triple_build] path=multipass bytes={} buckets={} words={} alloc_us={:.3} scratch_alloc_us={:.3} fill_us={:.3} merge_us={:.3} total_us={:.3}",
            all_bytes.len(),
            buckets.len(),
            n_words,
            alloc_us,
            scratch_alloc_us,
            fill_us,
            merge_us,
            total_t.elapsed().as_secs_f64() * 1e6,
        );
    }
    out
}

/// Nibble tables for one Teddy-3 pass: bit `b` per bucket across c1/c2/c3.
struct TripleTables {
    c1: NibbleTables,
    c2: NibbleTables,
    c3: NibbleTables,
}

impl TripleTables {
    fn build(buckets: &[(u8, Vec<u8>, Vec<u8>)]) -> Self {
        debug_assert!(buckets.len() <= MAX_SET_BYTES);
        let mut c1_lo = [0u8; 16];
        let mut c1_hi = [0u8; 16];
        let mut c2_lo = [0u8; 16];
        let mut c2_hi = [0u8; 16];
        let mut c3_lo = [0u8; 16];
        let mut c3_hi = [0u8; 16];
        for (b, (c1, c2_set, c3_set)) in buckets.iter().enumerate() {
            let bit = 1u8 << b;
            c1_lo[usize::from(c1 & 0x0F)] |= bit;
            c1_hi[usize::from(c1 >> 4)] |= bit;
            for &c2 in c2_set {
                c2_lo[usize::from(c2 & 0x0F)] |= bit;
                c2_hi[usize::from(c2 >> 4)] |= bit;
            }
            for &c3 in c3_set {
                c3_lo[usize::from(c3 & 0x0F)] |= bit;
                c3_hi[usize::from(c3 >> 4)] |= bit;
            }
        }
        Self {
            c1: NibbleTables {
                lo: c1_lo,
                hi: c1_hi,
            },
            c2: NibbleTables {
                lo: c2_lo,
                hi: c2_hi,
            },
            c3: NibbleTables {
                lo: c3_lo,
                hi: c3_hi,
            },
        }
    }
}

fn fill_bucketed_triple(all_bytes: &[u8], buckets: &[(u8, Vec<u8>, Vec<u8>)], out: &mut [u64]) {
    let tables = TripleTables::build(buckets);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 just detected.
            unsafe { fill_bucketed_triple_avx2(all_bytes, &tables, out) };
            return;
        }
    }
    fill_bucketed_triple_scalar(all_bytes, &tables, out);
}

fn fill_bucketed_triple_scalar(all_bytes: &[u8], tables: &TripleTables, out: &mut [u64]) {
    let len = all_bytes.len();
    if len < 3 {
        return;
    }
    for i in 0..len - 2 {
        let b1 = all_bytes[i];
        let b2 = all_bytes[i + 1];
        let b3 = all_bytes[i + 2];
        let c1_bits = tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
        let c2_bits = tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
        let c3_bits = tables.c3.lo[usize::from(b3 & 0x0F)] & tables.c3.hi[usize::from(b3 >> 4)];
        if (c1_bits & c2_bits & c3_bits) != 0 {
            out[i >> 6] |= 1u64 << (i & 63);
        }
    }
}

/// AVX2 Teddy-3 fill. Same shape as Teddy-2 but with three 32-byte
/// loads per 32-byte step (offsets `i`, `i+1`, `i+2`) and three nibble-
/// table lookups AND'd together before the movemask.
///
/// # Safety
///
/// Requires AVX2 at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn fill_bucketed_triple_avx2(all_bytes: &[u8], tables: &TripleTables, out: &mut [u64]) {
    use core::arch::x86_64::_mm256_or_si256;
    use core::arch::x86_64::_mm256_set1_epi8;

    let len = all_bytes.len();
    if len < 3 {
        return;
    }
    // At i = main_len - 32, the c3 load reads bytes [i+2 .. i+33] = [main_len-30 .. main_len+1].
    // Need main_len + 1 ≤ len - 1 ⇒ main_len ≤ len - 2.
    let main_len = ((len - 2) >> 5) << 5;

    let c1_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.lo.as_ptr() as *const __m128i));
    let c1_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.hi.as_ptr() as *const __m128i));
    let c2_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.lo.as_ptr() as *const __m128i));
    let c2_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.hi.as_ptr() as *const __m128i));
    let c3_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c3.lo.as_ptr() as *const __m128i));
    let c3_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c3.hi.as_ptr() as *const __m128i));
    let zero = _mm256_setzero_si256();
    let nibble_mask = _mm256_set1_epi8(0x0F);

    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    while i < main_len {
        // SAFETY: i + 32 ≤ main_len ≤ len - 2 so i + 33 < len; loads are in bounds.
        let v1 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let v2 = _mm256_loadu_si256(ptr.add(i + 1) as *const __m256i);
        let v3 = _mm256_loadu_si256(ptr.add(i + 2) as *const __m256i);

        let v1_lo = _mm256_and_si256(v1, nibble_mask);
        let v1_hi = _mm256_and_si256(_mm256_srli_epi64(v1, 4), nibble_mask);
        let v2_lo = _mm256_and_si256(v2, nibble_mask);
        let v2_hi = _mm256_and_si256(_mm256_srli_epi64(v2, 4), nibble_mask);
        let v3_lo = _mm256_and_si256(v3, nibble_mask);
        let v3_hi = _mm256_and_si256(_mm256_srli_epi64(v3, 4), nibble_mask);

        let c1_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c1_lo, v1_lo),
            _mm256_shuffle_epi8(c1_hi, v1_hi),
        );
        let c2_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c2_lo, v2_lo),
            _mm256_shuffle_epi8(c2_hi, v2_hi),
        );
        let c3_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c3_lo, v3_lo),
            _mm256_shuffle_epi8(c3_hi, v3_hi),
        );
        let triple = _mm256_and_si256(_mm256_and_si256(c1_bits, c2_bits), c3_bits);

        let pos = _mm256_cmpgt_epi8(triple, zero);
        let neg = _mm256_cmpgt_epi8(zero, triple);
        let hit = _mm256_or_si256(pos, neg);
        let mask = _mm256_movemask_epi8(hit) as u32;

        let word_idx = i >> 6;
        let bit_off = (i & 63) as u64;
        *out.get_unchecked_mut(word_idx) |= u64::from(mask) << bit_off;
        i += 32;
    }

    // Tail: scalar from i to len-3 (inclusive). Positions len-2, len-1 have no
    // 3-byte successor window and must remain 0.
    for j in i..len - 2 {
        let b1 = *all_bytes.get_unchecked(j);
        let b2 = *all_bytes.get_unchecked(j + 1);
        let b3 = *all_bytes.get_unchecked(j + 2);
        let c1_bits_b = tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
        let c2_bits_b = tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
        let c3_bits_b = tables.c3.lo[usize::from(b3 & 0x0F)] & tables.c3.hi[usize::from(b3 >> 4)];
        if (c1_bits_b & c2_bits_b & c3_bits_b) != 0 {
            let word_idx = j >> 6;
            let bit_off = (j & 63) as u64;
            *out.get_unchecked_mut(word_idx) |= 1u64 << bit_off;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused streaming Teddy scans (no materialized bitset)
//
// Replaces "build dense bitset + walk with tzcnt" with a single AVX2 pass
// that emits candidates directly to the DFA verifier. Eliminates a per-chunk
// `Vec<u64>` allocation (~22 KB × thousands of chunks per query) and a
// second pass over `all_bytes`. Empty 32-byte blocks cost ~1 ns thanks to
// the early `mask == 0` short-circuit.
// ---------------------------------------------------------------------------

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;

/// Streaming Teddy-2 + inline DFA verify. Single AVX2 pass; when a
/// 32-byte block has any candidate bits, peel with `tzcnt` and call
/// `verify_at(cand_pos, str_end)` inline. Multi-pass OR-merge when
/// `buckets.len() > MAX_SET_BYTES` (each later pass skips strings
/// already marked by an earlier pass).
///
/// When `ssa_codes` is non-empty, the SSA byte set is fused into the
/// per-block candidate mask via a single extra PSHUFB-Mula nibble
/// lookup on the same 32-byte block load — adding one AVX2 candidate
/// per position whose byte value matches an SSA code, without a
/// separate `all_bytes` pass. The verifier handles SSA candidates
/// identically to Teddy candidates (verify_from_candidate at the
/// position).
pub(super) fn fused_teddy_pair_scan<T, V>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    buckets: &[(u8, Vec<u8>)],
    ssa_codes: Option<&[u8]>,
    negated: bool,
    mut verify_at: V,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    let trace = std::env::var_os("VORTEX_FSST_STREAM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let total_t = trace.then(std::time::Instant::now);
    let alloc_t = trace.then(std::time::Instant::now);
    let mut bits = if negated {
        BitBufferMut::new_set(n)
    } else {
        BitBufferMut::new_unset(n)
    };
    let alloc_us = alloc_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if n == 0 || buckets.is_empty() || all_bytes.len() < 2 {
        return bits.freeze();
    }
    // SSA nibble tables: built once, reused across every bucket chunk.
    // None when there are no SSA codes (`ssa_codes` is None or empty)
    // or when the set exceeds `MAX_SET_BYTES` (FSST trainers rarely
    // mint more than a handful of SSA-eligible symbols, so this
    // ceiling is comfortable in practice).
    let ssa_tables = ssa_codes
        .filter(|codes| !codes.is_empty())
        .and_then(NibbleTables::build);
    let mut table_us = 0f64;
    let mut pass_us = 0f64;
    for (chunk_idx, chunk) in buckets.chunks(MAX_SET_BYTES).enumerate() {
        let table_t = trace.then(std::time::Instant::now);
        let tables = BucketTables::build(chunk);
        table_us += table_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        let pass_t = trace.then(std::time::Instant::now);
        // Fold SSA into the first chunk only — running it again on
        // every chunk would just re-emit the same candidates.
        let ssa_for_chunk = if chunk_idx == 0 {
            ssa_tables.as_ref()
        } else {
            None
        };
        run_teddy_pair_pass(
            &tables,
            ssa_for_chunk,
            n,
            offsets,
            all_bytes,
            negated,
            &mut bits,
            &mut verify_at,
        );
        pass_us += pass_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
    }
    let freeze_t = trace.then(std::time::Instant::now);
    let frozen = bits.freeze();
    let freeze_us = freeze_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if let Some(total_t) = total_t {
        eprintln!(
            "[fsst::stream_total] kind=pair rows={} bytes={} buckets={} chunks={} ssa_codes={} alloc_us={:.3} table_us={:.3} pass_us={:.3} freeze_us={:.3} total_us={:.3}",
            n,
            all_bytes.len(),
            buckets.len(),
            buckets.chunks(MAX_SET_BYTES).count(),
            ssa_codes.map_or(0, |c| c.len()),
            alloc_us,
            table_us,
            pass_us,
            freeze_us,
            total_t.elapsed().as_secs_f64() * 1e6,
        );
    }
    frozen
}

fn run_teddy_pair_pass<T, V>(
    tables: &BucketTables,
    ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 just detected.
            unsafe {
                teddy_pair_pass_avx2(
                    tables, ssa_tables, n, offsets, all_bytes, negated, bits, verify_at,
                )
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on AArch64.
        unsafe {
            teddy_pair_pass_neon(
                tables, ssa_tables, n, offsets, all_bytes, negated, bits, verify_at,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    teddy_pair_pass_scalar(
        tables, ssa_tables, n, offsets, all_bytes, negated, bits, verify_at,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn teddy_pair_pass_avx2<T, V>(
    tables: &BucketTables,
    ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    use core::arch::x86_64::_mm256_or_si256;
    use core::arch::x86_64::_mm256_set1_epi8;
    let trace = std::env::var_os("VORTEX_FSST_STREAM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let pass_t = trace.then(std::time::Instant::now);
    let len = all_bytes.len();
    if len < 2 {
        return;
    }
    let main_len = ((len - 1) >> 5) << 5;
    let setup_t = trace.then(std::time::Instant::now);
    let c1_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.lo.as_ptr() as *const __m128i));
    let c1_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.hi.as_ptr() as *const __m128i));
    let c2_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.lo.as_ptr() as *const __m128i));
    let c2_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.hi.as_ptr() as *const __m128i));
    let zero = _mm256_setzero_si256();
    let nibble_mask = _mm256_set1_epi8(0x0F);
    // SSA nibble tables broadcast to both 128-bit halves so the AVX2
    // PSHUFB lookup runs over the full 32-byte block. When `ssa_tables`
    // is None we splat the zero register into both slots — every PSHUFB
    // returns zero, and the `OR` into the Teddy mask is a no-op. This
    // keeps the per-block code path branch-free (no `is_some` check
    // inside the hot loop) at the cost of two extra vector ops per
    // block, which is negligible.
    let (ssa_lo, ssa_hi, has_ssa) = match ssa_tables {
        Some(t) => (
            _mm256_broadcastsi128_si256(_mm_loadu_si128(t.lo.as_ptr() as *const __m128i)),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(t.hi.as_ptr() as *const __m128i)),
            true,
        ),
        None => (zero, zero, false),
    };
    let setup_us = setup_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    let scan_start: usize = (*offsets.get_unchecked(0)).as_();
    let mut string_idx: usize = 0;
    let mut string_end: usize = (*offsets.get_unchecked(1)).as_();
    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    let mut main_blocks = 0usize;
    let mut nonzero_masks = 0usize;
    let mut candidates = 0usize;
    let mut offset_advances = 0usize;
    let mut already_marked = 0usize;
    let mut verifies = 0usize;
    let mut matches = 0usize;
    let mut bit_writes = 0usize;
    let mut candidate_us = 0f64;
    let mut verify_us = 0f64;
    while i < main_len {
        main_blocks += usize::from(trace);
        let v1 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let v2 = _mm256_loadu_si256(ptr.add(i + 1) as *const __m256i);
        let v1_lo = _mm256_and_si256(v1, nibble_mask);
        let v1_hi = _mm256_and_si256(_mm256_srli_epi64(v1, 4), nibble_mask);
        let v2_lo = _mm256_and_si256(v2, nibble_mask);
        let v2_hi = _mm256_and_si256(_mm256_srli_epi64(v2, 4), nibble_mask);
        let c1_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c1_lo, v1_lo),
            _mm256_shuffle_epi8(c1_hi, v1_hi),
        );
        let c2_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c2_lo, v2_lo),
            _mm256_shuffle_epi8(c2_hi, v2_hi),
        );
        let pair = _mm256_and_si256(c1_bits, c2_bits);
        // Fused SSA lookup on the same v1: bit `b` set in `ssa_bits`
        // iff `v1[lane]`'s nibbles selected bit `b` in both ssa
        // tables — i.e. v1[lane] equals an SSA code value. When
        // has_ssa is false, ssa_lo/ssa_hi are zero so `ssa_bits` is
        // zero and the OR is a no-op.
        let ssa_bits = if has_ssa {
            _mm256_and_si256(
                _mm256_shuffle_epi8(ssa_lo, v1_lo),
                _mm256_shuffle_epi8(ssa_hi, v1_hi),
            )
        } else {
            zero
        };
        let combined = _mm256_or_si256(pair, ssa_bits);
        let pos = _mm256_cmpgt_epi8(combined, zero);
        let neg = _mm256_cmpgt_epi8(zero, combined);
        let hit = _mm256_or_si256(pos, neg);
        let mut mask = _mm256_movemask_epi8(hit) as u32;
        if mask != 0 {
            nonzero_masks += usize::from(trace);
            let candidate_t = trace.then(std::time::Instant::now);
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let cand = i + bit;
                candidates += usize::from(trace);
                if cand < scan_start {
                    continue;
                }
                while cand >= string_end {
                    offset_advances += usize::from(trace);
                    string_idx += 1;
                    if string_idx >= n {
                        if let Some(pass_t) = pass_t {
                            let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
                            eprintln!(
                                "[fsst::stream_pass] kind=pair impl=avx2 bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} tail_positions=0 tail_candidates=0 offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us=0.000 vector_us={:.3} total_us={:.3}",
                                len,
                                n,
                                main_blocks,
                                nonzero_masks,
                                candidates,
                                offset_advances,
                                already_marked,
                                verifies,
                                matches,
                                bit_writes,
                                setup_us,
                                candidate_us,
                                verify_us,
                                pass_us - setup_us - candidate_us,
                                pass_us,
                            );
                        }
                        return;
                    }
                    string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
                }
                let already = bits.value(string_idx);
                if (!negated && already) || (negated && !already) {
                    already_marked += usize::from(trace);
                    continue;
                }
                verifies += usize::from(trace);
                let verify_t = trace.then(std::time::Instant::now);
                let accepted = verify_at(cand, string_end);
                verify_us += verify_t
                    .map(|t| t.elapsed().as_secs_f64() * 1e6)
                    .unwrap_or_default();
                if accepted {
                    matches += usize::from(trace);
                    bit_writes += usize::from(trace);
                    if negated {
                        bits.unset_unchecked(string_idx);
                    } else {
                        bits.set_unchecked(string_idx);
                    }
                }
            }
            candidate_us += candidate_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
        }
        i += 32;
    }
    // Tail scalar
    let tail_t = trace.then(std::time::Instant::now);
    let mut tail_positions = 0usize;
    let mut tail_candidates = 0usize;
    if len > 1 {
        for j in i..len - 1 {
            tail_positions += usize::from(trace);
            let b1 = *all_bytes.get_unchecked(j);
            let b2 = *all_bytes.get_unchecked(j + 1);
            let c1_bits_b =
                tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
            let c2_bits_b =
                tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
            let pair_hit = (c1_bits_b & c2_bits_b) != 0;
            let ssa_hit = ssa_tables
                .is_some_and(|t| (t.lo[usize::from(b1 & 0x0F)] & t.hi[usize::from(b1 >> 4)]) != 0);
            if !pair_hit && !ssa_hit {
                continue;
            }
            tail_candidates += usize::from(trace);
            let cand = j;
            if cand < scan_start {
                continue;
            }
            while cand >= string_end {
                offset_advances += usize::from(trace);
                string_idx += 1;
                if string_idx >= n {
                    break;
                }
                string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
            }
            if string_idx >= n {
                break;
            }
            let already = bits.value(string_idx);
            if (!negated && already) || (negated && !already) {
                already_marked += usize::from(trace);
                continue;
            }
            verifies += usize::from(trace);
            let verify_t = trace.then(std::time::Instant::now);
            let accepted = verify_at(cand, string_end);
            verify_us += verify_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            if accepted {
                matches += usize::from(trace);
                bit_writes += usize::from(trace);
                if negated {
                    bits.unset_unchecked(string_idx);
                } else {
                    bits.set_unchecked(string_idx);
                }
            }
        }
        // SSA can still match at position `len - 1` (no successor needed
        // for a single-byte check). The pair tail loop above stops at
        // `len - 2`, so check the last position separately when SSA is
        // enabled.
        if let Some(t) = ssa_tables {
            let j = len - 1;
            let b = *all_bytes.get_unchecked(j);
            let ssa_hit = (t.lo[usize::from(b & 0x0F)] & t.hi[usize::from(b >> 4)]) != 0;
            if ssa_hit && j >= scan_start {
                while j >= string_end && string_idx < n {
                    string_idx += 1;
                    if string_idx < n {
                        string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
                    }
                }
                if string_idx < n {
                    let already = bits.value(string_idx);
                    // already in "decided match" state? Skip.
                    //   non-negated: bit=1 means match.
                    //   negated:     bit=0 means match.
                    let already_match = if negated { !already } else { already };
                    if !already_match && verify_at(j, string_end) {
                        if negated {
                            bits.unset_unchecked(string_idx);
                        } else {
                            bits.set_unchecked(string_idx);
                        }
                    }
                }
            }
        }
    }
    let tail_us = tail_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if let Some(pass_t) = pass_t {
        let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
        eprintln!(
            "[fsst::stream_pass] kind=pair impl=avx2 bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} tail_positions={} tail_candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us={:.3} vector_us={:.3} total_us={:.3}",
            len,
            n,
            main_blocks,
            nonzero_masks,
            candidates,
            tail_positions,
            tail_candidates,
            offset_advances,
            already_marked,
            verifies,
            matches,
            bit_writes,
            setup_us,
            candidate_us,
            verify_us,
            tail_us,
            pass_us - setup_us - candidate_us - tail_us,
            pass_us,
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn teddy_pair_pass_neon<T, V>(
    tables: &BucketTables,
    // TODO: NEON SSA fusion. For now the NEON path doesn't handle the
    // SSA set inline; on AArch64 the caller falls back to the
    // non-fused 1-byte path when SSA codes exist, so correctness is
    // preserved.
    _ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    let trace = std::env::var_os("VORTEX_FSST_STREAM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let pass_t = trace.then(std::time::Instant::now);
    let len = all_bytes.len();
    if len < 2 {
        return;
    }
    let main_len = ((len - 1) >> 4) << 4;
    let setup_t = trace.then(std::time::Instant::now);
    let c1_lo = vld1q_u8(tables.c1.lo.as_ptr());
    let c1_hi = vld1q_u8(tables.c1.hi.as_ptr());
    let c2_lo = vld1q_u8(tables.c2.lo.as_ptr());
    let c2_hi = vld1q_u8(tables.c2.hi.as_ptr());
    let zero = vdupq_n_u8(0);
    let nibble_mask = vdupq_n_u8(0x0F);
    let lane_bits = vld1q_u8(NEON_MOVEMASK_BITS.as_ptr());
    let setup_us = setup_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    let scan_start: usize = (*offsets.get_unchecked(0)).as_();
    let mut string_idx: usize = 0;
    let mut string_end: usize = (*offsets.get_unchecked(1)).as_();
    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    let mut main_blocks = 0usize;
    let mut nonzero_masks = 0usize;
    let mut candidates = 0usize;
    let mut offset_advances = 0usize;
    let mut already_marked = 0usize;
    let mut verifies = 0usize;
    let mut matches = 0usize;
    let mut bit_writes = 0usize;
    let mut candidate_us = 0f64;
    let mut verify_us = 0f64;
    while i < main_len {
        main_blocks += usize::from(trace);
        let v1 = vld1q_u8(ptr.add(i));
        let v2 = vld1q_u8(ptr.add(i + 1));
        let c1_bits = neon_nibble_lookup(c1_lo, c1_hi, v1, nibble_mask);
        let c2_bits = neon_nibble_lookup(c2_lo, c2_hi, v2, nibble_mask);
        let pair = vandq_u8(c1_bits, c2_bits);
        let mut mask = u32::from(neon_nonzero_mask(pair, zero, lane_bits));
        if mask != 0 {
            nonzero_masks += usize::from(trace);
            let candidate_t = trace.then(std::time::Instant::now);
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let cand = i + bit;
                candidates += usize::from(trace);
                if cand < scan_start {
                    continue;
                }
                while cand >= string_end {
                    offset_advances += usize::from(trace);
                    string_idx += 1;
                    if string_idx >= n {
                        if let Some(pass_t) = pass_t {
                            let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
                            eprintln!(
                                "[fsst::stream_pass] kind=pair impl=neon bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} tail_positions=0 tail_candidates=0 offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us=0.000 vector_us={:.3} total_us={:.3}",
                                len,
                                n,
                                main_blocks,
                                nonzero_masks,
                                candidates,
                                offset_advances,
                                already_marked,
                                verifies,
                                matches,
                                bit_writes,
                                setup_us,
                                candidate_us,
                                verify_us,
                                pass_us - setup_us - candidate_us,
                                pass_us,
                            );
                        }
                        return;
                    }
                    string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
                }
                let already = bits.value(string_idx);
                if (!negated && already) || (negated && !already) {
                    already_marked += usize::from(trace);
                    continue;
                }
                verifies += usize::from(trace);
                let verify_t = trace.then(std::time::Instant::now);
                let accepted = verify_at(cand, string_end);
                verify_us += verify_t
                    .map(|t| t.elapsed().as_secs_f64() * 1e6)
                    .unwrap_or_default();
                if accepted {
                    matches += usize::from(trace);
                    bit_writes += usize::from(trace);
                    if negated {
                        bits.unset_unchecked(string_idx);
                    } else {
                        bits.set_unchecked(string_idx);
                    }
                }
            }
            candidate_us += candidate_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
        }
        i += 16;
    }
    let tail_t = trace.then(std::time::Instant::now);
    let mut tail_positions = 0usize;
    let mut tail_candidates = 0usize;
    if len > 1 {
        for j in i..len - 1 {
            tail_positions += usize::from(trace);
            let b1 = *all_bytes.get_unchecked(j);
            let b2 = *all_bytes.get_unchecked(j + 1);
            let c1_bits_b =
                tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
            let c2_bits_b =
                tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
            if (c1_bits_b & c2_bits_b) == 0 {
                continue;
            }
            tail_candidates += usize::from(trace);
            let cand = j;
            if cand < scan_start {
                continue;
            }
            while cand >= string_end {
                offset_advances += usize::from(trace);
                string_idx += 1;
                if string_idx >= n {
                    break;
                }
                string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
            }
            if string_idx >= n {
                break;
            }
            let already = bits.value(string_idx);
            if (!negated && already) || (negated && !already) {
                already_marked += usize::from(trace);
                continue;
            }
            verifies += usize::from(trace);
            let verify_t = trace.then(std::time::Instant::now);
            let accepted = verify_at(cand, string_end);
            verify_us += verify_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            if accepted {
                matches += usize::from(trace);
                bit_writes += usize::from(trace);
                if negated {
                    bits.unset_unchecked(string_idx);
                } else {
                    bits.set_unchecked(string_idx);
                }
            }
        }
    }
    let tail_us = tail_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if let Some(pass_t) = pass_t {
        let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
        eprintln!(
            "[fsst::stream_pass] kind=pair impl=neon bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} tail_positions={} tail_candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us={:.3} vector_us={:.3} total_us={:.3}",
            len,
            n,
            main_blocks,
            nonzero_masks,
            candidates,
            tail_positions,
            tail_candidates,
            offset_advances,
            already_marked,
            verifies,
            matches,
            bit_writes,
            setup_us,
            candidate_us,
            verify_us,
            tail_us,
            pass_us - setup_us - candidate_us - tail_us,
            pass_us,
        );
    }
}

fn teddy_pair_pass_scalar<T, V>(
    tables: &BucketTables,
    ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    let len = all_bytes.len();
    if len < 2 {
        return;
    }
    let scan_start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
    let mut string_idx: usize = 0;
    let mut string_end: usize = unsafe { *offsets.get_unchecked(1) }.as_();
    for j in 0..len - 1 {
        let b1 = all_bytes[j];
        let b2 = all_bytes[j + 1];
        let c1_bits = tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
        let c2_bits = tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
        let pair_hit = (c1_bits & c2_bits) != 0;
        let ssa_hit = ssa_tables
            .is_some_and(|t| (t.lo[usize::from(b1 & 0x0F)] & t.hi[usize::from(b1 >> 4)]) != 0);
        if !pair_hit && !ssa_hit {
            continue;
        }
        let cand = j;
        if cand < scan_start {
            continue;
        }
        while cand >= string_end {
            string_idx += 1;
            if string_idx >= n {
                return;
            }
            string_end = unsafe { *offsets.get_unchecked(string_idx + 1) }.as_();
        }
        let already = bits.value(string_idx);
        if (!negated && already) || (negated && !already) {
            continue;
        }
        if verify_at(cand, string_end) {
            if negated {
                unsafe { bits.unset_unchecked(string_idx) };
            } else {
                unsafe { bits.set_unchecked(string_idx) };
            }
        }
    }
    // Last position SSA-only candidate (no successor for the pair check).
    if let Some(t) = ssa_tables {
        let j = len - 1;
        let b = all_bytes[j];
        let ssa_hit = (t.lo[usize::from(b & 0x0F)] & t.hi[usize::from(b >> 4)]) != 0;
        if ssa_hit && j >= scan_start {
            while j >= string_end {
                string_idx += 1;
                if string_idx >= n {
                    return;
                }
                string_end = unsafe { *offsets.get_unchecked(string_idx + 1) }.as_();
            }
            let already = bits.value(string_idx);
            let already_match = if negated { !already } else { already };
            if !already_match && verify_at(j, string_end) {
                if negated {
                    unsafe { bits.unset_unchecked(string_idx) };
                } else {
                    unsafe { bits.set_unchecked(string_idx) };
                }
            }
        }
    }
}

/// Specialized streaming pair scan for the common "only `ESCAPE_CODE` can
/// start the match" shape with a tiny `c2` set. This replaces the generic
/// Teddy-2 byte-stream pass with one exact 2-byte search per candidate
/// `ESCAPE_CODE + c2` pair.
pub(super) fn fused_escape_pair_scan<T, V>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    c2_codes: &[u8],
    negated: bool,
    mut verify_at: V,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    let trace = std::env::var_os("VORTEX_FSST_STREAM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let total_t = trace.then(std::time::Instant::now);
    let alloc_t = trace.then(std::time::Instant::now);
    let mut bits = if negated {
        BitBufferMut::new_set(n)
    } else {
        BitBufferMut::new_unset(n)
    };
    let alloc_us = alloc_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();

    if n == 0 || c2_codes.is_empty() || all_bytes.len() < 2 {
        return bits.freeze();
    }

    let table_t = trace.then(std::time::Instant::now);
    let scan_start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
    let scan_slice = &all_bytes[scan_start..];
    let mut candidates_pos = Vec::new();
    for &c2 in c2_codes {
        let needle = [fsst::ESCAPE_CODE, c2];
        candidates_pos.extend(
            memchr::memmem::find_iter(scan_slice, &needle).map(|cand_rel| scan_start + cand_rel),
        );
    }
    candidates_pos.sort_unstable();
    candidates_pos.dedup();
    let table_us = table_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();

    let scan_t = trace.then(std::time::Instant::now);
    let mut string_idx: usize = 0;
    let mut string_end: usize = unsafe { *offsets.get_unchecked(1) }.as_();
    let mut candidates = 0usize;
    let mut offset_advances = 0usize;
    let mut already_marked = 0usize;
    let mut verifies = 0usize;
    let mut matches = 0usize;
    let mut bit_writes = 0usize;
    let mut verify_us = 0f64;

    for cand in candidates_pos {
        candidates += usize::from(trace);

        while cand >= string_end {
            offset_advances += usize::from(trace);
            string_idx += 1;
            if string_idx >= n {
                let frozen = bits.freeze();
                if let Some(total_t) = total_t {
                    let total_us = total_t.elapsed().as_secs_f64() * 1e6;
                    let scan_us = scan_t
                        .map(|t| t.elapsed().as_secs_f64() * 1e6)
                        .unwrap_or_default();
                    eprintln!(
                        "[fsst::stream_total] kind=escape_pair rows={} bytes={} c2_codes={} alloc_us={:.3} table_us={:.3} scan_us={:.3} candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} verify_us={:.3} total_us={:.3}",
                        n,
                        all_bytes.len(),
                        c2_codes.len(),
                        alloc_us,
                        table_us,
                        scan_us,
                        candidates,
                        offset_advances,
                        already_marked,
                        verifies,
                        matches,
                        bit_writes,
                        verify_us,
                        total_us,
                    );
                }
                return frozen;
            }
            string_end = unsafe { *offsets.get_unchecked(string_idx + 1) }.as_();
        }

        let already = bits.value(string_idx);
        if (!negated && already) || (negated && !already) {
            already_marked += usize::from(trace);
            continue;
        }

        verifies += usize::from(trace);
        let verify_t = trace.then(std::time::Instant::now);
        if verify_at(cand, string_end) {
            matches += usize::from(trace);
            bit_writes += usize::from(trace);
            unsafe {
                if negated {
                    bits.unset_unchecked(string_idx);
                } else {
                    bits.set_unchecked(string_idx);
                }
            }
        }
        verify_us += verify_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
    }

    let frozen = bits.freeze();
    if let Some(total_t) = total_t {
        let total_us = total_t.elapsed().as_secs_f64() * 1e6;
        let scan_us = scan_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        eprintln!(
            "[fsst::stream_total] kind=escape_pair rows={} bytes={} c2_codes={} alloc_us={:.3} table_us={:.3} scan_us={:.3} candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} verify_us={:.3} total_us={:.3}",
            n,
            all_bytes.len(),
            c2_codes.len(),
            alloc_us,
            table_us,
            scan_us,
            candidates,
            offset_advances,
            already_marked,
            verifies,
            matches,
            bit_writes,
            verify_us,
            total_us,
        );
    }
    frozen
}

/// Streaming Teddy-3 + inline verify.
pub(super) fn fused_teddy_triple_scan<T, V>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    buckets: &[(u8, Vec<u8>, Vec<u8>)],
    ssa_codes: Option<&[u8]>,
    negated: bool,
    mut verify_at: V,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    let trace = std::env::var_os("VORTEX_FSST_STREAM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let total_t = trace.then(std::time::Instant::now);
    let alloc_t = trace.then(std::time::Instant::now);
    let mut bits = if negated {
        BitBufferMut::new_set(n)
    } else {
        BitBufferMut::new_unset(n)
    };
    let alloc_us = alloc_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if n == 0 || buckets.is_empty() || all_bytes.len() < 3 {
        return bits.freeze();
    }
    let ssa_tables = ssa_codes
        .filter(|codes| !codes.is_empty())
        .and_then(NibbleTables::build);
    let mut table_us = 0f64;
    let mut pass_us = 0f64;
    for (chunk_idx, chunk) in buckets.chunks(MAX_SET_BYTES).enumerate() {
        let table_t = trace.then(std::time::Instant::now);
        let tables = TripleTables::build(chunk);
        table_us += table_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        let pass_t = trace.then(std::time::Instant::now);
        let ssa_for_chunk = if chunk_idx == 0 {
            ssa_tables.as_ref()
        } else {
            None
        };
        run_teddy_triple_pass(
            &tables,
            ssa_for_chunk,
            n,
            offsets,
            all_bytes,
            negated,
            &mut bits,
            &mut verify_at,
        );
        pass_us += pass_t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
    }
    let freeze_t = trace.then(std::time::Instant::now);
    let frozen = bits.freeze();
    let freeze_us = freeze_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if let Some(total_t) = total_t {
        eprintln!(
            "[fsst::stream_total] kind=triple rows={} bytes={} buckets={} chunks={} ssa_codes={} alloc_us={:.3} table_us={:.3} pass_us={:.3} freeze_us={:.3} total_us={:.3}",
            n,
            all_bytes.len(),
            buckets.len(),
            buckets.chunks(MAX_SET_BYTES).count(),
            ssa_codes.map_or(0, |c| c.len()),
            alloc_us,
            table_us,
            pass_us,
            freeze_us,
            total_t.elapsed().as_secs_f64() * 1e6,
        );
    }
    frozen
}

fn run_teddy_triple_pass<T, V>(
    tables: &TripleTables,
    ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F+BW just detected.
            unsafe {
                teddy_triple_pass_avx512(
                    tables, ssa_tables, n, offsets, all_bytes, negated, bits, verify_at,
                )
            };
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 just detected.
            unsafe {
                teddy_triple_pass_avx2(
                    tables, ssa_tables, n, offsets, all_bytes, negated, bits, verify_at,
                )
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on AArch64.
        unsafe {
            teddy_triple_pass_neon(
                tables, ssa_tables, n, offsets, all_bytes, negated, bits, verify_at,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    teddy_triple_pass_scalar(
        tables, ssa_tables, n, offsets, all_bytes, negated, bits, verify_at,
    );
}

/// AVX-512 streaming Teddy-3. Processes 64 input bytes per iteration:
/// three 64-byte loads → three pshufb-Mula nibble lookups (per-128-lane,
/// table broadcast to all 4 lanes) → one `vpternlogq` to fuse the
/// three-way AND of (c1_bits, c2_bits, c3_bits) → `vpcmpneqb` to produce
/// a 64-bit candidate mask directly (no `cmpgt | cmpgt` pair).
///
/// ~2× the throughput of the AVX2 path on AVX-512 parts (modulo memory
/// bandwidth limits), and the `vpternlogq` saves one instruction in the
/// hot inner loop vs the AVX2 version's two ANDs.
///
/// # Safety
///
/// Requires `avx512f` and `avx512bw` at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn teddy_triple_pass_avx512<T, V>(
    tables: &TripleTables,
    ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    use core::arch::x86_64::__m512i;
    use core::arch::x86_64::_mm512_and_si512;
    use core::arch::x86_64::_mm512_broadcast_i32x4;
    use core::arch::x86_64::_mm512_cmpneq_epi8_mask;
    use core::arch::x86_64::_mm512_loadu_si512;
    use core::arch::x86_64::_mm512_or_si512;
    use core::arch::x86_64::_mm512_set1_epi8;
    use core::arch::x86_64::_mm512_setzero_si512;
    use core::arch::x86_64::_mm512_shuffle_epi8;
    use core::arch::x86_64::_mm512_srli_epi64;
    use core::arch::x86_64::_mm512_ternarylogic_epi64;

    let trace = std::env::var_os("VORTEX_FSST_STREAM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let pass_t = trace.then(std::time::Instant::now);
    let len = all_bytes.len();
    if len < 3 {
        return;
    }
    // Largest multiple of 64 such that the c3 load at offset main_len-64+2
    // reads bytes ending at main_len+1 < len.
    let main_len = ((len - 2) >> 6) << 6;

    let setup_t = trace.then(std::time::Instant::now);
    let c1_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.c1.lo.as_ptr() as *const __m128i));
    let c1_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.c1.hi.as_ptr() as *const __m128i));
    let c2_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.c2.lo.as_ptr() as *const __m128i));
    let c2_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.c2.hi.as_ptr() as *const __m128i));
    let c3_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.c3.lo.as_ptr() as *const __m128i));
    let c3_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.c3.hi.as_ptr() as *const __m128i));
    let nibble_mask = _mm512_set1_epi8(0x0F);
    let zero = _mm512_setzero_si512();
    // Fused SSA nibble tables (broadcast to all 4 lanes). When absent,
    // zero tables ⇒ PSHUFB outputs zero ⇒ OR is a no-op.
    let (ssa_lo, ssa_hi, has_ssa) = match ssa_tables {
        Some(t) => (
            _mm512_broadcast_i32x4(_mm_loadu_si128(t.lo.as_ptr() as *const __m128i)),
            _mm512_broadcast_i32x4(_mm_loadu_si128(t.hi.as_ptr() as *const __m128i)),
            true,
        ),
        None => (zero, zero, false),
    };
    let setup_us = setup_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();

    let scan_start: usize = (*offsets.get_unchecked(0)).as_();
    let mut string_idx: usize = 0;
    let mut string_end: usize = (*offsets.get_unchecked(1)).as_();
    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    let mut main_blocks = 0usize;
    let mut nonzero_masks = 0usize;
    let mut candidates = 0usize;
    let mut offset_advances = 0usize;
    let mut already_marked = 0usize;
    let mut verifies = 0usize;
    let mut matches = 0usize;
    let mut bit_writes = 0usize;
    let mut candidate_us = 0f64;
    let mut verify_us = 0f64;
    while i < main_len {
        main_blocks += usize::from(trace);
        let v1 = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
        let v2 = _mm512_loadu_si512(ptr.add(i + 1) as *const __m512i);
        let v3 = _mm512_loadu_si512(ptr.add(i + 2) as *const __m512i);

        let v1_lo = _mm512_and_si512(v1, nibble_mask);
        let v1_hi = _mm512_and_si512(_mm512_srli_epi64(v1, 4), nibble_mask);
        let v2_lo = _mm512_and_si512(v2, nibble_mask);
        let v2_hi = _mm512_and_si512(_mm512_srli_epi64(v2, 4), nibble_mask);
        let v3_lo = _mm512_and_si512(v3, nibble_mask);
        let v3_hi = _mm512_and_si512(_mm512_srli_epi64(v3, 4), nibble_mask);

        let c1_bits = _mm512_and_si512(
            _mm512_shuffle_epi8(c1_lo, v1_lo),
            _mm512_shuffle_epi8(c1_hi, v1_hi),
        );
        let c2_bits = _mm512_and_si512(
            _mm512_shuffle_epi8(c2_lo, v2_lo),
            _mm512_shuffle_epi8(c2_hi, v2_hi),
        );
        let c3_bits = _mm512_and_si512(
            _mm512_shuffle_epi8(c3_lo, v3_lo),
            _mm512_shuffle_epi8(c3_hi, v3_hi),
        );
        // vpternlogq with imm 0x80 = A AND B AND C (truth table for 1110 0000 = bit 7).
        let triple = _mm512_ternarylogic_epi64(c1_bits, c2_bits, c3_bits, 0x80);

        // Fused SSA on v1: PSHUFB on v1's nibbles via the SSA tables.
        // When `has_ssa` is false the tables are zero, the lookup is
        // zero, and the OR is a no-op.
        let combined = if has_ssa {
            let ssa_bits = _mm512_and_si512(
                _mm512_shuffle_epi8(ssa_lo, v1_lo),
                _mm512_shuffle_epi8(ssa_hi, v1_hi),
            );
            _mm512_or_si512(triple, ssa_bits)
        } else {
            triple
        };

        // vpcmpneqb: 1-bit-per-byte, directly into a 64-bit kmask.
        let mut mask: u64 = _mm512_cmpneq_epi8_mask(combined, zero);
        if mask != 0 {
            nonzero_masks += usize::from(trace);
            let candidate_t = trace.then(std::time::Instant::now);
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let cand = i + bit;
                candidates += usize::from(trace);
                if cand < scan_start {
                    continue;
                }
                while cand >= string_end {
                    offset_advances += usize::from(trace);
                    string_idx += 1;
                    if string_idx >= n {
                        if let Some(pass_t) = pass_t {
                            let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
                            let tail_us = 0f64;
                            eprintln!(
                                "[fsst::stream_pass] kind=triple impl=avx512 bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us={:.3} vector_us={:.3} total_us={:.3}",
                                len,
                                n,
                                main_blocks,
                                nonzero_masks,
                                candidates,
                                offset_advances,
                                already_marked,
                                verifies,
                                matches,
                                bit_writes,
                                setup_us,
                                candidate_us,
                                verify_us,
                                tail_us,
                                pass_us - setup_us - candidate_us - tail_us,
                                pass_us,
                            );
                        }
                        return;
                    }
                    string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
                }
                let already = bits.value(string_idx);
                if (!negated && already) || (negated && !already) {
                    already_marked += usize::from(trace);
                    continue;
                }
                verifies += usize::from(trace);
                let verify_t = trace.then(std::time::Instant::now);
                let accepted = verify_at(cand, string_end);
                verify_us += verify_t
                    .map(|t| t.elapsed().as_secs_f64() * 1e6)
                    .unwrap_or_default();
                if accepted {
                    matches += usize::from(trace);
                    bit_writes += usize::from(trace);
                    if negated {
                        bits.unset_unchecked(string_idx);
                    } else {
                        bits.set_unchecked(string_idx);
                    }
                }
            }
            candidate_us += candidate_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
        }
        i += 64;
    }
    // Tail: scalar bytes from `i` up to `len - 3` (inclusive) cover any
    // positions left over by the 64-byte main loop.
    let tail_t = trace.then(std::time::Instant::now);
    let mut tail_positions = 0usize;
    let mut tail_candidates = 0usize;
    if len > 2 {
        for j in i..len - 2 {
            tail_positions += usize::from(trace);
            let b1 = *all_bytes.get_unchecked(j);
            let b2 = *all_bytes.get_unchecked(j + 1);
            let b3 = *all_bytes.get_unchecked(j + 2);
            let c1_bits_b =
                tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
            let c2_bits_b =
                tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
            let c3_bits_b =
                tables.c3.lo[usize::from(b3 & 0x0F)] & tables.c3.hi[usize::from(b3 >> 4)];
            let triple_hit = (c1_bits_b & c2_bits_b & c3_bits_b) != 0;
            let ssa_hit = ssa_tables
                .is_some_and(|t| (t.lo[usize::from(b1 & 0x0F)] & t.hi[usize::from(b1 >> 4)]) != 0);
            if !triple_hit && !ssa_hit {
                continue;
            }
            tail_candidates += usize::from(trace);
            let cand = j;
            if cand < scan_start {
                continue;
            }
            while cand >= string_end {
                offset_advances += usize::from(trace);
                string_idx += 1;
                if string_idx >= n {
                    break;
                }
                string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
            }
            if string_idx >= n {
                break;
            }
            let already = bits.value(string_idx);
            if (!negated && already) || (negated && !already) {
                already_marked += usize::from(trace);
                continue;
            }
            verifies += usize::from(trace);
            let verify_t = trace.then(std::time::Instant::now);
            let accepted = verify_at(cand, string_end);
            verify_us += verify_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            if accepted {
                matches += usize::from(trace);
                bit_writes += usize::from(trace);
                if negated {
                    bits.unset_unchecked(string_idx);
                } else {
                    bits.set_unchecked(string_idx);
                }
            }
        }
        // SSA-only candidates at positions `len - 2` and `len - 1` (the
        // triple tail loop stops at `len - 3`).
        if let Some(t) = ssa_tables {
            for &j in &[len.saturating_sub(2), len - 1] {
                if j < i || j >= len {
                    continue;
                }
                let b = *all_bytes.get_unchecked(j);
                if (t.lo[usize::from(b & 0x0F)] & t.hi[usize::from(b >> 4)]) == 0 {
                    continue;
                }
                let cand = j;
                if cand < scan_start {
                    continue;
                }
                while cand >= string_end {
                    string_idx += 1;
                    if string_idx >= n {
                        break;
                    }
                    string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
                }
                if string_idx >= n {
                    break;
                }
                let already = bits.value(string_idx);
                if (!negated && already) || (negated && !already) {
                    continue;
                }
                if verify_at(cand, string_end) {
                    if negated {
                        bits.unset_unchecked(string_idx);
                    } else {
                        bits.set_unchecked(string_idx);
                    }
                }
            }
        }
    }
    let tail_us = tail_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if let Some(pass_t) = pass_t {
        let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
        eprintln!(
            "[fsst::stream_pass] kind=triple impl=avx512 bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} tail_positions={} tail_candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us={:.3} vector_us={:.3} total_us={:.3}",
            len,
            n,
            main_blocks,
            nonzero_masks,
            candidates,
            tail_positions,
            tail_candidates,
            offset_advances,
            already_marked,
            verifies,
            matches,
            bit_writes,
            setup_us,
            candidate_us,
            verify_us,
            tail_us,
            pass_us - setup_us - candidate_us - tail_us,
            pass_us,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn teddy_triple_pass_avx2<T, V>(
    tables: &TripleTables,
    // TODO: AVX2 SSA fusion. AVX-512 (preferred path on AVX-512 parts)
    // already handles SSA; AVX2 falls back to no SSA, which on
    // SSA-present needles is corrected by the post-Teddy merge in the
    // caller. Kept identical to the original AVX2 logic for now.
    _ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    use core::arch::x86_64::_mm256_or_si256;
    use core::arch::x86_64::_mm256_set1_epi8;
    let len = all_bytes.len();
    if len < 3 {
        return;
    }
    let main_len = ((len - 2) >> 5) << 5;
    let c1_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.lo.as_ptr() as *const __m128i));
    let c1_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1.hi.as_ptr() as *const __m128i));
    let c2_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.lo.as_ptr() as *const __m128i));
    let c2_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2.hi.as_ptr() as *const __m128i));
    let c3_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c3.lo.as_ptr() as *const __m128i));
    let c3_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c3.hi.as_ptr() as *const __m128i));
    let zero = _mm256_setzero_si256();
    let nibble_mask = _mm256_set1_epi8(0x0F);
    let scan_start: usize = (*offsets.get_unchecked(0)).as_();
    let mut string_idx: usize = 0;
    let mut string_end: usize = (*offsets.get_unchecked(1)).as_();
    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    while i < main_len {
        let v1 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let v2 = _mm256_loadu_si256(ptr.add(i + 1) as *const __m256i);
        let v3 = _mm256_loadu_si256(ptr.add(i + 2) as *const __m256i);
        let v1_lo = _mm256_and_si256(v1, nibble_mask);
        let v1_hi = _mm256_and_si256(_mm256_srli_epi64(v1, 4), nibble_mask);
        let v2_lo = _mm256_and_si256(v2, nibble_mask);
        let v2_hi = _mm256_and_si256(_mm256_srli_epi64(v2, 4), nibble_mask);
        let v3_lo = _mm256_and_si256(v3, nibble_mask);
        let v3_hi = _mm256_and_si256(_mm256_srli_epi64(v3, 4), nibble_mask);
        let c1_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c1_lo, v1_lo),
            _mm256_shuffle_epi8(c1_hi, v1_hi),
        );
        let c2_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c2_lo, v2_lo),
            _mm256_shuffle_epi8(c2_hi, v2_hi),
        );
        let c3_bits = _mm256_and_si256(
            _mm256_shuffle_epi8(c3_lo, v3_lo),
            _mm256_shuffle_epi8(c3_hi, v3_hi),
        );
        let triple = _mm256_and_si256(_mm256_and_si256(c1_bits, c2_bits), c3_bits);
        let pos = _mm256_cmpgt_epi8(triple, zero);
        let neg = _mm256_cmpgt_epi8(zero, triple);
        let hit = _mm256_or_si256(pos, neg);
        let mut mask = _mm256_movemask_epi8(hit) as u32;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let cand = i + bit;
            if cand < scan_start {
                continue;
            }
            while cand >= string_end {
                string_idx += 1;
                if string_idx >= n {
                    return;
                }
                string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
            }
            let already = bits.value(string_idx);
            if (!negated && already) || (negated && !already) {
                continue;
            }
            if verify_at(cand, string_end) {
                if negated {
                    bits.unset_unchecked(string_idx);
                } else {
                    bits.set_unchecked(string_idx);
                }
            }
        }
        i += 32;
    }
    if len > 2 {
        for j in i..len - 2 {
            let b1 = *all_bytes.get_unchecked(j);
            let b2 = *all_bytes.get_unchecked(j + 1);
            let b3 = *all_bytes.get_unchecked(j + 2);
            let c1_bits_b =
                tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
            let c2_bits_b =
                tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
            let c3_bits_b =
                tables.c3.lo[usize::from(b3 & 0x0F)] & tables.c3.hi[usize::from(b3 >> 4)];
            if (c1_bits_b & c2_bits_b & c3_bits_b) == 0 {
                continue;
            }
            let cand = j;
            if cand < scan_start {
                continue;
            }
            while cand >= string_end {
                string_idx += 1;
                if string_idx >= n {
                    return;
                }
                string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
            }
            let already = bits.value(string_idx);
            if (!negated && already) || (negated && !already) {
                continue;
            }
            if verify_at(cand, string_end) {
                if negated {
                    bits.unset_unchecked(string_idx);
                } else {
                    bits.set_unchecked(string_idx);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn teddy_triple_pass_neon<T, V>(
    tables: &TripleTables,
    // TODO: NEON SSA fusion. See `teddy_pair_pass_neon`.
    _ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    let trace = std::env::var_os("VORTEX_FSST_STREAM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let pass_t = trace.then(std::time::Instant::now);
    let len = all_bytes.len();
    if len < 3 {
        return;
    }
    let main_len = ((len - 2) >> 4) << 4;
    let setup_t = trace.then(std::time::Instant::now);
    let c1_lo = vld1q_u8(tables.c1.lo.as_ptr());
    let c1_hi = vld1q_u8(tables.c1.hi.as_ptr());
    let c2_lo = vld1q_u8(tables.c2.lo.as_ptr());
    let c2_hi = vld1q_u8(tables.c2.hi.as_ptr());
    let c3_lo = vld1q_u8(tables.c3.lo.as_ptr());
    let c3_hi = vld1q_u8(tables.c3.hi.as_ptr());
    let zero = vdupq_n_u8(0);
    let nibble_mask = vdupq_n_u8(0x0F);
    let lane_bits = vld1q_u8(NEON_MOVEMASK_BITS.as_ptr());
    let setup_us = setup_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();

    let scan_start: usize = (*offsets.get_unchecked(0)).as_();
    let mut string_idx: usize = 0;
    let mut string_end: usize = (*offsets.get_unchecked(1)).as_();
    let ptr = all_bytes.as_ptr();
    let mut i: usize = 0;
    let mut main_blocks = 0usize;
    let mut nonzero_masks = 0usize;
    let mut candidates = 0usize;
    let mut offset_advances = 0usize;
    let mut already_marked = 0usize;
    let mut verifies = 0usize;
    let mut matches = 0usize;
    let mut bit_writes = 0usize;
    let mut candidate_us = 0f64;
    let mut verify_us = 0f64;
    while i < main_len {
        main_blocks += usize::from(trace);
        let v1 = vld1q_u8(ptr.add(i));
        let v2 = vld1q_u8(ptr.add(i + 1));
        let v3 = vld1q_u8(ptr.add(i + 2));

        let c1_bits = neon_nibble_lookup(c1_lo, c1_hi, v1, nibble_mask);
        let c2_bits = neon_nibble_lookup(c2_lo, c2_hi, v2, nibble_mask);
        let c3_bits = neon_nibble_lookup(c3_lo, c3_hi, v3, nibble_mask);
        let triple = vandq_u8(vandq_u8(c1_bits, c2_bits), c3_bits);

        let mut mask = u32::from(neon_nonzero_mask(triple, zero, lane_bits));
        if mask != 0 {
            nonzero_masks += usize::from(trace);
            let candidate_t = trace.then(std::time::Instant::now);
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let cand = i + bit;
                candidates += usize::from(trace);
                if cand < scan_start {
                    continue;
                }
                while cand >= string_end {
                    offset_advances += usize::from(trace);
                    string_idx += 1;
                    if string_idx >= n {
                        if let Some(pass_t) = pass_t {
                            let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
                            let tail_us = 0f64;
                            eprintln!(
                                "[fsst::stream_pass] kind=triple impl=neon bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us={:.3} vector_us={:.3} total_us={:.3}",
                                len,
                                n,
                                main_blocks,
                                nonzero_masks,
                                candidates,
                                offset_advances,
                                already_marked,
                                verifies,
                                matches,
                                bit_writes,
                                setup_us,
                                candidate_us,
                                verify_us,
                                tail_us,
                                pass_us - setup_us - candidate_us - tail_us,
                                pass_us,
                            );
                        }
                        return;
                    }
                    string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
                }
                let already = bits.value(string_idx);
                if (!negated && already) || (negated && !already) {
                    already_marked += usize::from(trace);
                    continue;
                }
                verifies += usize::from(trace);
                let verify_t = trace.then(std::time::Instant::now);
                let accepted = verify_at(cand, string_end);
                verify_us += verify_t
                    .map(|t| t.elapsed().as_secs_f64() * 1e6)
                    .unwrap_or_default();
                if accepted {
                    matches += usize::from(trace);
                    bit_writes += usize::from(trace);
                    if negated {
                        bits.unset_unchecked(string_idx);
                    } else {
                        bits.set_unchecked(string_idx);
                    }
                }
            }
            candidate_us += candidate_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
        }
        i += 16;
    }
    let tail_t = trace.then(std::time::Instant::now);
    let mut tail_positions = 0usize;
    let mut tail_candidates = 0usize;
    if len > 2 {
        for j in i..len - 2 {
            tail_positions += usize::from(trace);
            let b1 = *all_bytes.get_unchecked(j);
            let b2 = *all_bytes.get_unchecked(j + 1);
            let b3 = *all_bytes.get_unchecked(j + 2);
            let c1_bits_b =
                tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
            let c2_bits_b =
                tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
            let c3_bits_b =
                tables.c3.lo[usize::from(b3 & 0x0F)] & tables.c3.hi[usize::from(b3 >> 4)];
            if (c1_bits_b & c2_bits_b & c3_bits_b) == 0 {
                continue;
            }
            tail_candidates += usize::from(trace);
            let cand = j;
            if cand < scan_start {
                continue;
            }
            while cand >= string_end {
                offset_advances += usize::from(trace);
                string_idx += 1;
                if string_idx >= n {
                    break;
                }
                string_end = (*offsets.get_unchecked(string_idx + 1)).as_();
            }
            if string_idx >= n {
                break;
            }
            let already = bits.value(string_idx);
            if (!negated && already) || (negated && !already) {
                already_marked += usize::from(trace);
                continue;
            }
            verifies += usize::from(trace);
            let verify_t = trace.then(std::time::Instant::now);
            let accepted = verify_at(cand, string_end);
            verify_us += verify_t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            if accepted {
                matches += usize::from(trace);
                bit_writes += usize::from(trace);
                if negated {
                    bits.unset_unchecked(string_idx);
                } else {
                    bits.set_unchecked(string_idx);
                }
            }
        }
    }
    let tail_us = tail_t
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or_default();
    if let Some(pass_t) = pass_t {
        let pass_us = pass_t.elapsed().as_secs_f64() * 1e6;
        eprintln!(
            "[fsst::stream_pass] kind=triple impl=neon bytes={} rows={} main_blocks={} nonzero_masks={} candidates={} tail_positions={} tail_candidates={} offset_advances={} already_marked={} verifies={} matches={} bit_writes={} setup_us={:.3} candidate_us={:.3} verify_us={:.3} tail_us={:.3} vector_us={:.3} total_us={:.3}",
            len,
            n,
            main_blocks,
            nonzero_masks,
            candidates,
            tail_positions,
            tail_candidates,
            offset_advances,
            already_marked,
            verifies,
            matches,
            bit_writes,
            setup_us,
            candidate_us,
            verify_us,
            tail_us,
            pass_us - setup_us - candidate_us - tail_us,
            pass_us,
        );
    }
}

fn teddy_triple_pass_scalar<T, V>(
    tables: &TripleTables,
    // TODO: scalar SSA fusion. See `teddy_pair_pass_scalar`.
    _ssa_tables: Option<&NibbleTables>,
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify_at: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(usize, usize) -> bool,
{
    let len = all_bytes.len();
    if len < 3 {
        return;
    }
    let scan_start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
    let mut string_idx: usize = 0;
    let mut string_end: usize = unsafe { *offsets.get_unchecked(1) }.as_();
    for j in 0..len - 2 {
        let b1 = all_bytes[j];
        let b2 = all_bytes[j + 1];
        let b3 = all_bytes[j + 2];
        let c1_bits = tables.c1.lo[usize::from(b1 & 0x0F)] & tables.c1.hi[usize::from(b1 >> 4)];
        let c2_bits = tables.c2.lo[usize::from(b2 & 0x0F)] & tables.c2.hi[usize::from(b2 >> 4)];
        let c3_bits = tables.c3.lo[usize::from(b3 & 0x0F)] & tables.c3.hi[usize::from(b3 >> 4)];
        if (c1_bits & c2_bits & c3_bits) == 0 {
            continue;
        }
        let cand = j;
        if cand < scan_start {
            continue;
        }
        while cand >= string_end {
            string_idx += 1;
            if string_idx >= n {
                return;
            }
            string_end = unsafe { *offsets.get_unchecked(string_idx + 1) }.as_();
        }
        let already = bits.value(string_idx);
        if (!negated && already) || (negated && !already) {
            continue;
        }
        if verify_at(cand, string_end) {
            if negated {
                unsafe { bits.unset_unchecked(string_idx) };
            } else {
                unsafe { bits.set_unchecked(string_idx) };
            }
        }
    }
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

    /// Reference: bit `i` set iff there exists a bucket `b` such that
    /// `all_bytes[i] == c1_b` AND `all_bytes[i+1] ∈ c2_set_b`. This is the
    /// *exact* bucketed predicate without nibble-cross FPs — the SIMD
    /// path's output is a superset of this.
    fn naive_bucketed_pair_bitset(all_bytes: &[u8], buckets: &[(u8, Vec<u8>)]) -> Vec<u64> {
        let mut out = vec![0u64; all_bytes.len().div_ceil(64)];
        if all_bytes.len() < 2 {
            return out;
        }
        for i in 0..all_bytes.len() - 1 {
            let b1 = all_bytes[i];
            let b2 = all_bytes[i + 1];
            let hit = buckets
                .iter()
                .any(|(c1, c2_set)| b1 == *c1 && c2_set.contains(&b2));
            if hit {
                out[i >> 6] |= 1u64 << (i & 63);
            }
        }
        out
    }

    fn bit_is_set(bitset: &[u64], i: usize) -> bool {
        bitset[i >> 6] & (1u64 << (i & 63)) != 0
    }

    /// Single-bucket: identical to pure Cartesian for one (c1, c2_set) pair.
    /// Still has at most a small nibble-cross FP within the c2_set.
    #[test]
    fn bucketed_pair_single_bucket() {
        // c1 = 'g' = 0x67, c2_set = {'o' = 0x6F} — both nibble-unique, so
        // SIMD output matches naive exactly.
        let buckets: Vec<(u8, Vec<u8>)> = vec![(b'g', vec![b'o'])];
        let bytes = b"xgo gOg google bytes".to_vec();
        let got = build_bucketed_pair_bitset(&bytes, &buckets);
        let exp = naive_bucketed_pair_bitset(&bytes, &buckets);
        assert_eq!(got, exp);
    }

    /// Two distinct c1's, each with disjoint c2 sets. Bucketed must reject
    /// the cross-pair `(c1_a, c2_b)` that pure Cartesian would admit.
    #[test]
    fn bucketed_pair_rejects_cross_bucket() {
        let buckets: Vec<(u8, Vec<u8>)> = vec![(b'a', vec![b'b']), (b'c', vec![b'd'])];
        let bytes = b"abxxcd ad cb ab cd zzab".to_vec();
        let got = build_bucketed_pair_bitset(&bytes, &buckets);
        // Naive (exact) shouldn't set `ad` or `cb` positions; SIMD path
        // must also not set them — c1 lane bit for 'a' is bit 0; c2 lane
        // bit for 'd' is bit 1 (since 'd' is in bucket 1's c2 set). 0 & 1
        // = 0, so the cross-pair is correctly rejected.
        let exp = naive_bucketed_pair_bitset(&bytes, &buckets);
        assert_eq!(got, exp, "cross-bucket pairs leaked into SIMD bitset");
        // Spot-check: 'ad' at index 8, 'cb' at index 11 must be unset.
        let s = bytes.windows(2).position(|w| w == b"ad").unwrap();
        assert!(!bit_is_set(&got, s), "ad at {s} unexpectedly set");
        let s = bytes.windows(2).position(|w| w == b"cb").unwrap();
        assert!(!bit_is_set(&got, s), "cb at {s} unexpectedly set");
    }

    /// 8 buckets at the single-pass capacity. Tests that the per-bucket bit
    /// packing doesn't collide.
    #[test]
    #[expect(clippy::cast_possible_truncation)]
    fn bucketed_pair_eight_buckets() {
        let buckets: Vec<(u8, Vec<u8>)> = (0..8u8).map(|i| (0x10 + i, vec![0x20 + i])).collect();
        // Each valid pair at a distinct position; one cross-pair injected.
        let mut bytes: Vec<u8> = Vec::new();
        for i in 0..8u8 {
            bytes.push(0x10 + i);
            bytes.push(0x20 + i);
            bytes.push(0xFE); // separator, unlikely to alias
        }
        // Inject a cross-pair: (0x10, 0x21) — c1 from bucket 0, c2 from
        // bucket 1. Must NOT be set in the output.
        bytes.extend_from_slice(&[0x10, 0x21]);
        let got = build_bucketed_pair_bitset(&bytes, &buckets);
        let exp = naive_bucketed_pair_bitset(&bytes, &buckets);
        assert_eq!(got, exp, "8-bucket single-pass mismatch");
    }

    /// More than 8 buckets — multi-pass OR-merge. Verifies that the union of
    /// pair hits matches the naive exact predicate.
    #[test]
    fn bucketed_pair_multi_pass() {
        // 10 buckets: distinct c1, single-element c2 (nibble-unique
        // pairs ⇒ no within-bucket FP).
        let buckets: Vec<(u8, Vec<u8>)> = (0..10u8).map(|i| (0x40 + i, vec![0x50 + i])).collect();
        // Place each pair, plus cross-pairs and noise.
        let mut bytes: Vec<u8> = Vec::with_capacity(512);
        for i in 0..10u8 {
            bytes.extend_from_slice(&[0x40 + i, 0x50 + i, 0xFF]);
        }
        // Cross-pair: (0x40, 0x51) — must NOT match.
        bytes.extend_from_slice(&[0x40, 0x51]);
        // Pad to test multi-word output.
        bytes.extend((0..200u32).map(|j| (j.wrapping_mul(7) & 0xFF) as u8));
        let got = build_bucketed_pair_bitset(&bytes, &buckets);
        let exp = naive_bucketed_pair_bitset(&bytes, &buckets);
        assert_eq!(got, exp, "multi-pass bucketed bitset mismatch");
    }

    /// Tail handling: input length not a multiple of 32 + scalar tail.
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(31)]
    #[case(32)]
    #[case(33)]
    #[case(63)]
    #[case(64)]
    #[case(65)]
    #[case(127)]
    #[case(128)]
    #[case(257)]
    fn bucketed_pair_lengths(#[case] len: usize) {
        let buckets: Vec<(u8, Vec<u8>)> = vec![(b'g', vec![b'o'])];
        let bytes: Vec<u8> = (0..len)
            .map(|i| {
                // Sprinkle some 'g' and 'o' bytes.
                match i % 7 {
                    0 => b'g',
                    1 => b'o',
                    2 => b'x',
                    _ => ((i & 0xFF) as u8).wrapping_mul(31),
                }
            })
            .collect();
        let got = build_bucketed_pair_bitset(&bytes, &buckets);
        let exp = naive_bucketed_pair_bitset(&bytes, &buckets);
        assert_eq!(got, exp, "mismatch at len={len}");
    }

    /// SIMD output is a superset of the exact predicate for c2 sets that
    /// admit nibble-cross within-bucket FPs.
    #[test]
    fn bucketed_pair_superset_for_diverse_c2() {
        // c2_set has nibble-diverse entries: 0x12 and 0x34.
        // The nibble tables admit 0x14 and 0x32 as "any-bucket" matches.
        let buckets: Vec<(u8, Vec<u8>)> = vec![(0xAA, vec![0x12, 0x34])];
        let bytes: Vec<u8> = vec![0xAA, 0x14, 0xAA, 0x32, 0xAA, 0x12, 0xAA, 0x34];
        let got = build_bucketed_pair_bitset(&bytes, &buckets);
        let exp = naive_bucketed_pair_bitset(&bytes, &buckets);
        // Every bit in exp must be in got (no false negatives).
        for i in 0..bytes.len() {
            if bit_is_set(&exp, i) {
                assert!(bit_is_set(&got, i), "FN at {i}");
            }
        }
        // Spot-check: positions 4 and 6 (the true pairs) must be set.
        assert!(bit_is_set(&got, 4));
        assert!(bit_is_set(&got, 6));
        // Positions 0 and 2 are the nibble-cross FPs admitted by SIMD —
        // we don't assert exact equality here.
    }

    fn naive_bucketed_triple_bitset(
        all_bytes: &[u8],
        buckets: &[(u8, Vec<u8>, Vec<u8>)],
    ) -> Vec<u64> {
        let mut out = vec![0u64; all_bytes.len().div_ceil(64)];
        if all_bytes.len() < 3 {
            return out;
        }
        for i in 0..all_bytes.len() - 2 {
            let b1 = all_bytes[i];
            let b2 = all_bytes[i + 1];
            let b3 = all_bytes[i + 2];
            let hit = buckets.iter().any(|(c1, c2_set, c3_set)| {
                b1 == *c1 && c2_set.contains(&b2) && c3_set.contains(&b3)
            });
            if hit {
                out[i >> 6] |= 1u64 << (i & 63);
            }
        }
        out
    }

    /// Exact case with nibble-unique c2/c3 sets: SIMD matches naive predicate.
    #[test]
    fn triple_single_bucket_exact() {
        let buckets: Vec<(u8, Vec<u8>, Vec<u8>)> = vec![(b'g', vec![b'o'], vec![b'o'])];
        let bytes = b"xgoo googoo google bytes".to_vec();
        let got = build_bucketed_triple_bitset(&bytes, &buckets);
        let exp = naive_bucketed_triple_bitset(&bytes, &buckets);
        assert_eq!(got, exp);
    }

    /// Cross-bucket triples must NOT match.
    #[test]
    fn triple_rejects_cross_bucket() {
        // Two buckets: (a, b, c) and (x, y, z). The triple (a, b, z)
        // must be rejected — it crosses buckets.
        let buckets: Vec<(u8, Vec<u8>, Vec<u8>)> = vec![
            (b'a', vec![b'b'], vec![b'c']),
            (b'x', vec![b'y'], vec![b'z']),
        ];
        let bytes = b"abc xyz abz xyc abc".to_vec();
        let got = build_bucketed_triple_bitset(&bytes, &buckets);
        let exp = naive_bucketed_triple_bitset(&bytes, &buckets);
        assert_eq!(got, exp);
    }

    /// > 8 buckets → multi-pass OR-merge.
    #[test]
    fn triple_multi_pass() {
        let buckets: Vec<(u8, Vec<u8>, Vec<u8>)> = (0..10u8)
            .map(|i| (0x40 + i, vec![0x50 + i], vec![0x60 + i]))
            .collect();
        let mut bytes: Vec<u8> = Vec::new();
        for i in 0..10u8 {
            bytes.extend_from_slice(&[0x40 + i, 0x50 + i, 0x60 + i, 0xFF]);
        }
        // Cross-bucket triple: (0x40, 0x51, 0x62) — must NOT match.
        bytes.extend_from_slice(&[0x40, 0x51, 0x62]);
        bytes.extend((0..200u32).map(|j| (j.wrapping_mul(7) & 0xFF) as u8));
        let got = build_bucketed_triple_bitset(&bytes, &buckets);
        let exp = naive_bucketed_triple_bitset(&bytes, &buckets);
        assert_eq!(got, exp);
    }

    /// Length boundary: AVX2 tail + scalar fallback paths agree.
    #[rstest]
    #[case(2)]
    #[case(3)]
    #[case(32)]
    #[case(33)]
    #[case(34)]
    #[case(64)]
    #[case(65)]
    #[case(66)]
    #[case(257)]
    fn triple_lengths(#[case] len: usize) {
        let buckets: Vec<(u8, Vec<u8>, Vec<u8>)> = vec![(b'g', vec![b'o'], vec![b'o'])];
        let bytes: Vec<u8> = (0..len)
            .map(|i| match i % 5 {
                0 => b'g',
                1 => b'o',
                2 => b'o',
                3 => b'x',
                _ => ((i & 0xFF) as u8).wrapping_mul(31),
            })
            .collect();
        let got = build_bucketed_triple_bitset(&bytes, &buckets);
        let exp = naive_bucketed_triple_bitset(&bytes, &buckets);
        assert_eq!(got, exp, "mismatch at len={len}");
    }

    fn bit_values(bits: &BitBuffer, len: usize) -> Vec<bool> {
        (0..len).map(|i| bits.value(i)).collect()
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn teddy_pair_neon_matches_scalar() {
        let buckets: Vec<(u8, Vec<u8>)> = vec![(b'a', vec![b'b', b'x']), (b'c', vec![b'd'])];
        let tables = BucketTables::build(&buckets);
        let strings = [
            b"zzabyy".as_slice(),
            b"zzadyy",
            b"xxcd",
            b"nomatch",
            b"abxxab",
        ];
        let n = strings.len();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut all_bytes = Vec::new();
        offsets.push(0u32);
        for s in strings {
            all_bytes.extend_from_slice(s);
            offsets.push(u32::try_from(all_bytes.len()).unwrap());
        }

        let mut scalar_bits = BitBufferMut::new_unset(n);
        let mut scalar_verify = |cand: usize, end: usize| {
            cand + 1 < end && matches!(&all_bytes[cand..cand + 2], [b'a', b'b'] | [b'c', b'd'])
        };
        teddy_pair_pass_scalar(
            &tables,
            None,
            n,
            &offsets,
            &all_bytes,
            false,
            &mut scalar_bits,
            &mut scalar_verify,
        );

        let mut neon_bits = BitBufferMut::new_unset(n);
        let mut neon_verify = |cand: usize, end: usize| {
            cand + 1 < end && matches!(&all_bytes[cand..cand + 2], [b'a', b'b'] | [b'c', b'd'])
        };
        unsafe {
            teddy_pair_pass_neon(
                &tables,
                n,
                &offsets,
                &all_bytes,
                false,
                &mut neon_bits,
                &mut neon_verify,
            );
        }

        assert_eq!(
            bit_values(&scalar_bits.freeze(), n),
            bit_values(&neon_bits.freeze(), n)
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn teddy_triple_neon_matches_scalar() {
        let buckets: Vec<(u8, Vec<u8>, Vec<u8>)> = vec![
            (b'a', vec![b'b'], vec![b'c', b'x']),
            (b'x', vec![b'y'], vec![b'z']),
        ];
        let tables = TripleTables::build(&buckets);
        let strings = [b"abc".as_slice(), b"abz", b"xyz", b"nomatch", b"abcxxxyz"];
        let n = strings.len();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut all_bytes = Vec::new();
        offsets.push(0u32);
        for s in strings {
            all_bytes.extend_from_slice(s);
            offsets.push(u32::try_from(all_bytes.len()).unwrap());
        }

        let mut scalar_bits = BitBufferMut::new_unset(n);
        let mut scalar_verify = |cand: usize, end: usize| {
            cand + 2 < end
                && matches!(
                    &all_bytes[cand..cand + 3],
                    [b'a', b'b', b'c'] | [b'x', b'y', b'z']
                )
        };
        teddy_triple_pass_scalar(
            &tables,
            None,
            n,
            &offsets,
            &all_bytes,
            false,
            &mut scalar_bits,
            &mut scalar_verify,
        );

        let mut neon_bits = BitBufferMut::new_unset(n);
        let mut neon_verify = |cand: usize, end: usize| {
            cand + 2 < end
                && matches!(
                    &all_bytes[cand..cand + 3],
                    [b'a', b'b', b'c'] | [b'x', b'y', b'z']
                )
        };
        unsafe {
            teddy_triple_pass_neon(
                &tables,
                n,
                &offsets,
                &all_bytes,
                false,
                &mut neon_bits,
                &mut neon_verify,
            );
        }

        assert_eq!(
            bit_values(&scalar_bits.freeze(), n),
            bit_values(&neon_bits.freeze(), n)
        );
    }
}
