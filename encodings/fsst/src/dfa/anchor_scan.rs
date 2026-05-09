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

/// Build a packed bitset of length `all_bytes.len()` whose bit `i` is set
/// iff `all_bytes[i] ∈ c1_codes` AND `all_bytes[i+1] ∈ c2_codes`. The last
/// bit (`i == all_bytes.len() - 1`) is forced to 0 — there is no
/// successor byte for the c2 lookup.
///
/// Used by the folded-contains scan path as a much sparser alternative
/// to [`build_progressing_bitset_unbounded`] when both code sets fit in
/// [`MAX_SET_BYTES`]. On real ClickBench URL data with single-byte
/// `{c1=g, c2=o}` sets the pair bitset is ~100–1000× sparser than the
/// 1-byte progressing bitset, dramatically reducing per-string DFA
/// state-0 visits at a cost of one fused `vpshufb` pair per 32 input
/// bytes (vs. one for the 1-byte path). Returns `None` when either set
/// exceeds [`MAX_SET_BYTES`].
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

/// Fused fill of two bitsets in a single walk over `all_bytes`. The two
/// PSHUFB lookups per 32-byte block share the same input load and zero
/// vector — roughly ~1.4× one-table cost on typical x86_64 parts vs
/// 2.0× for two independent walks. Halves the bandwidth cost of the
/// 2-byte anchor scheme, which is the difference between net win and
/// net regression on data shapes where the pair selectivity gain is
/// modest.
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

#[cfg(target_arch = "x86_64")]
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
        let tables = NibbleTables::build(progressing_codes).expect("size already checked");
        fill_bitset(all_bytes, &tables, &mut out);
        return out;
    }
    // Multi-pass: chunk the codes, build a per-chunk bitset, OR-merge
    // into `out`. Reuse a scratch buffer across chunks to amortize
    // allocation.
    let mut scratch = vec![0u64; n_words];
    for chunk in progressing_codes.chunks(MAX_SET_BYTES) {
        let tables = NibbleTables::build(chunk).expect("chunk size <= MAX_SET_BYTES");
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

/// Probe the bitset for any set bit in the byte range `[start, end)`. Used by
/// the streaming-merge phase to decide whether to dispatch a per-string DFA
/// run or write `false` (or `negated`) directly.
#[inline]
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

/// Compute pair-eligible (c1, c2) code sets for the 2-byte anchor scan.
///
/// `transitions` is the full `(2N + 1) × 256` folded transition matrix.
/// `c1_codes` is the full state-0 progressing set.
///
/// The first returned vec is the pair-eligible subset of `c1_codes` —
/// codes whose one-step state is non-zero AND non-accept. (Single-step
/// accepts are excluded; they're captured by the existing accept-alone
/// path.) The second vec is the union of **strictly-advancing or escape**
/// c2 codes for those c1's: bytes `c2` such that `T[s1][c2] > s1` —
/// numerically advancing toward accept — or `c2 == ESCAPE_CODE` for
/// safety on escape sequences.
///
/// ## Why advancing-only?
///
/// Every match has some position `p` where the DFA strictly advances on
/// the very next byte (otherwise it never escapes `s1` and never reaches
/// accept). The first such `(p, p+1)` pair has `c1` in `c1_codes` and
/// `c2` strictly advancing — so its bit is set in the pair bitset. The
/// matcher then runs the DFA from `p` (state 0) and reaches accept,
/// possibly via KMP fallback on the prefix bytes before `p`.
///
/// Returns `None` when `accept_state < 2`, when no c1 is pair-eligible
/// (every progressing code is single-step accept), or when no c2 codes
/// strictly advance from any pair-eligible c1.
pub(super) fn collect_pair_codes(
    transitions: &[u8],
    c1_codes: &[u8],
    accept_state: u8,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if accept_state < 2 {
        return None;
    }
    debug_assert!(transitions.len() >= 256);
    let mut c2_seen = [false; 256];
    let mut c1_out: Vec<u8> = Vec::new();
    let mut c2_out: Vec<u8> = Vec::new();
    for &c1 in c1_codes {
        let s1 = transitions[usize::from(c1)];
        if s1 == 0 || s1 == accept_state {
            continue;
        }
        c1_out.push(c1);
        let row = usize::from(s1) * 256;
        // Advancing predicate: for **normal** states (`s1 ≤ accept_state`),
        // strictly higher state ids encode "moved further along the
        // match" (folded-DFA layout numbers normal states 0..=N
        // monotonically). For **escape** states (`s1 > accept_state`,
        // which is the post-`ESCAPE_CODE` "expecting a literal byte"
        // wrapper), the transition lands in a normal state (numerically
        // lower than `s1`), so the strict-greater predicate would
        // incorrectly drop *every* progressing literal. Use
        // "non-zero next" for escape c1's.
        let s1_is_escape = s1 > accept_state;
        for c2 in 0..=255usize {
            let next = transitions[row + c2];
            let advances = if s1_is_escape { next != 0 } else { next > s1 };
            let escape = c2 == usize::from(fsst::ESCAPE_CODE);
            if (advances || escape) && !c2_seen[c2] {
                c2_seen[c2] = true;
                c2_out.push(c2 as u8);
            }
        }
    }
    if c1_out.is_empty() || c2_out.is_empty() {
        None
    } else {
        Some((c1_out, c2_out))
    }
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

// ---------------------------------------------------------------------------
// Bucketed Cartesian Teddy
// ---------------------------------------------------------------------------
//
// Generalization of [`build_pair_bitset`]. Each Teddy bucket holds a single
// `(c1, c2_set)` pair where the c1 axis is exact (one byte per bucket) and
// the c2 axis is an over-approximating nibble-table membership test. Up to
// [`MAX_TEDDY_BUCKETS`] buckets fit in one PSHUFB pass.
//
// Selectivity vs. the existing single-bucket Cartesian path:
//
//   Cartesian:    P(hit) = (Σ_b |c1_b|) · (Σ_b |c2_b|) / 65536
//   Bucketed:     P(hit) = Σ_b (1 · |c2_b|) / 65536              (shared-c1)
//
// The cross terms `|c1_a| · |c2_b|` for `a ≠ b` are exactly the
// false-positive pairs the bucketing eliminates. On real FSST-trained URL
// data the bucketed path is typically 3–10× sparser than the Cartesian
// path at the same SIMD cost (4 PSHUFBs + ANDs per 32 input bytes).

/// Maximum number of Teddy buckets that fit in one PSHUFB pass. One bit per
/// bucket in the 8-bit nibble-table entries.
pub(super) const MAX_TEDDY_BUCKETS: usize = 8;

/// Precomputed nibble tables for the bucketed Cartesian Teddy scan.
///
/// For bucket `b` (indexed 0..n_buckets), bit `b` of `lo_c1[i]` is set iff
/// bucket `b`'s c1 has low nibble `i`; similarly for `hi_c1`, `lo_c2`,
/// `hi_c2`. The membership test for bucket `b` at input position `i` is:
///
/// ```text
/// c1_match = lo_c1[byte[i]   & 0xF] & hi_c1[byte[i]   >> 4]
/// c2_match = lo_c2[byte[i+1] & 0xF] & hi_c2[byte[i+1] >> 4]
/// bucket_b_hit = ((c1_match & c2_match) >> b) & 1
/// any_hit       = (c1_match & c2_match) != 0
/// ```
struct TeddyTables {
    lo_c1: [u8; 16],
    hi_c1: [u8; 16],
    lo_c2: [u8; 16],
    hi_c2: [u8; 16],
}

impl TeddyTables {
    /// Build nibble tables for `buckets`. Returns `None` if there are more
    /// than [`MAX_TEDDY_BUCKETS`] buckets.
    fn build(buckets: &[(u8, Vec<u8>)]) -> Option<Self> {
        if buckets.is_empty() || buckets.len() > MAX_TEDDY_BUCKETS {
            return None;
        }
        let mut lo_c1 = [0u8; 16];
        let mut hi_c1 = [0u8; 16];
        let mut lo_c2 = [0u8; 16];
        let mut hi_c2 = [0u8; 16];
        for (b, (c1, c2_set)) in buckets.iter().enumerate() {
            let bit = 1u8 << b;
            lo_c1[usize::from(*c1 & 0x0F)] |= bit;
            hi_c1[usize::from(*c1 >> 4)] |= bit;
            for &c in c2_set {
                lo_c2[usize::from(c & 0x0F)] |= bit;
                hi_c2[usize::from(c >> 4)] |= bit;
            }
        }
        Some(Self {
            lo_c1,
            hi_c1,
            lo_c2,
            hi_c2,
        })
    }
}

/// Build a bucketed Cartesian Teddy pair bitset.
///
/// `buckets[b] = (c1_b, c2_set_b)`: bit `i` of the output is set iff there
/// exists a bucket `b` such that
///
/// - `all_bytes[i] == c1_b` (exact byte match — zero false positives), and
/// - `all_bytes[i+1] ∈ c2_set_b` modulo nibble-table over-approximation
///   (admits at most a few phantom bytes when `|c2_set_b| > 1`).
///
/// The c1 axis is exact because each bucket gets exactly one c1 byte. Phantom
/// matches on the c2 axis are cheap to filter at the verifier and are bounded
/// by the per-bucket nibble cross-product of the c2 set.
///
/// The last bit (`i == all_bytes.len() - 1`) is forced to 0 — there is no
/// successor byte for the c2 lookup.
///
/// Returns `None` when there are more than [`MAX_TEDDY_BUCKETS`] buckets;
/// callers should fall back to multi-pass OR-merge or to the plain
/// [`build_pair_bitset`].
pub(super) fn build_pair_bitset_teddy(
    all_bytes: &[u8],
    buckets: &[(u8, Vec<u8>)],
) -> Option<Vec<u64>> {
    let tables = TeddyTables::build(buckets)?;
    let len = all_bytes.len();
    let n_words = len.div_ceil(64);
    if len == 0 {
        return Some(Vec::new());
    }

    // Materialize per-byte 8-bit bucket masks for c1 and c2. We use two
    // intermediate `Vec<u8>` buffers and a separate combine pass; this is
    // the simplest correct shape and matches how the existing Cartesian
    // path stages two independent bitsets before AND-shifting.
    let mut c1_byte_masks = vec![0u8; len];
    let mut c2_byte_masks = vec![0u8; len];
    fill_byte_masks(all_bytes, &tables, &mut c1_byte_masks, &mut c2_byte_masks);

    // Combine: bit `i` = (c1_byte_masks[i] & c2_byte_masks[i+1]) != 0.
    let mut out = vec![0u64; n_words];
    let mut word: u64 = 0;
    let mut word_idx: usize = 0;
    let mut bit_in_word: u64 = 0;
    let last = len - 1;
    for i in 0..last {
        // SAFETY: `i < last < len` and `i + 1 <= last < len`.
        let m = unsafe { *c1_byte_masks.get_unchecked(i) & *c2_byte_masks.get_unchecked(i + 1) };
        word |= u64::from(m != 0) << bit_in_word;
        bit_in_word += 1;
        if bit_in_word == 64 {
            // SAFETY: word_idx < n_words by construction.
            unsafe { *out.get_unchecked_mut(word_idx) = word };
            word_idx += 1;
            word = 0;
            bit_in_word = 0;
        }
    }
    // Last bit (i = len - 1): no successor byte for c2, leave 0.
    if bit_in_word > 0 {
        // SAFETY: word_idx < n_words.
        unsafe { *out.get_unchecked_mut(word_idx) = word };
    }
    Some(out)
}

/// Fill `c1_out[i]` and `c2_out[i]` with the per-bucket-bit nibble-table
/// match masks for `all_bytes[i]`. SIMD on AVX2, scalar otherwise.
fn fill_byte_masks(all_bytes: &[u8], tables: &TeddyTables, c1_out: &mut [u8], c2_out: &mut [u8]) {
    debug_assert_eq!(c1_out.len(), all_bytes.len());
    debug_assert_eq!(c2_out.len(), all_bytes.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature was just detected at runtime.
            unsafe { fill_byte_masks_avx2(all_bytes, tables, c1_out, c2_out) };
            return;
        }
    }
    fill_byte_masks_scalar(all_bytes, tables, c1_out, c2_out);
}

fn fill_byte_masks_scalar(
    all_bytes: &[u8],
    tables: &TeddyTables,
    c1_out: &mut [u8],
    c2_out: &mut [u8],
) {
    for (i, &b) in all_bytes.iter().enumerate() {
        let lo = usize::from(b & 0x0F);
        let hi = usize::from(b >> 4);
        c1_out[i] = tables.lo_c1[lo] & tables.hi_c1[hi];
        c2_out[i] = tables.lo_c2[lo] & tables.hi_c2[hi];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn fill_byte_masks_avx2(
    all_bytes: &[u8],
    tables: &TeddyTables,
    c1_out: &mut [u8],
    c2_out: &mut [u8],
) {
    use core::arch::x86_64::_mm256_set1_epi8;
    use core::arch::x86_64::_mm256_storeu_si256;

    let len = all_bytes.len();
    let main_len = len & !31;

    let lo_c1 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.lo_c1.as_ptr() as *const __m128i));
    let hi_c1 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.hi_c1.as_ptr() as *const __m128i));
    let lo_c2 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.lo_c2.as_ptr() as *const __m128i));
    let hi_c2 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.hi_c2.as_ptr() as *const __m128i));
    let nibble_mask = _mm256_set1_epi8(0x0F);

    let in_ptr = all_bytes.as_ptr();
    let c1_ptr = c1_out.as_mut_ptr();
    let c2_ptr = c2_out.as_mut_ptr();

    let mut i: usize = 0;
    while i < main_len {
        // SAFETY: `i + 31 < main_len <= len`.
        let v = _mm256_loadu_si256(in_ptr.add(i) as *const __m256i);
        let v_lo = _mm256_and_si256(v, nibble_mask);
        let v_hi = _mm256_and_si256(_mm256_srli_epi64(v, 4), nibble_mask);

        // Per-bucket-bit c1 / c2 membership masks (32 × u8).
        let c1_lo_b = _mm256_shuffle_epi8(lo_c1, v_lo);
        let c1_hi_b = _mm256_shuffle_epi8(hi_c1, v_hi);
        let c1_match = _mm256_and_si256(c1_lo_b, c1_hi_b);

        let c2_lo_b = _mm256_shuffle_epi8(lo_c2, v_lo);
        let c2_hi_b = _mm256_shuffle_epi8(hi_c2, v_hi);
        let c2_match = _mm256_and_si256(c2_lo_b, c2_hi_b);

        // SAFETY: `i + 31 < main_len <= len = c1_out.len() = c2_out.len()`.
        _mm256_storeu_si256(c1_ptr.add(i) as *mut __m256i, c1_match);
        _mm256_storeu_si256(c2_ptr.add(i) as *mut __m256i, c2_match);

        i += 32;
    }

    // Tail: scalar.
    while i < len {
        let b = *all_bytes.get_unchecked(i);
        let lo = usize::from(b & 0x0F);
        let hi = usize::from(b >> 4);
        *c1_out.get_unchecked_mut(i) = tables.lo_c1[lo] & tables.hi_c1[hi];
        *c2_out.get_unchecked_mut(i) = tables.lo_c2[lo] & tables.hi_c2[hi];
        i += 1;
    }
}

/// Collect per-c1 buckets `(c1, c2_set)` for the bucketed Teddy scan.
///
/// `transitions` is the full folded transition matrix; `c1_codes` is the
/// state-0 progressing-code set. For each c1 whose one-step state is
/// non-zero and non-accept, builds the set of strictly-advancing-or-escape
/// c2 codes (mirroring [`collect_pair_codes`]'s eligibility predicate).
///
/// Returns `None` when more than [`MAX_TEDDY_BUCKETS`] distinct c1's are
/// pair-eligible — multi-pass spillover is left as future work; callers
/// should fall back to the plain Cartesian path or the 1-byte path.
///
/// Returns `None` when no c1 is pair-eligible (every progressing code is
/// stuck or single-step accept) or when no c1 has any advancing c2.
pub(super) fn collect_pair_buckets_shared_c1(
    transitions: &[u8],
    c1_codes: &[u8],
    accept_state: u8,
) -> Option<Vec<(u8, Vec<u8>)>> {
    if accept_state < 2 {
        return None;
    }
    debug_assert!(transitions.len() >= 256);

    let mut buckets: Vec<(u8, Vec<u8>)> = Vec::new();
    for &c1 in c1_codes {
        let s1 = transitions[usize::from(c1)];
        if s1 == 0 || s1 == accept_state {
            continue;
        }
        let s1_is_escape = s1 > accept_state;
        let row = usize::from(s1) * 256;
        let mut c2_set: Vec<u8> = Vec::new();
        for c2 in 0..=255u8 {
            let next = transitions[row + usize::from(c2)];
            let advances = if s1_is_escape { next != 0 } else { next > s1 };
            let escape = c2 == fsst::ESCAPE_CODE;
            if advances || escape {
                c2_set.push(c2);
            }
        }
        if c2_set.is_empty() {
            continue;
        }
        if buckets.len() >= MAX_TEDDY_BUCKETS {
            return None;
        }
        buckets.push((c1, c2_set));
    }
    if buckets.is_empty() {
        None
    } else {
        Some(buckets)
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
        let codes: Vec<u8> = (0..u8::try_from(MAX_SET_BYTES).unwrap() + 1).collect();
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

    /// Naive bucketed Cartesian Teddy reference: bit `i` is set iff some
    /// bucket b satisfies `bytes[i] == c1_b && bytes[i+1] in c2_set_b`,
    /// computed without nibble-table over-approximation. Used as the
    /// correctness oracle for the SIMD/scalar [`build_pair_bitset_teddy`].
    fn naive_teddy_bitset(bytes: &[u8], buckets: &[(u8, Vec<u8>)]) -> Vec<u64> {
        let n_words = bytes.len().div_ceil(64);
        let mut out = vec![0u64; n_words];
        if bytes.len() < 2 {
            return out;
        }
        for i in 0..bytes.len() - 1 {
            let b1 = bytes[i];
            let b2 = bytes[i + 1];
            for (c1, c2_set) in buckets {
                if b1 == *c1 && c2_set.contains(&b2) {
                    out[i >> 6] |= 1u64 << (i & 63);
                    break;
                }
            }
        }
        out
    }

    #[test]
    fn teddy_single_bucket_single_pair() {
        // One bucket, one pair: equivalent to Cartesian with |c1|=|c2|=1.
        let buckets = vec![(0x67u8, vec![0x6Fu8])]; // ('g', 'o')
        let bytes: Vec<u8> = b"google goggles agog gone".to_vec();
        let got = build_pair_bitset_teddy(&bytes, &buckets).expect("fits");
        let want = naive_teddy_bitset(&bytes, &buckets);
        assert_eq!(got, want);
    }

    #[test]
    fn teddy_shared_c1_zero_false_positives() {
        // Two buckets with disjoint c1's. A pure Cartesian over c1 union x
        // c2 union would admit cross pairs (c1_a, c2_b); bucketed Teddy
        // (one c1 per bucket) does not. We sanity-check that the bucketed
        // bitset omits the cross pairs.
        let buckets = vec![
            (0x67u8, vec![0x6Fu8]), // ('g', 'o')
            (0x61u8, vec![0x6Du8]), // ('a', 'm')
        ];
        // Includes valid pairs ('g','o'), ('a','m'), and crosses ('g','m'),
        // ('a','o') that the bucketed path must NOT mark.
        let bytes: Vec<u8> = b"go am gm ao gogo amam".to_vec();
        let got = build_pair_bitset_teddy(&bytes, &buckets).expect("fits");
        let want = naive_teddy_bitset(&bytes, &buckets);
        assert_eq!(got, want);

        // Spot-check: position of "gm" should NOT be in the bitset.
        let gm_pos = bytes.windows(2).position(|w| w == b"gm").unwrap();
        assert_eq!(got[gm_pos >> 6] & (1u64 << (gm_pos & 63)), 0);
        // Position of "ao" should NOT be in the bitset.
        let ao_pos = bytes.windows(2).position(|w| w == b"ao").unwrap();
        assert_eq!(got[ao_pos >> 6] & (1u64 << (ao_pos & 63)), 0);
        // Position of "go" SHOULD be in the bitset.
        let go_pos = bytes.windows(2).position(|w| w == b"go").unwrap();
        assert_ne!(got[go_pos >> 6] & (1u64 << (go_pos & 63)), 0);
    }

    #[test]
    fn teddy_eight_buckets_max() {
        // Exercise the full 8-bucket capacity.
        let buckets: Vec<(u8, Vec<u8>)> = (0..8u8).map(|b| (b * 17, vec![b * 13 + 1])).collect();
        let bytes: Vec<u8> = (0..1000)
            .map(|i| u8::try_from(i & 0xFF).unwrap().wrapping_mul(31))
            .collect();
        let got = build_pair_bitset_teddy(&bytes, &buckets).expect("fits");
        let want = naive_teddy_bitset(&bytes, &buckets);
        assert_eq!(got, want);
    }

    #[test]
    fn teddy_rejects_too_many_buckets() {
        let n_buckets = u8::try_from(MAX_TEDDY_BUCKETS).unwrap() + 1;
        let buckets: Vec<(u8, Vec<u8>)> = (0..n_buckets).map(|b| (b, vec![b + 1])).collect();
        let bytes = vec![0u8; 100];
        assert!(build_pair_bitset_teddy(&bytes, &buckets).is_none());
    }

    #[test]
    fn teddy_empty_input() {
        let buckets = vec![(0x67u8, vec![0x6Fu8])];
        let got = build_pair_bitset_teddy(&[], &buckets).expect("fits");
        assert!(got.is_empty());
    }

    #[test]
    fn teddy_single_byte_input() {
        // Length 1: no successor for c2, output must be empty/all-zeros.
        let buckets = vec![(0x67u8, vec![0x6Fu8])];
        let bytes = b"g";
        let got = build_pair_bitset_teddy(bytes, &buckets).expect("fits");
        assert_eq!(got, vec![0u64; 1]);
    }

    #[rstest]
    #[case(0)]
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
    #[case(129)]
    #[case(2049)]
    fn teddy_lengths_match_naive(#[case] len: usize) {
        // Two buckets exercising distinct c1's; pseudo-random byte stream.
        let buckets = vec![
            (0x42u8, vec![0x10u8, 0x11u8, 0x12u8]),
            (0x99u8, vec![0xFFu8]),
        ];
        let bytes: Vec<u8> = (0..len)
            .map(|i| {
                u8::try_from(i & 0xFF)
                    .unwrap()
                    .wrapping_mul(37)
                    .wrapping_add(7)
            })
            .collect();
        let got = build_pair_bitset_teddy(&bytes, &buckets).expect("fits");
        let want = naive_teddy_bitset(&bytes, &buckets);
        assert_eq!(got, want, "len={len}");
    }

    #[test]
    fn teddy_c2_nibble_phantom_is_documented() {
        // Bucket 0: c1='X' (0x58), c2_set = {0x12, 0x34}. Nibble tables
        // OR bits for low nibbles {2,4} and high nibbles {1,3}, so the
        // membership test admits phantoms 0x14 and 0x32. Confirm the
        // bitset reflects the over-approximation (this is by design and
        // is filtered downstream by the DFA verifier).
        let buckets = vec![(0x58u8, vec![0x12u8, 0x34u8])];
        let bytes = vec![0x58, 0x14]; // ('X', 0x14) — phantom hit expected
        let got = build_pair_bitset_teddy(&bytes, &buckets).expect("fits");
        // Bit 0 should be set due to nibble-table phantom on c2.
        assert_ne!(got[0] & 1, 0);
    }

    #[test]
    fn collect_pair_buckets_shared_c1_basic() {
        // Build a tiny synthetic transition matrix: 3 normal states (0..=2),
        // accept = 2, no escape states for simplicity (not exercising the
        // escape-state branch here).
        // States: 0 (start), 1 (matched first byte), 2 (accept).
        // Code A (0x10): 0 → 1
        // Code B (0x20): 1 → 2 (accept)
        // Code C (0x30): 0 → 1, 1 → 1 (stays)
        // All else: 0 (no progress).
        let accept_state = 2u8;
        let n_states = 3usize;
        let row1 = 256usize;
        let mut transitions = vec![0u8; n_states * 256];
        transitions[0x10] = 1;
        transitions[0x30] = 1;
        transitions[row1 + 0x20] = 2;
        transitions[row1 + 0x30] = 1; // stays at 1, NOT advancing
        // Accept row sticks (we won't iterate accept anyway).

        let c1_codes = vec![0x10u8, 0x30u8];
        let buckets =
            collect_pair_buckets_shared_c1(&transitions, &c1_codes, accept_state).unwrap();
        // Both 0x10 and 0x30 progress 0 → 1, so both are c1-eligible.
        assert_eq!(buckets.len(), 2);
        // Each bucket's c2 set should contain 0x20 (advances 1 → 2) but not
        // 0x30 (stays at 1).
        for (c1, c2_set) in &buckets {
            assert!(*c1 == 0x10 || *c1 == 0x30);
            assert!(c2_set.contains(&0x20), "c1={c1:#x} c2_set={c2_set:?}");
            assert!(!c2_set.contains(&0x30), "c1={c1:#x} c2_set={c2_set:?}");
        }
    }

    #[test]
    fn collect_pair_buckets_shared_c1_too_many_returns_none() {
        // Build a transition matrix where 9 distinct c1's all progress
        // 0 → 1 and 1 → accept on some c2. With > MAX_TEDDY_BUCKETS
        // pair-eligible c1's, collection bails to None.
        let accept_state = 2u8;
        let n_states = 3usize;
        let row1 = 256usize;
        let mut transitions = vec![0u8; n_states * 256];
        let n_codes = u8::try_from(MAX_TEDDY_BUCKETS).unwrap() + 1;
        let c1_codes: Vec<u8> = (0..n_codes).map(|b| b + 1).collect();
        for &c1 in &c1_codes {
            transitions[usize::from(c1)] = 1;
        }
        transitions[row1 + 0x20] = 2;
        let got = collect_pair_buckets_shared_c1(&transitions, &c1_codes, accept_state);
        assert!(got.is_none());
    }
}
