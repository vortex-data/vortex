// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// The streaming Fat Teddy pass threads `tables`, `buckets`, `offsets`,
// `bits`, and `verify` through every architecture variant; splitting
// would obscure the SIMD code without making it more readable.
#![allow(clippy::too_many_arguments)]

//! # Fat Teddy: multi-needle OR prefilter
//!
//! Implements a Hyperscan-inspired Fat Teddy prefilter for evaluating
//! `LIKE x OR LIKE y OR ...` (or `LIKE IN (...)`) on FSST-compressed
//! strings with a single streaming pass over the input bytes.
//!
//! ## Why
//!
//! A naive implementation runs each needle as a separate
//! [`FoldedContainsDfa`](super::folded_contains::FoldedContainsDfa) scan
//! plus a final boolean OR — `N×` slower than necessary. Fat Teddy
//! collapses up to eight needles into a single streaming PSHUFB-Mula
//! pass: each bucket carries a single bit through the SIMD pipeline,
//! and the per-bucket verifier (the existing single-pattern
//! [`FoldedContainsDfa::matches`]) only runs on the positions where the
//! candidate mask fired for that bucket.
//!
//! ## Algorithm
//!
//! 1. For each needle build the usual [`FoldedContainsDfa`].
//! 2. Bucket-pack up to eight needles into bit positions `0..8`. A
//!    bucket carries the union of its needles' progressing c1 codes
//!    (plus single-step-accept c1 codes) and the union of their c2
//!    successor sets. Bucket-packing is greedy: pick the bucket whose
//!    merge least increases `|c1_union| * |c2_union|` (a coarse
//!    FP-rate proxy).
//! 3. For each 32-byte block of input, two PSHUFB-Mula lookups per
//!    byte position produce per-bucket bit lanes; AND-ing the c1 and
//!    c2 lookups intersects per-bucket. A `cmpgt + movemask`
//!    collapses 32 bytes into a 32-bit "any bucket fired" mask,
//!    while the underlying per-byte `pair` register is stored to a
//!    stack buffer so the candidate-handling loop can read the
//!    per-bucket bits.
//! 4. For each candidate position, the per-bucket bits indicate which
//!    bucket fired, and we run the verifier for each needle assigned
//!    to that bucket.
//!
//! ## Limits and fallbacks
//!
//! - Up to eight needles per Fat Teddy pass (one bit per bucket). The
//!   caller chunks larger needle sets into groups of eight and
//!   OR-merges across passes.
//! - Needles whose progressing c1 set includes [`ESCAPE_CODE`] are
//!   currently routed to an N-pass fallback. Cross-bucket FDR
//!   handling for ESCAPE-anchored needles is a follow-up — see the
//!   TODO in [`pack_needles`].
//! - Needles that don't expose a `bucketed_pair_codes` set (very short
//!   needles, escape-only patterns) are routed through the N-pass
//!   fallback. This matches the existing single-pattern routing
//!   ladder: those needles take faster specialized paths anyway.
//!
//! ## Verification
//!
//! Verification is exact: each candidate position calls
//! [`FoldedContainsDfa::matches`] on the current row. The Fat Teddy
//! prefilter is therefore only an over-approximation — a candidate
//! position implies "some needle *might* match this row at this
//! offset", and `matches` either confirms or refutes for each
//! assigned needle.

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
use core::arch::x86_64::_mm256_or_si256;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_set1_epi8;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_setzero_si256;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_shuffle_epi8;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_srli_epi64;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_storeu_si256;

use fsst::ESCAPE_CODE;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;

use super::folded_contains::FoldedContainsDfa;

/// Maximum number of buckets per Fat Teddy pass. Each bucket carries a
/// single bit through the SIMD pipeline; the PSHUFB-Mula nibble tables
/// can encode up to eight unique bits.
pub(super) const FAT_TEDDY_BUCKETS: usize = 8;

/// A single bucket: union of progressing c1 codes and c2 codes across
/// all needles assigned to the bucket, plus the indices (into the
/// caller-provided per-needle DFA array) of those needles.
#[derive(Debug, Clone)]
pub(super) struct Bucket {
    /// Union of progressing c1 codes across all needles in this bucket.
    c1: Vec<u8>,
    /// Union of valid c2 codes across all needles in this bucket.
    c2: Vec<u8>,
    /// Needle indices assigned to this bucket. The verifier runs
    /// [`FoldedContainsDfa::matches`] for each of these on every
    /// candidate position.
    needles: Vec<u16>,
}

impl Bucket {
    fn new() -> Self {
        Self {
            c1: Vec::new(),
            c2: Vec::new(),
            needles: Vec::new(),
        }
    }

    /// Approximate FP-rate proxy: `|c1| * |c2|`. The candidate density
    /// per position is roughly `|c1| / 256 * |c2| / 256`, so packing
    /// to minimize the product across buckets keeps the verifier
    /// dispatch budget tight.
    fn fp_proxy(&self) -> usize {
        self.c1.len() * self.c2.len()
    }

    /// FP proxy after merging `(c1, c2)` into this bucket. Doesn't
    /// mutate.
    fn fp_proxy_with(&self, c1: &[u8], c2: &[u8]) -> usize {
        let mut c1_seen = [false; 256];
        let mut c2_seen = [false; 256];
        for &b in &self.c1 {
            c1_seen[usize::from(b)] = true;
        }
        for &b in &self.c2 {
            c2_seen[usize::from(b)] = true;
        }
        let mut c1_count = self.c1.len();
        let mut c2_count = self.c2.len();
        for &b in c1 {
            if !c1_seen[usize::from(b)] {
                c1_seen[usize::from(b)] = true;
                c1_count += 1;
            }
        }
        for &b in c2 {
            if !c2_seen[usize::from(b)] {
                c2_seen[usize::from(b)] = true;
                c2_count += 1;
            }
        }
        c1_count * c2_count
    }

    fn add_needle(&mut self, idx: u16, c1: &[u8], c2: &[u8]) {
        let mut c1_seen = [false; 256];
        let mut c2_seen = [false; 256];
        for &b in &self.c1 {
            c1_seen[usize::from(b)] = true;
        }
        for &b in &self.c2 {
            c2_seen[usize::from(b)] = true;
        }
        for &b in c1 {
            if !c1_seen[usize::from(b)] {
                c1_seen[usize::from(b)] = true;
                self.c1.push(b);
            }
        }
        for &b in c2 {
            if !c2_seen[usize::from(b)] {
                c2_seen[usize::from(b)] = true;
                self.c2.push(b);
            }
        }
        self.needles.push(idx);
    }
}

/// Per-needle prefilter info extracted from a [`FoldedContainsDfa`].
struct NeedleInfo {
    /// The needle's original index in the caller-provided needle list.
    idx: u16,
    /// Progressing c1 codes — the union of all bucketed c1's plus any
    /// SSA c1's. ESCAPE_CODE is included if present.
    c1: Vec<u8>,
    /// c2 codes — the union of all c2 sets across the needle's
    /// bucketed pair codes.
    c2: Vec<u8>,
}

impl NeedleInfo {
    /// Build the prefilter info for `dfa`, or `None` if the needle
    /// can't participate in a Fat Teddy pass (e.g. no progressing
    /// codes, or no pair buckets).
    fn build(idx: u16, dfa: &FoldedContainsDfa) -> Option<Self> {
        let pair_buckets = dfa.bucketed_pair_codes_slice()?;
        if pair_buckets.is_empty() {
            return None;
        }
        // Union c1 across the needle's own pair buckets, plus any SSA
        // c1's. Unlike the single-pattern fused-Teddy path which folds
        // SSA via a side PSHUFB on the same v1 register, Fat Teddy
        // simply unions SSA into c1 — since the bucket's c2 union is
        // the union of all c2 sets, any byte succeeding an SSA c1
        // satisfies the c2 constraint. SSA candidates are verified by
        // `matches` regardless, so the over-approximation is harmless
        // beyond a small FP-rate increase.
        let mut c1_seen = [false; 256];
        let mut c1: Vec<u8> = Vec::new();
        let mut c2_seen = [false; 256];
        let mut c2: Vec<u8> = Vec::new();
        for (b1, c2_set) in pair_buckets {
            if !c1_seen[usize::from(*b1)] {
                c1_seen[usize::from(*b1)] = true;
                c1.push(*b1);
            }
            for &b2 in c2_set {
                if !c2_seen[usize::from(b2)] {
                    c2_seen[usize::from(b2)] = true;
                    c2.push(b2);
                }
            }
        }
        if let Some(ssa) = dfa.single_step_accept_codes_slice() {
            for &b in ssa {
                if !c1_seen[usize::from(b)] {
                    c1_seen[usize::from(b)] = true;
                    c1.push(b);
                }
            }
        }
        Some(Self { idx, c1, c2 })
    }

    /// Whether the needle's progressing c1 set includes the FSST
    /// [`ESCAPE_CODE`]. Such needles are routed through the N-pass
    /// fallback: cross-bucket FDR handling for ESCAPE-anchored
    /// candidates is a follow-up.
    fn has_escape_c1(&self) -> bool {
        self.c1.contains(&ESCAPE_CODE)
    }
}

/// Bucket-pack the given needle DFAs into at most [`FAT_TEDDY_BUCKETS`]
/// buckets. Returns `(buckets, fallback_indices)` where
/// `fallback_indices` is the set of needles that couldn't participate
/// (ESCAPE-anchored, no usable pair buckets, or didn't fit in a Fat
/// Teddy pass).
///
/// Needles in `fallback_indices` must be evaluated by the caller via
/// per-needle scans and OR-merged into the final bitbuf.
pub(super) fn pack_needles(needle_dfas: &[FoldedContainsDfa]) -> (Vec<Bucket>, Vec<u16>) {
    let mut fallback: Vec<u16> = Vec::new();
    let mut infos: Vec<NeedleInfo> = Vec::with_capacity(needle_dfas.len());

    for (idx, dfa) in needle_dfas.iter().enumerate() {
        // Caller ensures `needle_dfas.len() <= u16::MAX` via
        // `MultiNeedleMatcher::MAX_NEEDLES`. Saturate defensively.
        let idx_u16 = u16::try_from(idx).unwrap_or(u16::MAX);
        match NeedleInfo::build(idx_u16, dfa) {
            Some(info) if info.has_escape_c1() => {
                // TODO(fat-teddy): cross-bucket FDR handling for
                // ESCAPE-anchored needles. Today ESCAPE-anchored
                // needles fall back to N-pass scans. A correct
                // implementation would add a separate per-block
                // ESCAPE-anchored lookup with per-needle c2 verifiers.
                fallback.push(idx_u16);
            }
            Some(info) => infos.push(info),
            None => fallback.push(idx_u16),
        }
    }

    // Sort by ascending `|c1| + |c2|` so the easiest-to-pack needles
    // populate buckets first. Coarse heuristic but works well on real
    // corpora where needles vary in c1-set size by 5–10×.
    infos.sort_by_key(|i| i.c1.len() + i.c2.len());

    let mut buckets: Vec<Bucket> = Vec::with_capacity(FAT_TEDDY_BUCKETS);
    for info in infos {
        // If we have room for a new bucket AND the best merge would
        // cost more than just creating a new bucket, open a new one.
        // Otherwise pick the bucket whose merge increases the FP
        // proxy the least.
        let baseline = info.c1.len() * info.c2.len();
        let mut best: Option<(usize, usize)> = None;
        for (b_idx, bucket) in buckets.iter().enumerate() {
            let proposed = bucket.fp_proxy_with(&info.c1, &info.c2);
            let delta = proposed.saturating_sub(bucket.fp_proxy());
            best = match best {
                Some((_, d)) if d <= delta => best,
                _ => Some((b_idx, delta)),
            };
        }
        let open_new = buckets.len() < FAT_TEDDY_BUCKETS
            && match best {
                Some((_, delta)) => delta > baseline,
                None => true,
            };
        if open_new {
            let mut b = Bucket::new();
            b.add_needle(info.idx, &info.c1, &info.c2);
            buckets.push(b);
        } else if let Some((b_idx, _)) = best {
            buckets[b_idx].add_needle(info.idx, &info.c1, &info.c2);
        } else {
            fallback.push(info.idx);
        }
    }

    (buckets, fallback)
}

/// PSHUFB nibble tables for the per-bucket c1 and c2 byte sets.
struct FatTeddyTables {
    c1_lo: [u8; 16],
    c1_hi: [u8; 16],
    c2_lo: [u8; 16],
    c2_hi: [u8; 16],
}

impl FatTeddyTables {
    fn build(buckets: &[Bucket]) -> Self {
        debug_assert!(buckets.len() <= FAT_TEDDY_BUCKETS);
        let mut c1_lo = [0u8; 16];
        let mut c1_hi = [0u8; 16];
        let mut c2_lo = [0u8; 16];
        let mut c2_hi = [0u8; 16];
        for (b_idx, bucket) in buckets.iter().enumerate() {
            let bit = 1u8 << b_idx;
            for &c1 in &bucket.c1 {
                c1_lo[usize::from(c1 & 0x0F)] |= bit;
                c1_hi[usize::from(c1 >> 4)] |= bit;
            }
            for &c2 in &bucket.c2 {
                c2_lo[usize::from(c2 & 0x0F)] |= bit;
                c2_hi[usize::from(c2 >> 4)] |= bit;
            }
        }
        Self {
            c1_lo,
            c1_hi,
            c2_lo,
            c2_hi,
        }
    }
}

/// Per-bucket nibble lookup for a single input byte. Returns a u8 with
/// bit `b` set iff bucket `b`'s c1 set contains the byte.
#[inline]
fn lookup_c1(tables: &FatTeddyTables, b: u8) -> u8 {
    tables.c1_lo[usize::from(b & 0x0F)] & tables.c1_hi[usize::from(b >> 4)]
}

/// Same as [`lookup_c1`] but for c2.
#[inline]
fn lookup_c2(tables: &FatTeddyTables, b: u8) -> u8 {
    tables.c2_lo[usize::from(b & 0x0F)] & tables.c2_hi[usize::from(b >> 4)]
}

/// Run a single Fat Teddy pass: for each candidate `(position,
/// bucket)`, fire the verifier on every needle assigned to that
/// bucket. Updates `bits[row_idx]` per the standard set/unset
/// convention (XOR with `negated`).
///
/// `verify` is `FnMut(needle_idx, row_codes) -> bool` — the caller
/// looks up the needle's DFA and runs `matches` on the row.
pub(super) fn fat_teddy_or_scan<T, V>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    buckets: &[Bucket],
    negated: bool,
    mut verify: V,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(u16, &[u8]) -> bool,
{
    let mut bits = if negated {
        BitBufferMut::new_set(n)
    } else {
        BitBufferMut::new_unset(n)
    };
    if n == 0 || buckets.is_empty() || all_bytes.len() < 2 {
        return bits.freeze();
    }
    let tables = FatTeddyTables::build(buckets);

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 just detected.
            unsafe {
                fat_teddy_pass_avx2(
                    &tables,
                    buckets,
                    n,
                    offsets,
                    all_bytes,
                    negated,
                    &mut bits,
                    &mut verify,
                );
            }
            return bits.freeze();
        }
    }
    fat_teddy_pass_scalar(
        &tables,
        buckets,
        n,
        offsets,
        all_bytes,
        negated,
        &mut bits,
        &mut verify,
    );
    bits.freeze()
}

/// Verify one candidate: for each needle in `bucket`, run
/// [`FoldedContainsDfa::matches`] (via `verify`) on the current row.
/// Avoids re-verifying when the row's result bit is already in the
/// "match" state.
#[inline]
fn verify_candidate<V>(
    bucket: &Bucket,
    all_bytes: &[u8],
    row_start: usize,
    row_end: usize,
    row_idx: usize,
    bits: &mut BitBufferMut,
    negated: bool,
    verify: &mut V,
) where
    V: FnMut(u16, &[u8]) -> bool,
{
    let already = bits.value(row_idx);
    if (!negated && already) || (negated && !already) {
        return;
    }
    debug_assert!(row_start <= row_end && row_end <= all_bytes.len());
    // SAFETY: row_start <= row_end <= all_bytes.len().
    let row = unsafe { all_bytes.get_unchecked(row_start..row_end) };
    for &needle_idx in &bucket.needles {
        if verify(needle_idx, row) {
            // SAFETY: row_idx < n (caller invariant).
            unsafe {
                if negated {
                    bits.unset_unchecked(row_idx);
                } else {
                    bits.set_unchecked(row_idx);
                }
            }
            return;
        }
    }
}

/// Scalar Fat Teddy pass. Iterates input bytes one at a time,
/// computing the per-byte `(c1_bits, c2_bits)` and AND-ing to get the
/// per-position per-bucket candidate byte. When non-zero, peel bucket
/// bits and dispatch the verifier.
fn fat_teddy_pass_scalar<T, V>(
    tables: &FatTeddyTables,
    buckets: &[Bucket],
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(u16, &[u8]) -> bool,
{
    let len = all_bytes.len();
    if len < 2 {
        return;
    }
    debug_assert!(offsets.len() > n);
    // SAFETY: caller guarantees offsets.len() > n.
    let mut row_idx: usize = 0;
    let mut row_start: usize = unsafe { (*offsets.get_unchecked(0)).as_() };
    let mut row_end: usize = unsafe { (*offsets.get_unchecked(1)).as_() };

    for i in 0..len - 1 {
        // SAFETY: i + 1 < len.
        let b1 = unsafe { *all_bytes.get_unchecked(i) };
        let b2 = unsafe { *all_bytes.get_unchecked(i + 1) };
        let c1b = lookup_c1(tables, b1);
        let c2b = lookup_c2(tables, b2);
        let mut hit = c1b & c2b;
        if hit == 0 {
            continue;
        }
        // Advance row to include position i.
        while i >= row_end {
            row_idx += 1;
            if row_idx >= n {
                return;
            }
            row_start = row_end;
            // SAFETY: row_idx + 1 <= n, offsets.len() >= n + 1.
            row_end = unsafe { (*offsets.get_unchecked(row_idx + 1)).as_() };
        }
        while hit != 0 {
            let b_idx = hit.trailing_zeros() as usize;
            hit &= hit - 1;
            debug_assert!(b_idx < buckets.len());
            // SAFETY: b_idx < buckets.len() by table construction.
            let bucket = unsafe { buckets.get_unchecked(b_idx) };
            verify_candidate(
                bucket, all_bytes, row_start, row_end, row_idx, bits, negated, verify,
            );
            // Early-exit if row already decided to "match".
            let already = bits.value(row_idx);
            if (!negated && already) || (negated && !already) {
                break;
            }
        }
    }
}

/// AVX2 Fat Teddy pass. Processes 32 input bytes per iteration. For
/// each block, computes `pair = c1_bits(v1) & c2_bits(v2)` over the
/// 32-byte vectors `v1 = all_bytes[i..i+32]` and `v2 =
/// all_bytes[i+1..i+33]`, stores `pair` to a stack `[u8; 32]` buffer,
/// then peels candidates via tzcnt on the "any bit set" 32-bit mask
/// from `cmpgt + movemask`. For each candidate, the per-position byte
/// in the buffer carries the per-bucket bits; iterate those bits to
/// dispatch the right bucket's verifier.
///
/// # Safety
///
/// Caller must verify AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(unsafe_op_in_unsafe_fn)]
unsafe fn fat_teddy_pass_avx2<T, V>(
    tables: &FatTeddyTables,
    buckets: &[Bucket],
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    bits: &mut BitBufferMut,
    verify: &mut V,
) where
    T: vortex_array::dtype::IntegerPType,
    V: FnMut(u16, &[u8]) -> bool,
{
    let len = all_bytes.len();
    if len < 2 {
        return;
    }
    let main_len = ((len - 1) >> 5) << 5;
    let c1_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1_lo.as_ptr() as *const __m128i));
    let c1_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c1_hi.as_ptr() as *const __m128i));
    let c2_lo =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2_lo.as_ptr() as *const __m128i));
    let c2_hi =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.c2_hi.as_ptr() as *const __m128i));
    let zero = _mm256_setzero_si256();
    let nibble_mask = _mm256_set1_epi8(0x0F);

    debug_assert!(offsets.len() > n);
    let mut row_idx: usize = 0;
    let mut row_start: usize = (*offsets.get_unchecked(0)).as_();
    let mut row_end: usize = (*offsets.get_unchecked(1)).as_();
    let ptr = all_bytes.as_ptr();
    let mut pair_buf = [0u8; 32];
    let mut i: usize = 0;
    while i < main_len {
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
        let pos = _mm256_cmpgt_epi8(pair, zero);
        let neg = _mm256_cmpgt_epi8(zero, pair);
        let hit = _mm256_or_si256(pos, neg);
        let mut mask = _mm256_movemask_epi8(hit) as u32;
        if mask != 0 {
            _mm256_storeu_si256(pair_buf.as_mut_ptr() as *mut __m256i, pair);
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let cand = i + lane;
                while cand >= row_end {
                    row_idx += 1;
                    if row_idx >= n {
                        return;
                    }
                    row_start = row_end;
                    row_end = (*offsets.get_unchecked(row_idx + 1)).as_();
                }
                let mut bucket_bits = pair_buf[lane];
                while bucket_bits != 0 {
                    let b_idx = bucket_bits.trailing_zeros() as usize;
                    bucket_bits &= bucket_bits - 1;
                    debug_assert!(b_idx < buckets.len());
                    let bucket = buckets.get_unchecked(b_idx);
                    verify_candidate(
                        bucket, all_bytes, row_start, row_end, row_idx, bits, negated, verify,
                    );
                    let already = bits.value(row_idx);
                    if (!negated && already) || (negated && !already) {
                        break;
                    }
                }
            }
        }
        i += 32;
    }
    // Tail scalar.
    if len > 1 {
        for j in i..len - 1 {
            let b1 = *all_bytes.get_unchecked(j);
            let b2 = *all_bytes.get_unchecked(j + 1);
            let c1b = lookup_c1(tables, b1);
            let c2b = lookup_c2(tables, b2);
            let mut hit = c1b & c2b;
            if hit == 0 {
                continue;
            }
            while j >= row_end {
                row_idx += 1;
                if row_idx >= n {
                    return;
                }
                row_start = row_end;
                row_end = (*offsets.get_unchecked(row_idx + 1)).as_();
            }
            while hit != 0 {
                let b_idx = hit.trailing_zeros() as usize;
                hit &= hit - 1;
                debug_assert!(b_idx < buckets.len());
                let bucket = buckets.get_unchecked(b_idx);
                verify_candidate(
                    bucket, all_bytes, row_start, row_end, row_idx, bits, negated, verify,
                );
                let already = bits.value(row_idx);
                if (!negated && already) || (negated && !already) {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bucket-level FP proxy arithmetic.
    #[test]
    fn test_bucket_fp_proxy() {
        let mut b0 = Bucket::new();
        b0.add_needle(0, b"a", b"b");
        assert_eq!(b0.fp_proxy(), 1);
        // Adding a disjoint (c1, c2) grows the proxy to 2 × 2 = 4.
        let proxy_with = b0.fp_proxy_with(b"c", b"d");
        assert_eq!(proxy_with, 4);
        // Adding overlap with c1='a' but new c2='d' grows to 1 × 2 = 2.
        let proxy_with = b0.fp_proxy_with(b"a", b"d");
        assert_eq!(proxy_with, 2);
    }

    /// Nibble lookup correctness.
    #[test]
    fn test_lookup_c1_c2() {
        let mut b = Bucket::new();
        b.add_needle(0, b"a", b"b");
        let tables = FatTeddyTables::build(&[b]);
        assert_eq!(lookup_c1(&tables, b'a'), 1);
        assert_eq!(lookup_c1(&tables, b'b'), 0);
        assert_eq!(lookup_c2(&tables, b'b'), 1);
        assert_eq!(lookup_c2(&tables, b'a'), 0);
    }

    /// Two needles whose (c1, c2) don't overlap should land in two
    /// different buckets. Note we test through [`pack_needles`], which
    /// requires real `FoldedContainsDfa`s; we test the underlying
    /// `Bucket` packing arithmetic directly here.
    #[test]
    fn test_two_disjoint_needles_two_buckets() {
        // Simulate the post-info-extraction packing loop.
        let mut buckets: Vec<Bucket> = Vec::new();
        // Two needles with disjoint c1's and c2's.
        let need_a_c1 = vec![b'a'];
        let need_a_c2 = vec![b'b'];
        let need_b_c1 = vec![b'c'];
        let need_b_c2 = vec![b'd'];

        // First needle creates bucket 0.
        let mut b0 = Bucket::new();
        b0.add_needle(0, &need_a_c1, &need_a_c2);
        buckets.push(b0);
        // Second needle: best merge with bucket 0 adds 4 - 1 = 3 to
        // proxy, while baseline of new bucket is 1. Since baseline <
        // delta, open a new bucket.
        let baseline = need_b_c1.len() * need_b_c2.len();
        let proposed = buckets[0].fp_proxy_with(&need_b_c1, &need_b_c2);
        let delta = proposed - buckets[0].fp_proxy();
        assert!(delta > baseline);
        let mut b1 = Bucket::new();
        b1.add_needle(1, &need_b_c1, &need_b_c2);
        buckets.push(b1);

        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].needles, vec![0]);
        assert_eq!(buckets[1].needles, vec![1]);
    }
}
