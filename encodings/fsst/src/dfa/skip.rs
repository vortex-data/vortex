// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared skip strategies for DFA state-0 (or phase-start) fast paths.
//!
//! When a DFA is in a "searching" state (state 0 for single-segment contains,
//! or a phase-start state for multi-segment), most code bytes leave the state
//! unchanged. A skip strategy accelerates the search by jumping directly to
//! the next code that could advance the DFA.

use fsst::ESCAPE_CODE;
use vortex_array::dtype::IntegerPType;
use vortex_buffer::BitBufferMut;

/// Strategy for skipping non-progressing codes at a DFA start state.
///
/// Chosen at construction time based on how many code bytes "progress" the
/// DFA past the start state:
/// - 1–3 progressing codes → SIMD-accelerated `memchr` (32+ bytes/cycle)
/// - 4+ → packed `[u64; 4]` bitmap (branchless per-code check)
pub(super) enum SkipStrategy {
    /// Only 1 code byte progresses — use `memchr::memchr` (SIMD).
    Memchr1(u8),
    /// 2 code bytes progress — use `memchr::memchr2` (SIMD).
    Memchr2(u8, u8),
    /// 3 code bytes progress — use `memchr::memchr3` (SIMD).
    Memchr3(u8, u8, u8),
    /// More than 3 — use packed bitmap (1 cache line).
    Bitmap([u64; 4]),
}

impl SkipStrategy {
    /// Build a `SkipStrategy` from a 256-entry transition row.
    ///
    /// A code is "progressing" if `transition_row[code] != start_state`
    /// (it advances the DFA) or `code == ESCAPE_CODE` (the escaped literal
    /// might advance).
    pub(super) fn from_transition_row(transition_row: &[u8], start_state: u8) -> Self {
        debug_assert!(transition_row.len() >= 256);
        let mut prog_codes: Vec<u8> = Vec::new();
        for code in 0..=255u8 {
            if transition_row[usize::from(code)] != start_state || code == ESCAPE_CODE {
                prog_codes.push(code);
            }
        }

        match prog_codes.len() {
            0 => SkipStrategy::Bitmap([0u64; 4]),
            1 => SkipStrategy::Memchr1(prog_codes[0]),
            2 => SkipStrategy::Memchr2(prog_codes[0], prog_codes[1]),
            3 => SkipStrategy::Memchr3(prog_codes[0], prog_codes[1], prog_codes[2]),
            _ => {
                let mut bitmap = [0u64; 4];
                for &code in &prog_codes {
                    bitmap[usize::from(code >> 6)] |= 1u64 << (code & 63);
                }
                SkipStrategy::Bitmap(bitmap)
            }
        }
    }

    /// Find the next progressing code in `codes[start..]`.
    ///
    /// Returns the absolute index in `codes`, or `None` if no progressing code
    /// exists from `start` onward.
    #[inline]
    pub(super) fn find_next_progressing(&self, codes: &[u8], start: usize) -> Option<usize> {
        let slice = &codes[start..];
        match self {
            SkipStrategy::Memchr1(a) => memchr::memchr(*a, slice).map(|i| start + i),
            SkipStrategy::Memchr2(a, b) => memchr::memchr2(*a, *b, slice).map(|i| start + i),
            SkipStrategy::Memchr3(c0, c1, c2) => {
                memchr::memchr3(*c0, *c1, *c2, slice).map(|i| start + i)
            }
            SkipStrategy::Bitmap(bm) => {
                for (i, &code) in slice.iter().enumerate() {
                    if bm[usize::from(code >> 6)] & (1u64 << (code & 63)) != 0 {
                        return Some(start + i);
                    }
                }
                None
            }
        }
    }

    /// Build a per-string candidate `BitBufferMut` of length `n`, where bit
    /// `i` is set iff string `i` (the byte range `offsets[i]..offsets[i+1]`)
    /// contains at least one progressing (anchor) code.
    ///
    /// Strategy: run a single SIMD-accelerated pass over the entire `all_bytes`
    /// buffer to enumerate anchor byte positions, then walk the offsets in
    /// tandem with the (already-sorted) anchor positions to mark candidates.
    ///
    /// This is a "false positives are fine, false negatives are forbidden"
    /// pre-filter: the DFA still verifies each candidate.
    #[inline(always)]
    pub(super) fn build_candidate_bits<T: IntegerPType>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
    ) -> BitBufferMut {
        let mut candidates = BitBufferMut::new_unset(n);
        if n == 0 {
            return candidates;
        }
        let mut walker = StringWalker::new(offsets);
        match self {
            SkipStrategy::Memchr1(c0) => {
                for pos in memchr::memchr_iter(*c0, all_bytes) {
                    walker.mark_at(pos, &mut candidates);
                }
            }
            SkipStrategy::Memchr2(c0, c1) => {
                for pos in memchr::memchr2_iter(*c0, *c1, all_bytes) {
                    walker.mark_at(pos, &mut candidates);
                }
            }
            SkipStrategy::Memchr3(c0, c1, c2) => {
                for pos in memchr::memchr3_iter(*c0, *c1, *c2, all_bytes) {
                    walker.mark_at(pos, &mut candidates);
                }
            }
            SkipStrategy::Bitmap(_) => {
                let codes = self.anchor_codes();
                byte_cmp_scan(all_bytes, &codes, |pos| {
                    walker.mark_at(pos, &mut candidates);
                });
            }
        }
        candidates
    }

    /// Return the explicit list of progressing (anchor) code bytes.
    fn anchor_codes(&self) -> Vec<u8> {
        match self {
            SkipStrategy::Memchr1(c0) => vec![*c0],
            SkipStrategy::Memchr2(c0, c1) => vec![*c0, *c1],
            SkipStrategy::Memchr3(c0, c1, c2) => vec![*c0, *c1, *c2],
            SkipStrategy::Bitmap(bm) => {
                let mut out = Vec::with_capacity(8);
                for code in 0..=255u8 {
                    if bm[usize::from(code >> 6)] & (1u64 << (code & 63)) != 0 {
                        out.push(code);
                    }
                }
                out
            }
        }
    }
}

/// Byte-compare based SIMD scanner: invoke `mark` for each position in
/// `all_bytes` whose byte equals one of `codes`.
///
/// On x86_64 with AVX2 detected at runtime and `codes.len() <= 8`, processes
/// 32 bytes per iteration using `vpcmpeqb` + `vpor` + `vpmovmskb`. Falls back
/// to a chunked 64-bit bitmap-lookup scalar scan otherwise.
#[inline]
fn byte_cmp_scan(all_bytes: &[u8], codes: &[u8], mut mark: impl FnMut(usize)) {
    #[cfg(target_arch = "x86_64")]
    {
        if codes.len() <= 8 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime; the scanner reads only from
            // `all_bytes` (with bounds-checked tail) and uses
            // `_mm256_loadu_si256` which tolerates unaligned loads.
            unsafe { byte_cmp_scan_avx2(all_bytes, codes, mark) };
            return;
        }
    }
    byte_cmp_scan_scalar(all_bytes, codes, &mut mark);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn byte_cmp_scan_avx2<F: FnMut(usize)>(all_bytes: &[u8], codes: &[u8], mut mark: F) {
    use std::arch::x86_64::_mm256_cmpeq_epi8;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_movemask_epi8;
    use std::arch::x86_64::_mm256_or_si256;
    use std::arch::x86_64::_mm256_set1_epi8;
    use std::arch::x86_64::_mm256_setzero_si256;

    let n_codes = codes.len();
    debug_assert!((1..=8).contains(&n_codes));

    let mut cv = [_mm256_setzero_si256(); 8];
    for i in 0..n_codes {
        cv[i] = _mm256_set1_epi8(codes[i] as i8);
    }

    let total = all_bytes.len();
    let chunk = 32usize;
    let aligned_end = total - (total % chunk);

    let mut p = 0usize;
    while p < aligned_end {
        // SAFETY: p + 32 <= aligned_end <= total = all_bytes.len(), and
        // `_mm256_loadu_si256` tolerates unaligned loads.
        let v = unsafe { _mm256_loadu_si256(all_bytes.as_ptr().add(p) as *const _) };
        let mut acc = _mm256_cmpeq_epi8(v, cv[0]);
        if n_codes >= 2 {
            acc = _mm256_or_si256(acc, _mm256_cmpeq_epi8(v, cv[1]));
        }
        if n_codes >= 3 {
            acc = _mm256_or_si256(acc, _mm256_cmpeq_epi8(v, cv[2]));
        }
        if n_codes >= 4 {
            acc = _mm256_or_si256(acc, _mm256_cmpeq_epi8(v, cv[3]));
        }
        if n_codes >= 5 {
            acc = _mm256_or_si256(acc, _mm256_cmpeq_epi8(v, cv[4]));
        }
        if n_codes >= 6 {
            acc = _mm256_or_si256(acc, _mm256_cmpeq_epi8(v, cv[5]));
        }
        if n_codes >= 7 {
            acc = _mm256_or_si256(acc, _mm256_cmpeq_epi8(v, cv[6]));
        }
        if n_codes >= 8 {
            acc = _mm256_or_si256(acc, _mm256_cmpeq_epi8(v, cv[7]));
        }
        let mut mask = _mm256_movemask_epi8(acc) as u32;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            mark(p + bit);
            mask &= mask - 1;
        }
        p += chunk;
    }

    // Tail: scalar over the last <32 bytes.
    while p < total {
        // SAFETY: p < total = all_bytes.len().
        let b = unsafe { *all_bytes.get_unchecked(p) };
        for i in 0..n_codes {
            if codes[i] == b {
                mark(p);
                break;
            }
        }
        p += 1;
    }
}

#[inline]
fn byte_cmp_scan_scalar(all_bytes: &[u8], codes: &[u8], mark: &mut impl FnMut(usize)) {
    let mut bm = [0u64; 4];
    for &c in codes {
        bm[usize::from(c >> 6)] |= 1u64 << (c & 63);
    }
    for (i, &b) in all_bytes.iter().enumerate() {
        if bm[usize::from(b >> 6)] & (1u64 << (b & 63)) != 0 {
            mark(i);
        }
    }
}

/// Helper: walk anchor positions in tandem with sorted string offsets, marking
/// candidate strings in a `BitBufferMut`.
///
/// `mark_at(pos, &mut candidates)` advances the cursor through the offsets
/// until it finds the string whose byte range contains `pos`, then sets the
/// corresponding candidate bit.
struct StringWalker<'a, T: IntegerPType> {
    offsets: &'a [T],
    cur: usize,
    cur_end: usize,
    n: usize,
}

impl<'a, T: IntegerPType> StringWalker<'a, T> {
    #[inline(always)]
    fn new(offsets: &'a [T]) -> Self {
        let n = offsets.len().saturating_sub(1);
        let cur_end = if n > 0 { offsets[1].as_() } else { 0 };
        Self {
            offsets,
            cur: 0,
            cur_end,
            n,
        }
    }

    #[inline(always)]
    fn mark_at(&mut self, pos: usize, candidates: &mut BitBufferMut) {
        while self.cur < self.n && pos >= self.cur_end {
            self.cur += 1;
            if self.cur < self.n {
                // SAFETY: cur < n means cur+1 < offsets.len().
                self.cur_end = unsafe { self.offsets.get_unchecked(self.cur + 1) }.as_();
            }
        }
        if self.cur < self.n {
            // SAFETY: cur < n == candidates.len().
            unsafe { candidates.set_unchecked(self.cur) }
        }
    }
}
