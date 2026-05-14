// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for multi-wildcard contains matching
//! (`LIKE '%seg1%seg2%...%segN%'`).
//!
//! Chains multiple KMP automata into a single linear state space. Each segment's
//! states are concatenated: phase k's accept state IS phase k+1's start state.
//! The final segment's accept is the global accept (sticky).
//!
//! ## State Layout
//!
//! ```text
//! Pattern: %abc%def%
//! Segments: ["abc", "def"]
//!
//! Global states:
//!   0: 0 of "abc" matched   (phase 0 start)
//!   1: 1 of "abc" matched
//!   2: 2 of "abc" matched
//!   3: all of "abc" matched = 0 of "def" matched  (phase 1 start)
//!   4: 1 of "def" matched
//!   5: 2 of "def" matched
//!   6: ACCEPT (all of "def" matched)
//! ```
//!
//! Each phase uses its own independent KMP failure function for backtracking.
//! The `%` between segments is implicit: once phase k accepts, phase k+1
//! searches for its needle anywhere in the remaining input.
//!
//! ## Optimizations
//!
//! Two optimizations, mirroring [`super::flat_contains::FlatContainsDfa`]:
//!
//! - **Per-phase SIMD seek-verify**: at each phase start state, use
//!   [`super::skip::SkipStrategy`] (memchr or bitmap) to skip non-progressing
//!   codes. A `[u64; 4]` bitmap provides O(1) phase-start detection.
//!
//! - **Decompress+memmem fallback**: for long strings (>28 codes), decompress
//!   the FSST codes and run sequential `memmem::find()` per segment.
//!
//! Uses the same escape-sentinel strategy as [`super::flat_contains::FlatContainsDfa`].

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_escape_only_encoded_pattern;
use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_failure_table;
use super::needle_bytes_absent_from_all_symbols;
use super::scan_to_bitbuf_with;
use super::skip::SkipStrategy;

/// Flat `u8` transition table DFA for multi-wildcard contains matching.
pub(crate) struct MultiContainsDfa {
    /// `transitions[state * 256 + code_byte]` -> next state.
    transitions: Vec<u8>,
    /// `escape_transitions[state * 256 + literal_byte]` -> next state for escaped bytes.
    escape_transitions: Vec<u8>,
    accept_state: u8,
    sentinel: u8,
    /// Per-phase skip strategy. `phase_skips[k]` is the strategy for phase k's
    /// start state.
    phase_skips: Vec<SkipStrategy>,
    /// Bitmap: bit `s` is set if state `s` is a phase start state.
    /// Indexed as `phase_start_bitmap[s >> 6] & (1 << (s & 63))`.
    phase_start_bitmap: [u64; 4],
    /// Maps a phase-start state to its index in `phase_skips`.
    /// Only valid for states where the corresponding bit is set in `phase_start_bitmap`.
    phase_index: [u8; 256],
    /// Symbol expansion table for decompress+memmem fallback.
    /// Layout: `expansions[code * 8 .. code * 8 + exp_lens[code]]`.
    expansions: Vec<u8>,
    /// Length of each symbol's expansion.
    exp_lens: Vec<u8>,
    /// Owned segment bytes for memmem fallback.
    segments: Vec<Vec<u8>>,
    /// If a compressed string has more codes than this, use decompress+memmem.
    decompress_threshold: usize,
    /// Compressed `[ESCAPE, anchor_seg[0], …, ESCAPE, anchor_seg[L-1]]`
    /// for the longest segment, populated when no symbol's expansion
    /// contains any byte from any segment. Every segment must appear in
    /// the row for a match, so the longest segment's encoded pattern is
    /// a sound (and most-selective) row-level prefilter; rows without it
    /// can never match, while rows with a hit are verified by the
    /// standard [`Self::matches`].
    escape_only_anchor_pattern: Option<Vec<u8>>,
}

impl MultiContainsDfa {
    /// Maximum total needle length (sum of all segments): need accept + sentinel in u8.
    pub(crate) const MAX_TOTAL_LEN: usize = u8::MAX as usize - 1;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        segments: &[&[u8]],
    ) -> VortexResult<Self> {
        let total_len: usize = segments.iter().map(|s| s.len()).sum();
        if total_len > Self::MAX_TOTAL_LEN {
            vortex_bail!(
                "total segment length {} exceeds maximum {} for multi-contains DFA",
                total_len,
                Self::MAX_TOTAL_LEN
            );
        }

        let accept_state = u8::try_from(total_len)
            .vortex_expect("MultiContainsDfa: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = chained_kmp_byte_transitions(segments, accept_state);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        // Compute phase offsets and build per-phase skip strategies.
        let mut phase_skips = Vec::with_capacity(segments.len());
        let mut phase_start_bitmap = [0u64; 4];
        let mut phase_index = [0u8; 256];
        let mut off = 0usize;
        for (k, seg) in segments.iter().enumerate() {
            let state = u8::try_from(off)
                .vortex_expect("MultiContainsDfa: phase start state must fit in u8");
            let row_start = usize::from(state) * 256;
            phase_skips.push(SkipStrategy::from_transition_row(
                &transitions[row_start..row_start + 256],
                state,
            ));
            phase_start_bitmap[usize::from(state >> 6)] |= 1u64 << (state & 63);
            phase_index[usize::from(state)] =
                u8::try_from(k).vortex_expect("MultiContainsDfa: phase index must fit in u8");
            off += seg.len();
        }

        // Build expansion table for decompress+memmem fallback.
        let n_symbols = symbols.len();
        let mut expansions = vec![0u8; n_symbols * 8];
        let mut exp_lens = vec![0u8; n_symbols];
        for (i, (sym, &len)) in symbols.iter().zip(symbol_lengths.iter()).enumerate() {
            let bytes = sym.to_u64().to_le_bytes();
            expansions[i * 8..i * 8 + usize::from(len)].copy_from_slice(&bytes[..usize::from(len)]);
            exp_lens[i] = len;
        }

        let segments_owned: Vec<Vec<u8>> = segments.iter().map(|s| s.to_vec()).collect();

        // Escape-only anchor: when no symbol's expansion contains any byte
        // from any segment, every segment's bytes in the decompressed
        // stream must come from `ESCAPE` pairs. Each segment therefore
        // appears in the compressed stream as a contiguous
        // `[ESCAPE, seg[0], …, ESCAPE, seg[L-1]]` block. A match
        // requires every segment to appear in order, so the longest
        // segment's encoded pattern (the most selective single test) is
        // a sound row-level prefilter — rows without it can't match.
        let escape_only_anchor_pattern =
            compute_escape_only_anchor(symbols, symbol_lengths, segments);

        Ok(Self {
            transitions,
            escape_transitions: byte_table,
            accept_state,
            sentinel,
            phase_skips,
            phase_start_bitmap,
            phase_index,
            expansions,
            exp_lens,
            segments: segments_owned,
            decompress_threshold: 28,
            escape_only_anchor_pattern,
        })
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        if codes.len() > self.decompress_threshold {
            return self.matches_decompress(codes);
        }
        self.matches_dfa(codes)
    }

    /// Specialized scan over `n` strings, returning a `BitBuffer` of accept
    /// results (XOR `negated`). The `matches` body is monomorphized into the
    /// bit-packing loop.
    #[inline]
    pub(crate) fn scan_to_bitbuf<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        if let Some(pattern) = self.escape_only_anchor_pattern.as_deref() {
            return self.scan_via_escape_only_anchor(n, offsets, all_bytes, pattern, negated);
        }
        scan_to_bitbuf_with(n, offsets, all_bytes, negated, |codes| self.matches(codes))
    }

    /// Prefilter via a single `memmem` for the longest segment's encoded
    /// pattern. Rows without a hit are guaranteed to miss; rows with a
    /// hit are verified through [`Self::matches`] (which checks the
    /// full ordered chain of segments).
    fn scan_via_escape_only_anchor<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        pattern: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        let mut bits = if negated {
            BitBufferMut::new_set(n)
        } else {
            BitBufferMut::new_unset(n)
        };
        if n == 0 || pattern.len() > all_bytes.len() {
            return bits.freeze();
        }
        debug_assert!(offsets.len() > n);

        let mut string_idx: usize = 0;
        // SAFETY: caller guarantees `offsets.len() > n`.
        let mut string_start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
        let mut string_end: usize = unsafe { *offsets.get_unchecked(1) }.as_();
        let mut last_processed_row: Option<usize> = None;

        for hit in memchr::memmem::find_iter(all_bytes, pattern) {
            while hit >= string_end {
                string_idx += 1;
                if string_idx >= n {
                    return bits.freeze();
                }
                string_start = string_end;
                // SAFETY: `string_idx < n` and `offsets.len() >= n + 1`.
                string_end = unsafe { *offsets.get_unchecked(string_idx + 1) }.as_();
            }
            if last_processed_row == Some(string_idx) {
                continue;
            }
            if hit + pattern.len() > string_end {
                last_processed_row = Some(string_idx);
                continue;
            }
            // SAFETY: `string_start <= string_end <= all_bytes.len()`.
            let row = unsafe { all_bytes.get_unchecked(string_start..string_end) };
            if self.matches(row) {
                // SAFETY: `string_idx < n`.
                unsafe {
                    if negated {
                        bits.unset_unchecked(string_idx);
                    } else {
                        bits.set_unchecked(string_idx);
                    }
                }
            }
            last_processed_row = Some(string_idx);
        }

        bits.freeze()
    }

    /// DFA path with per-phase seek-verify.
    #[inline]
    fn matches_dfa(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            // Phase-start fast path: SIMD-seek to next progressing code.
            if self.phase_start_bitmap[usize::from(state >> 6)] & (1u64 << (state & 63)) != 0 {
                let idx = usize::from(self.phase_index[usize::from(state)]);
                match self.phase_skips[idx].find_next_progressing(codes, pos) {
                    Some(next) => pos = next,
                    None => return false,
                }
            }

            let code = codes[pos];
            pos += 1;
            let next = self.transitions[usize::from(state) * 256 + usize::from(code)];
            if next == self.sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[usize::from(state) * 256 + usize::from(b)];
            } else {
                state = next;
            }
            if state == self.accept_state {
                return true;
            }
        }
        false
    }

    /// Decompress+memmem fallback: decompress FSST codes and run sequential
    /// `memmem::find()` for each segment (greedy-first-match).
    #[inline]
    fn matches_decompress(&self, codes: &[u8]) -> bool {
        let mut raw = Vec::with_capacity(codes.len() * 3);
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            if code == ESCAPE_CODE {
                if pos < codes.len() {
                    raw.push(codes[pos]);
                    pos += 1;
                }
            } else {
                let c = usize::from(code);
                if c < self.exp_lens.len() {
                    let len = usize::from(self.exp_lens[c]);
                    raw.extend_from_slice(&self.expansions[c * 8..c * 8 + len]);
                }
            }
        }

        // Greedy sequential memmem: find each segment in order.
        let mut search_start = 0;
        for segment in &self.segments {
            match memchr::memmem::find(&raw[search_start..], segment) {
                Some(offset) => search_start += offset + segment.len(),
                None => return false,
            }
        }
        true
    }
}

/// Build a chained KMP byte-level transition table for multiple segments.
///
/// States are the concatenation of each segment's progress states:
/// - Phase k occupies global states `offsets[k]..offsets[k] + segments[k].len()`
/// - Phase k's accept (= `offsets[k+1]`) is phase k+1's start state
/// - The final phase's accept is the global accept state (sticky)
///
/// Each phase has its own KMP failure function for intra-segment backtracking.
fn chained_kmp_byte_transitions(segments: &[&[u8]], accept_state: u8) -> Vec<u8> {
    let n_states = accept_state + 1;
    let mut table = vec![0u8; usize::from(n_states) * 256];

    // Phase offsets: offsets[k] = global state index for phase k's start
    let mut offsets = Vec::with_capacity(segments.len() + 1);
    let mut off = 0usize;
    for seg in segments {
        offsets.push(off);
        off += seg.len();
    }
    offsets.push(off); // = total_len = accept_state

    for (k, segment) in segments.iter().enumerate() {
        let base = offsets[k];
        let failure = kmp_failure_table(segment);

        for local_s in 0..segment.len() {
            let global_s = base + local_s;
            for byte in 0..256usize {
                let mut s = local_s;
                loop {
                    if byte == usize::from(segment[s]) {
                        s += 1;
                        break;
                    }
                    if s == 0 {
                        break;
                    }
                    s = usize::from(failure[s - 1]);
                }
                // If s == segment.len(), this maps to offsets[k+1] =
                // phase k+1's start (or the final accept for the last phase).
                table[global_s * 256 + byte] =
                    u8::try_from(base + s).vortex_expect("chained KMP state must fit in u8");
            }
        }
    }

    // Final accept state: sticky
    let acc = usize::from(accept_state);
    for byte in 0..256 {
        table[acc * 256 + byte] = accept_state;
    }

    table
}

/// Pick a row-level prefilter pattern for the escape-only regime.
///
/// Returns `Some(encoded_pattern)` for the longest segment iff:
///   - There are at least two segments (single-segment `%foo%` is routed
///     to [`super::folded_contains::FoldedContainsDfa`] /
///     [`super::flat_contains::FlatContainsDfa`], which have their own
///     escape-only paths).
///   - The longest segment has at least two bytes (the encoded pattern
///     is 4+ bytes; below that there's no win over the standard scan).
///   - No symbol's expansion contains any byte that appears in any
///     segment.
fn compute_escape_only_anchor(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    segments: &[&[u8]],
) -> Option<Vec<u8>> {
    if segments.len() < 2 {
        return None;
    }
    let longest = segments.iter().max_by_key(|s| s.len())?;
    if longest.len() < 2 || !super::needle_is_literal(longest) {
        // The encoded pattern is only well-defined when the longest
        // (anchor) segment is wildcard-free.
        return None;
    }
    // Union of every segment's bytes is what must be absent from symbols.
    let mut union = Vec::with_capacity(segments.iter().map(|s| s.len()).sum());
    for seg in segments {
        union.extend_from_slice(seg);
    }
    if !needle_bytes_absent_from_all_symbols(symbols, symbol_lengths, &union) {
        return None;
    }
    Some(build_escape_only_encoded_pattern(longest))
}
