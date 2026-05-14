// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for contains matching (`LIKE '%needle%'`).
//!
//! ## State-0 skip strategies
//!
//! The DFA is a sequential dependency chain. We break it while in state 0:
//!
//! - **memchr skip** (1-3 advancing codes): use `memchr`/`memchr2`/`memchr3`
//!   inline in the DFA loop. SIMD-accelerated, 32+ bytes/cycle. Only fires
//!   when the DFA drops back to state 0, so no overhead for high-match patterns
//!   where the DFA rarely returns to state 0.
//!
//! - **bitmap skip** (4+ advancing codes): packed `[u64; 4]` bitmap check.
//!   1 cache line, branchless per code.
//!
//! Additionally, a **memchr anchor prefilter** uses the longest FSST symbol
//! whose expansion is a substring of the needle. If that code byte is absent
//! from the compressed string, the needle can't match.
//!
//! For needles ≤ 127 bytes, [`super::folded_contains::FoldedContainsDfa`] is
//! preferred — it encodes the post-escape state directly so the inner loop
//! does a single table lookup per code byte with no sentinel branch.
//! [`FlatContainsDfa`] remains in use for longer needles (128–254 bytes).

use fsst::Symbol;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_escape_only_encoded_pattern;
use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;
use super::needle_bytes_absent_from_all_symbols;
use super::scan_to_bitbuf_with;
use super::skip::SkipStrategy;

/// Flat `u8` transition table DFA for contains matching.
pub(crate) struct FlatContainsDfa {
    /// `transitions[state * 256 + byte]` -> next state.
    transitions: Vec<u8>,
    /// `escape_transitions[state * 256 + byte]` -> next state for escaped bytes.
    escape_transitions: Vec<u8>,
    accept_state: u8,
    sentinel: u8,
    /// State-0 skip strategy.
    skip: SkipStrategy,
    /// Optional memchr anchor prefilter: a code byte that MUST appear for a match.
    anchor: Option<u8>,
    /// Compressed `[ESCAPE, needle[0], …, ESCAPE, needle[L-1]]`, populated
    /// when no symbol's expansion contains any needle byte. In that
    /// regime the only DFA-accepting compressed sequence is exactly this
    /// 2L-byte pattern, so [`Self::scan_to_bitbuf`] can prefilter rows
    /// with a single `memmem` over `all_bytes` rather than running the
    /// sentinel-branching per-code DFA on every row.
    escape_only_pattern: Option<Vec<u8>>,
}

impl FlatContainsDfa {
    /// Maximum needle length: need accept + sentinel to fit in u8.
    pub(crate) const MAX_NEEDLE_LEN: usize = u8::MAX as usize - 1;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        if needle.len() > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for flat contains DFA",
                needle.len(),
                Self::MAX_NEEDLE_LEN
            );
        }

        let accept_state = u8::try_from(needle.len())
            .vortex_expect("FlatContainsDfa: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        // NOTE: anchor prefilter disabled — FSST greedy compression means a
        // symbol's expansion can be a substring of the needle without that code
        // appearing (a longer symbol may cover the same bytes). The prefilter
        // would need to account for all possible encodings, which is complex.
        let anchor = None;

        // Escape-only fast path: when no symbol's expansion contains any
        // needle byte, every needle byte in the decompressed stream must
        // come from an `ESCAPE` pair and any symbol code resets the DFA
        // back to state 0. So the unique compressed sequence reaching
        // accept is exactly `[ESCAPE, needle[0], …, ESCAPE, needle[L-1]]`,
        // and `scan_to_bitbuf` can prefilter with a single `memmem` over
        // `all_bytes` whose length matches the encoded needle. Disabled
        // for L < 2 (no possible win over the existing scan path).
        let escape_only_pattern = (needle.len() >= 2
            && super::needle_is_literal(needle)
            && needle_bytes_absent_from_all_symbols(symbols, symbol_lengths, needle))
        .then(|| build_escape_only_encoded_pattern(needle));

        Ok(Self {
            transitions,
            escape_transitions: byte_table,
            accept_state,
            sentinel,
            skip,
            anchor,
            escape_only_pattern,
        })
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        // Anchor prefilter: if the anchor code is absent, no match possible.
        if let Some(a) = self.anchor
            && memchr::memchr(a, codes).is_none()
        {
            return false;
        }

        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            // State-0 fast path: SIMD skip to next advancing code.
            if state == 0 {
                match self.skip.find_next_progressing(codes, pos) {
                    Some(next) => pos = next,
                    None => return false,
                }
            }

            // Slow path: stateful DFA transition.
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
        if let Some(pattern) = self.escape_only_pattern.as_deref() {
            return self.scan_via_escape_only_memmem(n, offsets, all_bytes, pattern, negated);
        }
        scan_to_bitbuf_with(n, offsets, all_bytes, negated, |codes| self.matches(codes))
    }

    /// Single-`memmem` prefilter for the escape-only regime. Each hit is
    /// mapped to its containing row; rare false positives at literal-byte
    /// positions are eliminated by re-running [`Self::matches`] on the row.
    fn scan_via_escape_only_memmem<T>(
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
}

/// Find the best "anchor" symbol for the memchr prefilter.
///
/// Scans all symbols to find one whose expansion is the longest substring of
/// the needle. Returns `None` if no multi-byte symbol matches.
fn find_anchor_symbol(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Option<u8> {
    if needle.is_empty() {
        return None;
    }

    let n_symbols = symbols.len();
    let mut best_code: Option<u8> = None;
    let mut best_len: usize = 0;

    for code in 0..n_symbols {
        let sym_bytes = symbols[code].to_u64().to_le_bytes();
        let sym_len = usize::from(symbol_lengths[code]);
        if sym_len == 0 || sym_len > 8 || sym_len <= best_len || sym_len > needle.len() {
            continue;
        }
        let expansion = &sym_bytes[..sym_len];

        for start in 0..=needle.len() - sym_len {
            if &needle[start..start + sym_len] == expansion {
                best_len = sym_len;
                best_code = u8::try_from(code).ok();
                break;
            }
        }
    }

    if best_len >= 2 { best_code } else { None }
}
