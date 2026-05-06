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
use vortex_array::dtype::IntegerPType;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;
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

        Ok(Self {
            transitions,
            escape_transitions: byte_table,
            accept_state,
            sentinel,
            skip,
            anchor,
        })
    }

    /// Scan `n` FSST-compressed strings to a bit-packed boolean output.
    ///
    /// Tight 64-string-block loop that packs per-string DFA results directly
    /// into 64-bit words, avoiding the closure-call overhead of
    /// `BitBuffer::collect_bool`. Uses the state-0 skip strategy as a global
    /// anchor pre-scan only when it's a `Memchr1/2/3` (SIMD-friendly): the
    /// non-candidate path then skips the per-string DFA call entirely.
    /// Otherwise (Bitmap), goes directly to per-string DFA.
    pub(crate) fn scan_to_bitbuf<T: IntegerPType>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer {
        let scan_skip: Option<&SkipStrategy> = match &self.skip {
            SkipStrategy::Memchr1(_)
            | SkipStrategy::Memchr2(_, _)
            | SkipStrategy::Memchr3(_, _, _) => Some(&self.skip),
            SkipStrategy::Bitmap(_) => None,
        };

        if let Some(skip) = scan_skip {
            self.scan_with_anchor(n, offsets, all_bytes, negated, skip)
        } else {
            self.scan_no_anchor(n, offsets, all_bytes, negated)
        }
    }

    fn scan_no_anchor<T: IntegerPType>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer {
        use vortex_buffer::BufferMut;
        let mut out = BufferMut::<u64>::with_capacity(n.div_ceil(64));
        let chunks = n / 64;
        let remainder = n % 64;
        let neg_word: u64 = if negated { u64::MAX } else { 0 };

        for chunk in 0..chunks {
            let mut packed_match: u64 = 0;
            let base = chunk * 64;
            for bit in 0..64usize {
                let i = base + bit;
                // SAFETY: i + 1 <= offsets.len() - 1.
                let start: usize = unsafe { offsets.get_unchecked(i) }.as_();
                let end: usize = unsafe { offsets.get_unchecked(i + 1) }.as_();
                // SAFETY: s..e is valid in all_bytes.
                let codes = unsafe { all_bytes.get_unchecked(start..end) };
                if self.matches(codes) {
                    packed_match |= 1u64 << bit;
                }
            }
            let packed = packed_match ^ neg_word;
            // SAFETY: out has capacity.
            unsafe { out.push_unchecked(packed) };
        }

        if remainder != 0 {
            let mut packed_match: u64 = 0;
            let base = chunks * 64;
            for bit in 0..remainder {
                let i = base + bit;
                // SAFETY: i + 1 <= offsets.len() - 1.
                let start: usize = unsafe { offsets.get_unchecked(i) }.as_();
                let end: usize = unsafe { offsets.get_unchecked(i + 1) }.as_();
                // SAFETY: s..e is valid in all_bytes.
                let codes = unsafe { all_bytes.get_unchecked(start..end) };
                if self.matches(codes) {
                    packed_match |= 1u64 << bit;
                }
            }
            let mask = if remainder == 64 {
                u64::MAX
            } else {
                (1u64 << remainder) - 1
            };
            let packed = (packed_match ^ neg_word) & mask;
            // SAFETY: out has capacity.
            unsafe { out.push_unchecked(packed) };
        }

        BitBuffer::new(out.into_byte_buffer().freeze(), n)
    }

    fn scan_with_anchor<T: IntegerPType>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
        skip: &SkipStrategy,
    ) -> BitBuffer {
        let candidates = skip.build_candidate_bits(n, offsets, all_bytes);
        let cand_bytes = candidates.as_slice();

        use vortex_buffer::BufferMut;
        let mut out = BufferMut::<u64>::with_capacity(n.div_ceil(64));
        let chunks = n / 64;
        let remainder = n % 64;
        let neg_word: u64 = if negated { u64::MAX } else { 0 };

        for chunk in 0..chunks {
            // SAFETY: chunk * 8 + 8 <= candidates byte capacity.
            let cand_word: u64 = unsafe {
                let p = cand_bytes.as_ptr().add(chunk * 8);
                u64::from_le_bytes([
                    *p,
                    *p.add(1),
                    *p.add(2),
                    *p.add(3),
                    *p.add(4),
                    *p.add(5),
                    *p.add(6),
                    *p.add(7),
                ])
            };
            let packed = if cand_word == 0 {
                neg_word
            } else {
                let mut packed_match: u64 = 0;
                let mut bm = cand_word;
                while bm != 0 {
                    let bit = bm.trailing_zeros() as usize;
                    let i = chunk * 64 + bit;
                    // SAFETY: i + 1 <= offsets.len() - 1.
                    let start: usize = unsafe { offsets.get_unchecked(i) }.as_();
                    let end: usize = unsafe { offsets.get_unchecked(i + 1) }.as_();
                    // SAFETY: s..e is valid in all_bytes.
                    let codes = unsafe { all_bytes.get_unchecked(start..end) };
                    if self.matches(codes) {
                        packed_match |= 1u64 << bit;
                    }
                    bm &= bm - 1;
                }
                packed_match ^ neg_word
            };
            // SAFETY: out has capacity.
            unsafe { out.push_unchecked(packed) };
        }

        if remainder != 0 {
            let chunk = chunks;
            let cand_byte_off = chunk * 8;
            let mut cand_word: u64 = 0;
            let cand_bytes_left = cand_bytes.len() - cand_byte_off;
            for j in 0..cand_bytes_left.min(8) {
                // SAFETY: chunk*8 + j < cand_bytes.len().
                cand_word |=
                    (unsafe { *cand_bytes.get_unchecked(cand_byte_off + j) } as u64) << (j * 8);
            }
            let mask = if remainder == 64 {
                u64::MAX
            } else {
                (1u64 << remainder) - 1
            };
            let packed = if cand_word & mask == 0 {
                neg_word & mask
            } else {
                let mut packed_match: u64 = 0;
                let mut bm = cand_word & mask;
                while bm != 0 {
                    let bit = bm.trailing_zeros() as usize;
                    let i = chunk * 64 + bit;
                    // SAFETY: i + 1 <= offsets.len() - 1.
                    let start: usize = unsafe { offsets.get_unchecked(i) }.as_();
                    let end: usize = unsafe { offsets.get_unchecked(i + 1) }.as_();
                    // SAFETY: s..e is valid in all_bytes.
                    let codes = unsafe { all_bytes.get_unchecked(start..end) };
                    if self.matches(codes) {
                        packed_match |= 1u64 << bit;
                    }
                    bm &= bm - 1;
                }
                (packed_match ^ neg_word) & mask
            };
            // SAFETY: out has capacity.
            unsafe { out.push_unchecked(packed) };
        }

        BitBuffer::new(out.into_byte_buffer().freeze(), n)
    }

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
