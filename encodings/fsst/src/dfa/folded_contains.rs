// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Escape-folded flat `u8` transition table DFA for contains matching
//! (`LIKE '%needle%'`).
//!
//! ## Why escape-fold?
//!
//! The plain [`super::flat_contains::FlatContainsDfa`] keeps a sentinel branch
//! in its inner loop: when the current code is `ESCAPE_CODE`, the table maps to
//! a sentinel value, the scanner detects it, and a second table lookup (in a
//! separate byte table) consumes the following literal byte. That's a hard-to-
//! predict branch on every code byte.
//!
//! The escape-folded DFA encodes "we just saw an `ESCAPE_CODE`, expecting a
//! literal byte" directly into the state space. With needle length `N`, where
//! `N <= 127`:
//!
//! - **Normal states** `0..=N`: regular KMP-style match progress; `N` is the
//!   accept state (sticky).
//! - **Escape states** `N+1..=2N`: "in-escape from base normal state
//!   `s = state - (N + 1)`" for `s` in `0..=N-1`. A read here is interpreted
//!   as a literal byte, advancing per the byte-level transition table for `s`.
//!
//! Total states: `2N + 1 <= 255`, so the state id fits in `u8`.
//!
//! The transition table is a flat `Vec<u8>` of size `(2N + 1) * 256`. For
//! normal states, the entry on `ESCAPE_CODE` goes to the matching escape
//! state `s + N + 1`. For escape states, all 256 entries are read as literal
//! bytes and dispatched through the byte table for the base state. There is
//! no sentinel branch in the inner loop -- every code byte produces exactly
//! one table lookup.
//!
//! The state-0 skip strategy (`memchr` / bitmap) still applies in the same way
//! as the plain DFA: when in state 0 we jump to the next code that could
//! progress the match.

use fsst::Symbol;
use vortex_array::dtype::IntegerPType;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::ESCAPE_CODE;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;
use super::skip::SkipStrategy;

/// Escape-folded flat `u8` transition table DFA for contains matching.
///
/// Supports needles up to [`Self::MAX_NEEDLE_LEN`] bytes (so the state count
/// `2N + 1` fits in `u8`).
pub(crate) struct FoldedContainsDfa {
    /// `transitions[state * 256 + byte]` -> next state.
    ///
    /// Layout: rows `0..=N` are normal states (regular byte/symbol dispatch);
    /// rows `N+1..=2N` are escape states whose 256 entries are literal-byte
    /// dispatches via the underlying byte table.
    transitions: Vec<u8>,
    accept_state: u8,
    /// State-0 skip strategy.
    skip: SkipStrategy,
    /// Rare-byte anchor strategy: a SkipStrategy whose progressing-code set is
    /// the FSST codes whose expansion contains the rarest needle byte (plus
    /// `ESCAPE_CODE`). This is a NECESSARY-condition prefilter: any matching
    /// string must contain at least one anchor code. Selected only when the
    /// resulting set is ≤ 3 codes (i.e., a `Memchr1/2/3` strategy), which lets
    /// us run a SIMD `memchrN` over the entire `all_bytes` buffer for the
    /// global scan.
    rare_byte_anchor: Option<SkipStrategy>,
}

impl FoldedContainsDfa {
    /// Maximum needle length: `2N + 1 <= 255` so `N <= 127`.
    pub(crate) const MAX_NEEDLE_LEN: usize = 127;

    /// Build a folded contains DFA for `needle`.
    ///
    /// Returns `Err` if `needle.len() > `[`Self::MAX_NEEDLE_LEN`].
    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        if needle.len() > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for folded contains DFA",
                needle.len(),
                Self::MAX_NEEDLE_LEN
            );
        }
        // Empty needles are handled at a higher level (MatchAll), but we still
        // accept them here defensively (N=0 -> only the accept state).
        let accept_state =
            u8::try_from(needle.len()).vortex_expect("FoldedContainsDfa: accept state fits in u8");
        let n_normal = accept_state + 1; // states 0..=N
        // Total states: 2N+1 (normal 0..=N, escape N+1..=2N for base 0..=N-1).
        let n_states_usize = 2 * usize::from(accept_state) + 1;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_normal, accept_state);

        // Build the folded fused table: (2N+1) * 256.
        let n_symbols = symbols.len();
        let mut transitions = vec![0u8; n_states_usize * 256];

        // Rows 0..=N: normal states.
        for s in 0..n_normal {
            let row = usize::from(s) * 256;
            // Symbol codes 0..n_symbols.
            for code in 0..n_symbols {
                transitions[row + code] = sym_trans[usize::from(s) * n_symbols + code];
            }
            // ESCAPE_CODE: go to the matching escape state, except for accept
            // (which is sticky -- all transitions remain at accept).
            let escape_target = if s == accept_state {
                accept_state
            } else {
                // Escape state for base s = N + 1 + s.
                accept_state + 1 + s
            };
            transitions[row + usize::from(ESCAPE_CODE)] = escape_target;
            // Other code bytes (n_symbols..255 except ESCAPE_CODE) default to 0,
            // matching the plain `FlatContainsDfa` semantics.
        }

        // Rows N+1..=2N: escape states. For escape state e = N + 1 + s where
        // s in 0..=N-1, all 256 entries dispatch the next byte as a literal
        // through `byte_table[s * 256 + b]`.
        for s in 0..accept_state {
            let escape_state = accept_state + 1 + s;
            let row = usize::from(escape_state) * 256;
            let byte_row = usize::from(s) * 256;
            transitions[row..row + 256].copy_from_slice(&byte_table[byte_row..byte_row + 256]);
        }

        // Build the skip strategy from row 0 of the transitions (the first 256
        // entries). State 0 is reached either initially or by KMP fallback,
        // and we want to skip codes that leave us at 0.
        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        // Compute a rare-byte anchor: pick the needle byte whose contributing
        // FSST code set is smallest, and build a Memchr1/2/3 over those codes
        // (plus ESCAPE_CODE). This is sound because the literal byte must
        // appear in the decompressed string for any match, and that byte must
        // come from either a code whose expansion contains it, or from an
        // ESCAPE_CODE+literal pair. We only use it when the set is ≤ 3 codes.
        let rare_byte_anchor = compute_rare_byte_anchor(symbols, symbol_lengths, needle);

        Ok(Self {
            transitions,
            accept_state,
            skip,
            rare_byte_anchor,
        })
    }

    /// Scan `n` FSST-compressed strings to a bit-packed boolean output.
    ///
    /// Two-tier scan:
    /// 1. If a SIMD-friendly anchor prefilter is available (the rare-byte
    ///    anchor or the state-0 skip strategy fits in `Memchr1/2/3`), run a
    ///    single global SIMD `memchrN` pass over `all_bytes` to mark
    ///    candidate strings. Strings without any anchor code skip the
    ///    per-string DFA call entirely.
    /// 2. Otherwise (state-0 needs a `Bitmap` — i.e., 4+ progressing codes),
    ///    skip the global pre-scan: it would not be faster than the
    ///    per-string Bitmap that the DFA's `matches` already does. Pack the
    ///    per-string DFA results directly into 64-string blocks.
    ///
    /// In both cases, the output is bit-identical to running
    /// [`Self::matches`] on each string and packing into a `BitBuffer`. The
    /// 64-block manual packing avoids the closure-call overhead of
    /// `BitBuffer::collect_bool`.
    pub(crate) fn scan_to_bitbuf<T: IntegerPType>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer {
        // Pick the best SIMD-friendly anchor prefilter, if any. Prefer the
        // rare-byte anchor (it's a NECESSARY-condition prefilter, more
        // selective than the state-0 progressing set). Fall back to the
        // state-0 skip strategy if it's already a `Memchr1/2/3`.
        let state0_anchor = match &self.skip {
            SkipStrategy::Memchr1(_)
            | SkipStrategy::Memchr2(_, _)
            | SkipStrategy::Memchr3(_, _, _) => Some(&self.skip),
            SkipStrategy::Bitmap(_) => None,
        };
        let scan_skip: Option<&SkipStrategy> = self.rare_byte_anchor.as_ref().or(state0_anchor);

        if let Some(skip) = scan_skip {
            self.scan_with_anchor(n, offsets, all_bytes, negated, skip)
        } else {
            self.scan_no_anchor(n, offsets, all_bytes, negated)
        }
    }

    /// Tight 64-string-block packed scan with NO anchor prefilter.
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
                // SAFETY: s..e is valid in all_bytes by construction of FSST codes.
                let codes = unsafe { all_bytes.get_unchecked(start..end) };
                if self.matches(codes) {
                    packed_match |= 1u64 << bit;
                }
            }
            let packed = packed_match ^ neg_word;
            // SAFETY: out has capacity for n.div_ceil(64) words.
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

    /// Tight 64-string-block packed scan WITH a global anchor pre-scan.
    /// The anchor scan uses a single SIMD `memchrN` pass over the entire
    /// `all_bytes` buffer to mark candidate strings; non-candidates skip the
    /// per-string DFA call.
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
            // SAFETY: chunk * 8 + 8 <= candidates.len().div_ceil(8).
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
            // No candidates in this 64-string block -> output is `neg_word`.
            let packed = if cand_word == 0 {
                neg_word
            } else {
                let mut packed_match: u64 = 0;
                let mut bm = cand_word;
                while bm != 0 {
                    let bit = bm.trailing_zeros() as usize;
                    let i = chunk * 64 + bit;
                    // SAFETY: i < n implies i + 1 <= offsets.len() - 1.
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

    /// Run the matcher over `codes`. Returns `true` iff the needle appears.
    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let transitions = self.transitions.as_slice();
        let accept = self.accept_state;
        let mut pos: usize = 0;
        let len = codes.len();

        // Outer loop: SIMD-skip in state 0 to the next progressing code, then
        // run a tight inner loop while state != 0. The inner loop is uniform:
        // one table lookup per code byte, no sentinel branch. We only return
        // to the outer loop when the DFA falls back to state 0 (KMP failure).
        loop {
            match self.skip.find_next_progressing(codes, pos) {
                Some(next) => pos = next,
                None => return false,
            }

            // We're at a progressing code: step once.
            let code = codes[pos];
            pos += 1;
            let mut state = transitions[usize::from(code)];
            if state == accept {
                return true;
            }

            // Inner loop while state != 0.
            while state != 0 && pos < len {
                let c = codes[pos];
                pos += 1;
                state = transitions[usize::from(state) * 256 + usize::from(c)];
                if state == accept {
                    return true;
                }
            }
            if pos >= len {
                return false;
            }
        }
    }
}

/// Compute a "rare-byte anchor" `SkipStrategy` for the contains DFA.
///
/// For each unique byte `b` in `needle`, compute the set of FSST code bytes
/// whose symbol expansion contains `b`. Pick the byte with the smallest set;
/// add `ESCAPE_CODE` (so escape-coded literal bytes are included).
///
/// Returns `Some(SkipStrategy)` only if the resulting set fits in `Memchr1`,
/// `Memchr2`, or `Memchr3` (≤ 3 codes), so the global scan benefits from
/// memchr SIMD. For larger sets, the state-0 progressing skip strategy is
/// usually a better (more selective) prefilter.
fn compute_rare_byte_anchor(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    needle: &[u8],
) -> Option<SkipStrategy> {
    if needle.is_empty() {
        return None;
    }

    let n_symbols = symbols.len();
    let mut best_codes: Option<Vec<u8>> = None;

    let mut seen_byte = [false; 256];
    for &nb in needle {
        if seen_byte[usize::from(nb)] {
            continue;
        }
        seen_byte[usize::from(nb)] = true;

        // Codes whose expansion contains `nb`.
        let mut codes: Vec<u8> = Vec::new();
        for code in 0..n_symbols {
            let bytes = symbols[code].to_u64().to_le_bytes();
            let len = usize::from(symbol_lengths[code]);
            if len == 0 || len > 8 {
                continue;
            }
            if bytes[..len].contains(&nb)
                && let Ok(c) = u8::try_from(code)
            {
                codes.push(c);
            }
        }
        // Always include ESCAPE_CODE: the literal byte after an ESCAPE_CODE
        // could be `nb`, contributing to a match without the byte appearing
        // in any compressed symbol.
        if !codes.contains(&ESCAPE_CODE) {
            codes.push(ESCAPE_CODE);
        }

        // Stop early if this byte yields the smallest possible set (≤ 1).
        if codes.len() <= 1 {
            best_codes = Some(codes);
            break;
        }
        match &best_codes {
            None => best_codes = Some(codes),
            Some(prev) if codes.len() < prev.len() => best_codes = Some(codes),
            _ => {}
        }
    }

    let codes = best_codes?;
    match codes.len() {
        0 => None,
        1 => Some(SkipStrategy::Memchr1(codes[0])),
        2 => Some(SkipStrategy::Memchr2(codes[0], codes[1])),
        3 => Some(SkipStrategy::Memchr3(codes[0], codes[1], codes[2])),
        _ => None,
    }
}
