// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # FSST LIKE Pushdown via DFA Construction
//!
//! This module implements DFA-based pattern matching directly on FSST-compressed
//! strings, without decompressing them. It handles two pattern shapes:
//!
//! - **Prefix**: `'prefix%'`  — matches strings starting with a literal prefix.
//! - **Contains**: `'%needle%'` — matches strings containing a literal substring.
//!
//! Pushdown is intentionally conservative. If the pattern shape is unsupported,
//! or if the pattern exceeds the DFA's representable state space, construction
//! returns `None` and the caller must fall back to ordinary decompression-based
//! LIKE evaluation.
//!
//! TODO(joe): suffix (`'%suffix'`) pushdown. Two approaches:
//! - **Forward DFA**: use a non-sticky accept state with KMP fallback transitions,
//!   check `state == accept` after processing all codes. Branchless and vectorizable.
//! - **Backward scan**: walk the compressed code stream in reverse, comparing symbol
//!   bytes from the end. Simpler, no DFA construction, but requires reverse parsing
//!   of the FSST escape mechanism.
//!
//! ## Background: FSST Encoding
//!
//! [FSST](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf) compresses strings by
//! replacing frequent byte sequences with single-byte **symbol codes** (0–254). Code
//! byte 255 is reserved as the **escape code**: the next byte is a literal (uncompressed)
//! byte. So a compressed string is a stream of:
//!
//! ```text
//! [symbol_code] ... [symbol_code] [ESCAPE literal_byte] [symbol_code] ...
//! ```
//!
//! A single symbol can expand to 1–8 bytes. Matching on compressed codes requires
//! the DFA to handle multi-byte symbol expansions and the escape mechanism.
//!
//! ## The Algorithm: KMP → Byte Table → Symbol Table → Packed DFA
//!
//! Construction proceeds through four stages:
//!
//! ### Stage 1: KMP Failure Function
//!
//! We compute the standard [KMP](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)
//! failure function for the needle bytes. This tells us, on a mismatch at
//! position `i`, the longest proper prefix of `needle[0..i]` that is also a
//! suffix — i.e., where to resume matching instead of starting over.
//!
//! ```text
//! Needle: "abcabd"
//! Failure: [0, 0, 0, 1, 2, 0]
//!                      ^  ^
//!                      At position 3 ('a'), the prefix "a" matches suffix "a"
//!                      At position 4 ('b'), the prefix "ab" matches suffix "ab"
//! ```
//!
//! ### Stage 2: Byte-Level Transition Table
//!
//! From the failure function, we build a full `(state × byte) → state` transition
//! table. State `i` means "we have matched `needle[0..i]`". State `n` (= needle
//! length) is the **accept** state.
//!
//! ```text
//! Needle: "aba"  (3 states + accept)
//!
//!         Input byte
//! State   'a'    'b'    other
//! ─────   ────   ────   ─────
//!   0      1      0      0      ← looking for first 'a'
//!   1      1      2      0      ← matched "a", want 'b'
//!   2      3✓     0      0      ← matched "ab", want 'a'
//!   3✓     3✓     3✓     3✓     ← accept (sticky)
//! ```
//!
//! For prefix matching, a mismatch at any state goes to a **fail** state (no
//! fallback). For contains matching, mismatches follow KMP fallback transitions
//! so we can find the needle anywhere in the string.
//!
//! ### Stage 3: Symbol-Level Transition Table
//!
//! FSST symbols can be multi-byte. To compute the transition for symbol code `c`
//! in state `s`, we simulate feeding each byte of the symbol through the byte
//! table:
//!
//! ```text
//! Symbol #42 = "the" (3 bytes)
//! State 0 + 't' → 0, + 'h' → 0, + 'e' → 0  ⟹ sym_trans[0][42] = 0
//!
//! If needle = "them":
//! State 0 + 't' → 1, + 'h' → 2, + 'e' → 3  ⟹ sym_trans[0][42] = 3
//! ```
//!
//! We then build a **fused 256-wide table**: for code bytes 0–254, use the
//! symbol transition; for code byte 255 (ESCAPE_CODE), transition to a
//! special sentinel that tells the scanner to read the next literal byte.
//!
//! ### Stage 4: Packing into the Final Representation
//!
//! The fused table can be stored in different layouts depending on the number
//! of states:
//!
//! - **Shift-packed `u64`** (≤16 states): Each state needs 4 bits. All state
//!   transitions for one input byte fit in a single `u64`. Lookup:
//!   `next = (table[byte] >> (state * 4)) & 0xF`. One cache line per lookup.
//!
//! - **Flat `u8` table** (≤255 states): `transitions[state * 256 + byte]`.
//!   Larger, but still bounded by the `u8` state representation.
//!
//! ## State-Space Limits
//!
//! The public behavior is shaped by two implementation limits, both measured in
//! pattern **bytes** rather than Unicode scalar values:
//!
//! - `prefix%` pushdown is limited to **13 bytes**. The packed prefix DFA uses
//!   4-bit state ids and needs room for normal prefix-progress states, an
//!   accept state, a fail state, and one escape sentinel for FSST literals.
//! - `%needle%` pushdown is limited to **254 bytes**. The long-needle DFA stores
//!   states in `u8`, so it needs room for every match-progress state plus both
//!   the accept state and the escape sentinel.
//!
//! Patterns beyond those limits are still valid LIKE patterns; they simply do
//! not use FSST pushdown and must be evaluated through the fallback path.
//!
//! ## DFA Variants and When Each Is Used
//!
//! ```text
//! ┌───────────────┬──────────────────────────────────────────────────────┐
//! │ Pattern       │ Needle length → DFA variant                        │
//! ├───────────────┼──────────────────────────────────────────────────────┤
//! │ prefix%       │ 0–13 → FsstPrefixDfa (shift-packed, no KMP)        │
//! ├───────────────┼──────────────────────────────────────────────────────┤
//! │ %needle%      │ 1–7     → BranchlessShiftDfa (hierarchical 4-byte) │
//! │               │ 8–127   → FlatContainsDfa (flat u8, esc-folded)   │
//! │               │ 128–254 → FlatContainsDfa (flat u8, esc-sentinel) │
//! └───────────────┴──────────────────────────────────────────────────────┘
//! ```
//!
//! ## Escape Handling Strategies
//!
//! There are two ways to handle the FSST escape code in the DFA:
//!
//! **Escape sentinel** (used by `FlatContainsDfa` for long needles, `FsstPrefixDfa`):
//! The escape code maps to a sentinel state. The scanner checks for it and
//! reads the next byte from a separate escape transition table.
//!
//! ```text
//! loop:
//!   state = transitions[byte]       // might be sentinel
//!   if state == SENTINEL:
//!     state = escape_transitions[next_byte]  // branch
//! ```
//!
//! **Escape folding** (used by `BranchlessShiftDfa`, `FlatContainsDfa` for short needles):
//! Escape states are folded into the state space. State `s+N+1` means "was in
//! state `s`, just consumed ESCAPE_CODE". The next byte's transition from an
//! escape state uses the byte-level table. No branch needed in the scanner.
//!
//! ```text
//! States: [0..N-1: normal] [N: accept] [N+1..2N: escape shadows]
//! Total: 2N+1 states. With 4-bit packing, max N=7.
//!
//! loop:
//!   state = transitions[state][byte]   // branchless!
//! ```

mod branchless_shift;
mod flat_contains;
mod prefix;
#[cfg(test)]
mod tests;

use branchless_shift::BranchlessShiftDfa;
use flat_contains::FlatContainsDfa;
use fsst::ESCAPE_CODE;
use fsst::Symbol;
use prefix::FsstPrefixDfa;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

// ---------------------------------------------------------------------------
// FsstMatcher — unified public API
// ---------------------------------------------------------------------------

/// A compiled matcher for LIKE patterns on FSST-compressed strings.
///
/// Encapsulates pattern parsing and DFA variant selection. Returns `None` from
/// [`try_new`](Self::try_new) for patterns that cannot be evaluated without
/// decompression (e.g., `_` wildcards, multiple `%` in non-standard positions,
/// or patterns that exceed the DFA's representable byte-length limits).
pub(crate) struct FsstMatcher {
    inner: MatcherInner,
}

enum MatcherInner {
    MatchAll,
    Prefix(Box<FsstPrefixDfa>),
    ContainsBranchless(Box<BranchlessShiftDfa>),
    ContainsFlat(FlatContainsDfa),
}

impl FsstMatcher {
    /// Try to build a matcher for the given LIKE pattern.
    ///
    /// Returns `Ok(None)` if the pattern shape is not supported for pushdown
    /// (e.g. `_` wildcards, multiple non-bookend `%`, `prefix%` longer than
    /// 13 bytes, or `%needle%` longer than 254 bytes).
    pub(crate) fn try_new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        pattern: &str,
    ) -> VortexResult<Option<Self>> {
        let Some(like_kind) = LikeKind::parse(pattern) else {
            return Ok(None);
        };

        let inner = match like_kind {
            LikeKind::Prefix("") => MatcherInner::MatchAll,
            LikeKind::Prefix(prefix) => {
                let prefix = prefix.as_bytes();
                if prefix.len() > FsstPrefixDfa::MAX_PREFIX_LEN {
                    return Ok(None);
                }
                MatcherInner::Prefix(Box::new(FsstPrefixDfa::new(
                    symbols,
                    symbol_lengths,
                    prefix,
                )?))
            }
            LikeKind::Contains(needle) => {
                let needle = needle.as_bytes();
                if needle.len() > FlatContainsDfa::MAX_NEEDLE_LEN {
                    return Ok(None);
                }
                if needle.len() <= BranchlessShiftDfa::MAX_NEEDLE_LEN {
                    MatcherInner::ContainsBranchless(Box::new(BranchlessShiftDfa::new(
                        symbols,
                        symbol_lengths,
                        needle,
                    )?))
                } else {
                    MatcherInner::ContainsFlat(FlatContainsDfa::new(
                        symbols,
                        symbol_lengths,
                        needle,
                    )?)
                }
            }
        };

        Ok(Some(Self { inner }))
    }

    /// Run the matcher on a single FSST-compressed code sequence.
    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.inner {
            MatcherInner::MatchAll => true,
            MatcherInner::Prefix(dfa) => dfa.matches(codes),
            MatcherInner::ContainsBranchless(dfa) => dfa.matches(codes),
            MatcherInner::ContainsFlat(dfa) => dfa.matches(codes),
        }
    }
}

/// The subset of LIKE patterns we can handle without decompression.
enum LikeKind<'a> {
    /// `prefix%`
    Prefix(&'a str),
    /// `%needle%`
    Contains(&'a str),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a str) -> Option<Self> {
        // `prefix%` (including just `%` where prefix is empty)
        if let Some(prefix) = pattern.strip_suffix('%')
            && !prefix.contains(['%', '_'])
        {
            return Some(LikeKind::Prefix(prefix));
        }

        // `%needle%`
        let inner = pattern.strip_prefix('%')?.strip_suffix('%')?;
        if !inner.contains(['%', '_']) {
            return Some(LikeKind::Contains(inner));
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Scan helper
// ---------------------------------------------------------------------------

// TODO: add N-way ILP overrun scan for higher throughput on short strings.
#[inline]
pub(crate) fn dfa_scan_to_bitbuf<T, F>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    matcher: F,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    F: Fn(&[u8]) -> bool,
{
    let mut start: usize = offsets[0].as_();
    BitBuffer::collect_bool(n, |i| {
        let end: usize = offsets[i + 1].as_();
        let result = matcher(&all_bytes[start..end]) != negated;
        start = end;
        result
    })
}

// ---------------------------------------------------------------------------
// Shared helpers — used by multiple DFA implementations
// ---------------------------------------------------------------------------

/// Extract a state id from a shift-packed `u64` word.
///
/// Each state occupies `bits` bits. The mask `(1 << bits) - 1` guarantees the
/// result is at most 15 (for `bits = 4`), which always fits in `u8`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "masked to `bits` bits (≤4), result ≤ 15"
)]
#[inline(always)]
fn shift_extract(packed: u64, state: u8, bits: u32) -> u8 {
    let mask = (1u64 << bits) - 1;
    ((packed >> (u32::from(state) * bits)) & mask) as u8
}

/// Compose two shift-packed transition `u64`s: for each state, apply `first`
/// then `second`, packing the result back into a single `u64`.
fn compose_packed(first: u64, second: u64, total_states: u8, bits: u32) -> u64 {
    let mut packed = 0u64;
    for state in 0..total_states {
        let mid = shift_extract(first, state, bits);
        let final_s = shift_extract(second, mid, bits);
        packed |= u64::from(final_s) << (u32::from(state) * bits);
    }
    packed
}

// ---------------------------------------------------------------------------
// DFA construction helpers
// ---------------------------------------------------------------------------

/// Builds the per-symbol transition table for FSST symbols.
///
/// For each `(state, symbol_code)` pair, simulates feeding the symbol's bytes
/// through the byte-level transition table to compute the resulting state.
///
/// Returns a flat `Vec<u8>` indexed as `[state * n_symbols + code]`.
fn build_symbol_transitions(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    byte_table: &[u8],
    n_states: u8,
    accept_state: u8,
) -> Vec<u8> {
    let n_states = usize::from(n_states);
    let n_symbols = symbols.len();
    let mut sym_trans = vec![0u8; n_states * n_symbols];
    for state in 0..n_states {
        for code in 0..n_symbols {
            if state == usize::from(accept_state) {
                sym_trans[state * n_symbols + code] = accept_state;
                continue;
            }
            let sym = symbols[code].to_u64().to_le_bytes();
            let sym_len = usize::from(symbol_lengths[code]);
            let mut s = state;
            for &b in &sym[..sym_len] {
                if s == usize::from(accept_state) {
                    break;
                }
                s = usize::from(byte_table[s * 256 + usize::from(b)]);
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "s is a state id < n_states ≤ 256"
            )]
            {
                sym_trans[state * n_symbols + code] = s as u8;
            }
        }
    }
    sym_trans
}

/// Builds a fused 256-wide transition table from symbol transitions.
///
/// For each `(state, code_byte)`:
/// - Code bytes `0..n_symbols`: use the symbol transition
/// - `ESCAPE_CODE`: maps to `escape_value` (either a sentinel or escape state)
/// - All others: use `default` (typically 0 for contains, fail_state for prefix)
///
/// Returns a flat `Vec<u8>` indexed as `[state * 256 + code_byte]`.
fn build_fused_table(
    sym_trans: &[u8],
    n_symbols: usize,
    n_states: u8,
    escape_value_fn: impl Fn(u8) -> u8,
    default: u8,
) -> Vec<u8> {
    let mut fused = vec![default; usize::from(n_states) * 256];
    for state in 0..n_states {
        let s = usize::from(state);
        for code in 0..n_symbols {
            fused[s * 256 + code] = sym_trans[s * n_symbols + code];
        }
        fused[s * 256 + usize::from(ESCAPE_CODE)] = escape_value_fn(state);
    }
    fused
}

/// Packs a fused table into shift-encoded `u64` arrays.
///
/// Each `u64` encodes transitions for ALL states for one input byte.
/// Lookup: `next = (table[byte] >> (state * BITS)) & MASK`.
fn pack_shift_table(fused: &[u8], n_states: u8, bits: u32) -> [u64; 256] {
    let mut packed = [0u64; 256];
    for code_byte in 0..256usize {
        let mut val = 0u64;
        for state in 0..n_states {
            val |=
                u64::from(fused[usize::from(state) * 256 + code_byte]) << (u32::from(state) * bits);
        }
        packed[code_byte] = val;
    }
    packed
}

/// Builds an escape-folded fused transition table for contains matching.
///
/// State layout: `[0..n-1]` match progress, `[n]` accept (sticky), `[n+1..2n]` escape shadows.
/// Total states: `2 * needle.len() + 1`.
///
/// For normal states, the escape code maps to the corresponding escape shadow state.
/// Escape shadow states use byte-level KMP transitions so the next literal byte
/// resumes matching correctly — no branch needed in the scanner.
fn build_escape_folded_table(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Vec<u8> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "needle.len() ≤ FlatContainsDfa::MAX_FOLDED_LEN (127)"
    )]
    let n = needle.len() as u8;
    let accept_state = n;
    let total_states = usize::from(2 * n + 1);

    let byte_table = kmp_byte_transitions(needle);
    let sym_trans =
        build_symbol_transitions(symbols, symbol_lengths, &byte_table, n + 1, accept_state);

    let n_symbols = symbols.len();
    let n_usize = usize::from(n);
    let mut fused = vec![0u8; total_states * 256];
    for code_byte in 0..256usize {
        // Normal states 0..n
        for s in 0..n_usize {
            if code_byte == usize::from(ESCAPE_CODE) {
                #[expect(clippy::cast_possible_truncation, reason = "s + n + 1 ≤ 2*127+1 = 255")]
                {
                    fused[s * 256 + code_byte] = (s + n_usize + 1) as u8;
                }
            } else if code_byte < n_symbols {
                fused[s * 256 + code_byte] = sym_trans[s * n_symbols + code_byte];
            }
        }
        // Accept state (sticky)
        fused[n_usize * 256 + code_byte] = accept_state;
        // Escape shadow states n+1..2n
        for s in 0..n_usize {
            let esc_state = s + n_usize + 1;
            fused[esc_state * 256 + code_byte] = byte_table[s * 256 + code_byte];
        }
    }
    fused
}

// ---------------------------------------------------------------------------
// KMP helpers
// ---------------------------------------------------------------------------

fn kmp_byte_transitions(needle: &[u8]) -> Vec<u8> {
    let n_states = needle.len() + 1;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "needle.len() ≤ 254, accept state fits in u8"
    )]
    let accept = needle.len() as u8;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u8; n_states * 256];
    for state in 0..n_states {
        for byte in 0..256usize {
            if state == needle.len() {
                table[state * 256 + byte] = accept;
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "state < needle.len() ≤ 254"
            )]
            let mut s = state as u8;
            loop {
                if byte == usize::from(needle[usize::from(s)]) {
                    s += 1;
                    break;
                }
                if s == 0 {
                    break;
                }
                s = failure[usize::from(s) - 1];
            }
            table[state * 256 + byte] = s;
        }
    }
    table
}

fn kmp_failure_table(needle: &[u8]) -> Vec<u8> {
    let mut failure = vec![0u8; needle.len()];
    let mut k = 0u8;
    for i in 1..needle.len() {
        while k > 0 && needle[usize::from(k)] != needle[i] {
            k = failure[usize::from(k) - 1];
        }
        if needle[usize::from(k)] == needle[i] {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}
