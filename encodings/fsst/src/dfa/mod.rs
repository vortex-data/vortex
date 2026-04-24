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
//! ## The Algorithm: KMP → Byte Table → Symbol Table → Flat DFA
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
//! ### Stage 4: Flat `u8` Table
//!
//! The fused table is stored as a flat `Vec<u8>` indexed as
//! `transitions[state * 256 + byte]`. Both the prefix and contains DFAs use
//! escape-sentinel handling: when the scanner sees the sentinel value, it reads
//! the next byte from a separate byte-level escape table.
//!
//! TODO(joe): for short contains needles (≤7 bytes), a branchless escape-folded
//! DFA with hierarchical 4-byte composition is ~2x faster. For needles ≤127
//! bytes, an escape-folded flat DFA (2N+1 states) avoids the sentinel branch.
//! See commit 7faf9f36f for those implementations.
//!
//! ## State-Space Limits
//!
//! The public behavior is shaped by two implementation limits, both measured in
//! pattern **bytes** rather than Unicode scalar values:
//!
//! - `prefix%` pushdown is limited to **253 bytes**. The flat prefix DFA uses
//!   `u8` state ids and needs room for progress states, an accept state, a
//!   fail state, and one escape sentinel (N+3 ≤ 256).
//! - `%needle%` pushdown is limited to **254 bytes**. The contains DFA stores
//!   states in `u8`, so it needs room for every match-progress state plus both
//!   the accept state and the escape sentinel.
//!
//! Patterns beyond those limits are still valid LIKE patterns; they simply do
//! not use FSST pushdown and must be evaluated through the fallback path.

mod flat_contains;
mod prefix;
#[cfg(test)]
mod tests;

use flat_contains::FlatContainsDfa;
use fsst::ESCAPE_CODE;
use fsst::Symbol;
use prefix::FlatPrefixDfa;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
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
    Prefix(FlatPrefixDfa),
    Contains(FlatContainsDfa),
}

impl FsstMatcher {
    /// Try to build a matcher for the given LIKE pattern.
    ///
    /// Returns `Ok(None)` if the pattern shape is not supported for pushdown
    /// (e.g. `_` wildcards, multiple non-bookend `%`, `prefix%` longer than
    /// 253 bytes, or `%needle%` longer than 254 bytes).
    pub(crate) fn try_new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        pattern: &[u8],
    ) -> VortexResult<Option<Self>> {
        let Some(like_kind) = LikeKind::parse(pattern) else {
            return Ok(None);
        };

        let inner = match like_kind {
            LikeKind::Prefix(b"") | LikeKind::Contains(b"") => MatcherInner::MatchAll,
            LikeKind::Prefix(prefix) => {
                if prefix.len() > FlatPrefixDfa::MAX_PREFIX_LEN {
                    return Ok(None);
                }
                MatcherInner::Prefix(FlatPrefixDfa::new(symbols, symbol_lengths, prefix)?)
            }
            LikeKind::Contains(needle) => {
                if needle.len() > FlatContainsDfa::MAX_NEEDLE_LEN {
                    return Ok(None);
                }
                MatcherInner::Contains(FlatContainsDfa::new(symbols, symbol_lengths, needle)?)
            }
        };

        Ok(Some(Self { inner }))
    }

    /// Run the matcher on a single FSST-compressed code sequence.
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.inner {
            MatcherInner::MatchAll => true,
            MatcherInner::Prefix(dfa) => dfa.matches(codes),
            MatcherInner::Contains(dfa) => dfa.matches(codes),
        }
    }
}

/// The subset of LIKE patterns we can handle without decompression.
enum LikeKind<'a> {
    /// `prefix%`
    Prefix(&'a [u8]),
    /// `%needle%`
    Contains(&'a [u8]),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a [u8]) -> Option<Self> {
        // `prefix%` (including just `%` where prefix is empty)
        if let Some(prefix) = pattern.strip_suffix(b"%")
            && !prefix.contains(&b'%')
            && !prefix.contains(&b'_')
        {
            return Some(LikeKind::Prefix(prefix));
        }

        // `%needle%`
        let inner = pattern.strip_prefix(b"%")?.strip_suffix(b"%")?;
        if !inner.contains(&b'%') && !inner.contains(&b'_') {
            return Some(LikeKind::Contains(inner));
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Scan helper
// ---------------------------------------------------------------------------

// TODO: add N-way ILP overrun scan for higher throughput on short strings.
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
    let n_symbols = symbols.len();
    let mut sym_trans = vec![0u8; n_states as usize * n_symbols];
    for state in 0..n_states {
        for code in 0..n_symbols {
            if state == accept_state {
                sym_trans[state as usize * n_symbols + code] = accept_state;
                continue;
            }
            let sym = symbols[code].to_u64().to_le_bytes();
            let sym_len = usize::from(symbol_lengths[code]);
            let mut s = state;
            for &b in &sym[..sym_len] {
                if s == accept_state {
                    break;
                }
                s = byte_table[s as usize * 256 + b as usize];
            }
            sym_trans[state as usize * n_symbols + code] = s;
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

// ---------------------------------------------------------------------------
// KMP helpers
// ---------------------------------------------------------------------------

fn kmp_byte_transitions(needle: &[u8]) -> Vec<u8> {
    let n_states = u8::try_from(needle.len() + 1)
        .vortex_expect("kmp_byte_transitions: must have needle.len() ≤ 255");
    let accept = n_states - 1;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u8; n_states as usize * 256];
    for state in 0..n_states {
        for byte in 0..256usize {
            if state == accept {
                table[state as usize * 256 + byte] = accept;
                continue;
            }
            let mut s = state;
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
            table[state as usize * 256 + byte] = s;
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
