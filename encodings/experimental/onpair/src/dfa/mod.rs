// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # OnPair LIKE Pushdown via DFA Construction
//!
//! This module evaluates `LIKE` patterns directly on OnPair-compressed code
//! streams, without decompressing them. It handles two pattern shapes:
//!
//! - **Prefix**: `'prefix%'`  — matches strings starting with a literal prefix.
//! - **Contains**: `'%needle%'` — matches strings containing a literal substring.
//!
//! Pushdown is conservative. If the pattern shape is unsupported (e.g. `_`
//! wildcards, suffix patterns, or patterns that exceed the DFA's representable
//! state space), [`OnPairMatcher::try_new`] returns `None` and the caller falls
//! back to ordinary decompression-based LIKE evaluation.
//!
//! ## Background: OnPair encoding
//!
//! OnPair compresses strings by replacing byte sequences with single `u16`
//! **token codes**. A code indexes a variable-length (1–16 byte) token in the
//! dictionary: token `c` is `dict_bytes[dict_offsets[c]..dict_offsets[c + 1]]`.
//! A compressed row is just a sequence of codes, and the concatenation of those
//! tokens' bytes is exactly the decompressed row.
//!
//! Unlike FSST, OnPair has **no escape code**: the trainer always emits all 256
//! single-byte tokens, so every byte is representable as a token and every code
//! is a valid dictionary index. That makes the DFA strictly simpler than the
//! FSST one — there is no escape sentinel and no separate escape table.
//!
//! ## The algorithm: byte DFA → per-code transition table → scan
//!
//! 1. Build a byte-level transition table for the needle/prefix (KMP for
//!    contains, a linear table for prefix), exactly as FSST does.
//! 2. Lift it to a **per-code** table: for each `(state, code)` pair, feed the
//!    code's token bytes through the byte table and record the resulting state.
//!    Because a row's token bytes are its decompressed bytes, stepping this
//!    table over a row's codes is equivalent to running the byte DFA over the
//!    decompressed string — so the result is exactly correct regardless of how
//!    the encoder tokenized the row.
//! 3. Scan: walk a row's `u16` codes, one table lookup per code, and test the
//!    accept state.
//!
//! The per-code table has `n_states * n_codes` entries. For the default
//! `dict-12` preset (≤ 4096 tokens) this is at most ~1 MiB and is built once per
//! query.

mod contains;
mod prefix;
#[cfg(test)]
mod tests;

use std::borrow::Cow;

use contains::FlatContainsDfa;
use prefix::FlatPrefixDfa;
use vortex_array::dtype::IntegerPType;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

// ---------------------------------------------------------------------------
// OnPairMatcher — unified public API
// ---------------------------------------------------------------------------

/// A compiled matcher for LIKE patterns on OnPair-compressed code streams.
///
/// Encapsulates pattern parsing and DFA variant selection. Returns `None` from
/// [`try_new`](Self::try_new) for patterns that cannot be evaluated without
/// decompression (e.g. `_` wildcards, suffix patterns, or patterns that exceed
/// the DFA's representable byte-length limits).
pub(crate) struct OnPairMatcher {
    inner: MatcherInner,
}

enum MatcherInner {
    MatchAll,
    Prefix(FlatPrefixDfa),
    Contains(FlatContainsDfa),
}

impl OnPairMatcher {
    /// Try to build a matcher for the given LIKE pattern over the OnPair
    /// dictionary (`dict_bytes` + `dict_offsets`).
    ///
    /// Returns `Ok(None)` if the pattern shape is not supported for pushdown
    /// (e.g. `_` wildcards, non-bookend `%`, `prefix%` longer than 253 bytes, or
    /// `%needle%` longer than 254 bytes).
    pub(crate) fn try_new(
        dict_bytes: &[u8],
        dict_offsets: &[u32],
        pattern: &[u8],
    ) -> VortexResult<Option<Self>> {
        let Some(like_kind) = LikeKind::parse(pattern) else {
            return Ok(None);
        };

        let inner = match like_kind {
            LikeKind::Prefix(pattern) | LikeKind::Contains(pattern) if pattern.is_empty() => {
                MatcherInner::MatchAll
            }
            LikeKind::Prefix(prefix) => {
                if prefix.len() > FlatPrefixDfa::MAX_PREFIX_LEN {
                    return Ok(None);
                }
                MatcherInner::Prefix(FlatPrefixDfa::new(
                    dict_bytes,
                    dict_offsets,
                    prefix.as_ref(),
                ))
            }
            LikeKind::Contains(needle) => {
                if needle.len() > FlatContainsDfa::MAX_NEEDLE_LEN {
                    return Ok(None);
                }
                MatcherInner::Contains(FlatContainsDfa::new(
                    dict_bytes,
                    dict_offsets,
                    needle.as_ref(),
                ))
            }
        };

        Ok(Some(Self { inner }))
    }

    /// Evaluate every row of an OnPair `codes` window into a [`BitBuffer`].
    ///
    /// `offsets` are the per-row boundaries into the *original* `codes` child;
    /// `code_start` is the absolute index the sliced `codes` window begins at,
    /// so `offsets[i] - code_start` indexes `codes`.
    ///
    /// The matcher variant is selected once, outside the row loop, so the
    /// concrete DFA's `matches` inlines into a monomorphic scan rather than
    /// re-dispatching the enum per row.
    pub(crate) fn scan_to_bitbuf<T: IntegerPType>(
        &self,
        n: usize,
        offsets: &[T],
        code_start: usize,
        codes: &[u16],
        negated: bool,
    ) -> BitBuffer {
        match &self.inner {
            MatcherInner::MatchAll => BitBuffer::collect_bool(n, |_| !negated),
            MatcherInner::Prefix(dfa) => {
                scan_rows(n, offsets, code_start, codes, negated, |c| dfa.matches(c))
            }
            MatcherInner::Contains(dfa) => {
                scan_rows(n, offsets, code_start, codes, negated, |c| dfa.matches(c))
            }
        }
    }
}

/// The subset of LIKE patterns we can handle without decompression.
enum LikeKind<'a> {
    /// `prefix%`
    Prefix(Cow<'a, [u8]>),
    /// `%needle%`
    Contains(Cow<'a, [u8]>),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a [u8]) -> Option<Self> {
        Self::parse_prefix(pattern).or_else(|| Self::parse_contains(pattern))
    }

    fn parse_prefix(pattern: &'a [u8]) -> Option<Self> {
        Self::parse_literal_until_final_percent(pattern, 0).map(LikeKind::Prefix)
    }

    fn parse_contains(pattern: &'a [u8]) -> Option<Self> {
        if !pattern.starts_with(b"%") {
            return None;
        }
        Self::parse_literal_until_final_percent(pattern, 1).map(LikeKind::Contains)
    }

    /// Parse `pattern[literal_start..]` as a literal terminated by a single
    /// trailing `%`. Returns `None` if `_` or a non-final `%` is encountered.
    ///
    /// `literal` stays `None` until an escape forces us to materialize bytes;
    /// from then on we push into the owned `Vec`. Otherwise we return a borrowed
    /// slice straight from `pattern`.
    fn parse_literal_until_final_percent(
        pattern: &'a [u8],
        literal_start: usize,
    ) -> Option<Cow<'a, [u8]>> {
        let mut literal: Option<Vec<u8>> = None;
        let mut idx = literal_start;
        while idx < pattern.len() {
            match pattern[idx] {
                b'\\' => {
                    // Trailing `\` is treated as a literal backslash.
                    let escaped = pattern.get(idx + 1).copied().unwrap_or(b'\\');
                    literal
                        .get_or_insert_with(|| pattern[literal_start..idx].to_vec())
                        .push(escaped);
                    idx = (idx + 2).min(pattern.len());
                }
                b'%' if idx + 1 == pattern.len() => {
                    return Some(match literal {
                        Some(buf) => Cow::Owned(buf),
                        None => Cow::Borrowed(&pattern[literal_start..idx]),
                    });
                }
                b'%' | b'_' => return None,
                byte => {
                    // No-op on the borrowed path; only push once we've started copying.
                    if let Some(literal) = &mut literal {
                        literal.push(byte);
                    }
                    idx += 1;
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Scan helper
// ---------------------------------------------------------------------------

/// Walk a `codes` window row by row with a single concrete row matcher,
/// carrying a running start cursor (consecutive rows are contiguous in
/// `codes`) instead of re-reading both boundaries each row.
fn scan_rows<T, F>(
    n: usize,
    offsets: &[T],
    code_start: usize,
    codes: &[u16],
    negated: bool,
    row_matches: F,
) -> BitBuffer
where
    T: IntegerPType,
    F: Fn(&[u16]) -> bool,
{
    let mut start: usize = offsets[0].as_() - code_start;
    BitBuffer::collect_bool(n, |i| {
        let end: usize = offsets[i + 1].as_() - code_start;
        let result = row_matches(&codes[start..end]) != negated;
        start = end;
        result
    })
}

// ---------------------------------------------------------------------------
// Per-code transition table
// ---------------------------------------------------------------------------

/// Number of dictionary tokens (`= dict_offsets.len() - 1`).
fn n_codes(dict_offsets: &[u32]) -> usize {
    dict_offsets.len().saturating_sub(1)
}

/// Lift a byte-level transition table to a per-code table, indexed as
/// `[state * n_codes + code]`.
///
/// ## Only the relevant codes are built
///
/// The needle/prefix can only interact with the (usually tiny) set of dictionary
/// tokens that contain one of its bytes. A token whose bytes are **all** absent
/// from `relevant` drives the byte table from every live state to the same
/// `skip_value` (for contains: a non-needle byte falls all the way back to 0
/// via KMP from any non-accept state; for prefix: it fails). The accept/fail
/// rows are never read — the scan returns the instant it reaches them — so such
/// a token's whole column is just `skip_value`. We pre-fill the table with
/// `skip_value` and only compute columns for codes that contain a relevant byte.
///
/// For a column that *is* built, the token is read once while advancing all
/// `n_states` starting states in lockstep (`cur[s] = byte_table[cur[s]*256+b]`),
/// a per-byte gather over the independent states.
fn build_code_transitions(
    dict_bytes: &[u8],
    dict_offsets: &[u32],
    byte_table: &[u8],
    n_states: usize,
    skip_value: u8,
    relevant: &[bool; 256],
) -> Vec<u8> {
    let n_codes = n_codes(dict_offsets);
    let mut trans = vec![skip_value; n_states * n_codes];
    let n_states_u8 = u8::try_from(n_states).vortex_expect("n_states fits in u8");
    let identity: Vec<u8> = (0..n_states_u8).collect();
    let mut cur = identity.clone();
    for code in 0..n_codes {
        let begin = dict_offsets[code] as usize;
        let end = dict_offsets[code + 1] as usize;
        let token = &dict_bytes[begin..end];
        if !token.iter().any(|&b| relevant[usize::from(b)]) {
            continue; // column is entirely `skip_value` (already filled)
        }
        cur.copy_from_slice(&identity);
        for &b in token {
            let col = usize::from(b);
            for c in &mut cur {
                *c = byte_table[usize::from(*c) * 256 + col];
            }
        }
        for (s, &c) in cur.iter().enumerate() {
            trans[s * n_codes + code] = c;
        }
    }
    trans
}

/// Build a 256-entry presence mask of the bytes that appear in `bytes`.
fn byte_mask(bytes: &[u8]) -> [bool; 256] {
    let mut mask = [false; 256];
    for &b in bytes {
        mask[usize::from(b)] = true;
    }
    mask
}

// ---------------------------------------------------------------------------
// KMP helpers (shared with the contains DFA)
// ---------------------------------------------------------------------------

fn kmp_byte_transitions(needle: &[u8]) -> Vec<u8> {
    let n_states = u8::try_from(needle.len() + 1)
        .vortex_expect("kmp_byte_transitions: must have needle.len() <= 255");
    let accept = n_states - 1;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u8; usize::from(n_states) * 256];
    for state in 0..n_states {
        for byte in 0..256usize {
            if state == accept {
                table[usize::from(state) * 256 + byte] = accept;
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
            table[usize::from(state) * 256 + byte] = s;
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
