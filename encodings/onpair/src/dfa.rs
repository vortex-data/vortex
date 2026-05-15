// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Token-level matchers for `LIKE 'prefix%'` and `LIKE '%needle%'` over
//! OnPair-compressed `codes: &[u16]` — no row decode at all in the hot
//! path (prefix), and a dict-bloom skip + bounded per-row decode for
//! contains.
//!
//! Mirrors `onpair_cpp/include/onpair/search/automata/prefix_automaton.h`
//! and `…/aho_corasick_automaton.h`. The trick that makes both work is
//! the dictionary's lexicographic ordering: the set of dict ids whose
//! tokens start with byte sequence `S` is always a contiguous
//! `[lo, hi)` range — found in O(|S| · log dict) by binary search.
//!
//! ## PrefixAutomaton
//!
//! 1. LPM-tokenise the prefix into `query[0..q]`.
//! 2. For each `i ∈ 0..q`, precompute `intervals[i] = prefix_range(
//!    remaining_prefix_suffix_at_i)` — the dict token range whose bytes
//!    start with the prefix's remaining bytes from position `i` onward.
//! 3. Walk the row's tokens. If token `j` equals `query[j]` advance.
//!    If it differs but is within `intervals[j]` the token must cover
//!    the whole remaining prefix → accept. Otherwise reject. If we run
//!    out of query tokens → accept (rest of row is irrelevant).
//!
//! Per-row cost: at most `q + 1` `u16` comparisons + 1 interval check.
//! For URL-shape data with `q ≈ 5–10` this is ~10 ns / row.
//!
//! ## Contains (dict-bloom + bounded decode)
//!
//! `LIKE '%needle%'` doesn't have a token-level shortcut as clean as
//! prefix because the LPM of "…[bytes]…needle…[bytes]…" tokenises
//! differently depending on the surrounding context. We do:
//!
//! 1. Per-token bloom: precompute `dict_contains[c] = true` iff dict
//!    entry `c` contains `needle` as a byte substring. If any code in
//!    the row has the bit set, the row matches with no decode.
//! 2. Per-token "could be left of a cross-boundary match" bloom:
//!    `dict_could_extend[c] = true` iff some non-empty suffix of dict
//!    entry `c` is a non-empty prefix of `needle`. Rows where no code
//!    has this bit can't match across boundaries either, so we skip
//!    them entirely.
//! 3. Otherwise, decode the row and run `memchr::memmem`.
//!
//! For URL/log shapes the bloom resolves the vast majority of rows
//! without touching `dict_bytes` at all.

use crate::decode::DecodeView;

// ─── prefix_range helper ────────────────────────────────────────────

/// Returns the half-open `[lo, hi)` range of dict ids whose bytes start
/// with `prefix`. The dict is sorted lexicographically (per OnPair
/// `core/dictionary.h`) so the answer is contiguous.
///
/// Empty range if no dict entry starts with `prefix`.
fn prefix_range(dv: &DecodeView<'_>, prefix: &[u8]) -> std::ops::Range<usize> {
    let n = dv.dict_table.len();
    if prefix.is_empty() {
        return 0..n;
    }
    let lo = lower_bound(dv, prefix);
    if lo == n {
        return n..n;
    }
    // Check the actual entry at lo starts with `prefix`; if not, range
    // is empty (lower_bound only guarantees ≥).
    if !dict_starts_with(dv, lo, prefix) {
        return n..n;
    }
    let hi = upper_bound_with_prefix(dv, prefix, lo);
    lo..hi
}

#[inline]
fn dict_token_bytes<'a>(dv: &DecodeView<'a>, id: usize) -> &'a [u8] {
    let entry = dv.dict_table[id];
    let off = (entry >> 16) as usize;
    let len = (entry & 0xffff) as usize;
    &dv.dict_bytes[off..off + len]
}

#[inline]
fn dict_starts_with(dv: &DecodeView<'_>, id: usize, prefix: &[u8]) -> bool {
    let bytes = dict_token_bytes(dv, id);
    bytes.starts_with(prefix)
}

/// First dict id whose bytes are `>= prefix` lexicographically.
fn lower_bound(dv: &DecodeView<'_>, prefix: &[u8]) -> usize {
    let mut lo = 0usize;
    let mut hi = dv.dict_table.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if dict_token_bytes(dv, mid) < prefix {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// First dict id `>= start` whose bytes do **not** start with `prefix`.
fn upper_bound_with_prefix(dv: &DecodeView<'_>, prefix: &[u8], start: usize) -> usize {
    let mut lo = start;
    let mut hi = dv.dict_table.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if dict_starts_with(dv, mid, prefix) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// ─── PrefixAutomaton ────────────────────────────────────────────────

pub(crate) struct PrefixAutomaton {
    query: Vec<u16>,
    /// `intervals[i]` is the dict range whose bytes start with the
    /// prefix's remaining suffix at position `i`. The row's `i`-th token
    /// "covers" the rest of the prefix iff it falls in this range.
    intervals: Vec<std::ops::Range<u32>>,
}

impl PrefixAutomaton {
    /// Build the automaton. Returns `None` if the prefix has a byte
    /// missing from the dict (no row can match) — caller emits an
    /// all-false result.
    pub(crate) fn build(dv: &DecodeView<'_>, prefix: &[u8]) -> Option<Self> {
        if prefix.is_empty() {
            // Empty prefix matches everything — caller short-circuits
            // before calling us.
            return Some(Self {
                query: Vec::new(),
                intervals: Vec::new(),
            });
        }

        let query = crate::lpm::tokenize_needle(dv, &crate::lpm::DictIndex::build(dv), prefix)?;

        // For each query token at position i, the remaining prefix at
        // that position is `prefix[byte_pos..]`. The valid-divergence
        // range is `prefix_range(prefix[byte_pos..])`.
        let mut intervals = Vec::with_capacity(query.len());
        let mut byte_pos = 0usize;
        for &tok in &query {
            let remaining = &prefix[byte_pos..];
            let range = prefix_range(dv, remaining);
            // Dict size is capped at 2^16 by OnPair training; `range.start`
            // and `range.end` are dict ids that comfortably fit in u32.
            let start = u32::try_from(range.start)
                .unwrap_or_else(|_| vortex_error::vortex_panic!("dict id > u32::MAX"));
            let end = u32::try_from(range.end)
                .unwrap_or_else(|_| vortex_error::vortex_panic!("dict id > u32::MAX"));
            intervals.push(start..end);
            // Advance by the token's true length.
            let entry = dv.dict_table[tok as usize];
            byte_pos += (entry & 0xffff) as usize;
        }
        debug_assert_eq!(byte_pos, prefix.len());
        Some(Self { query, intervals })
    }

    /// Returns `true` iff some prefix of the decoded row equals the
    /// literal prefix.
    #[inline]
    pub(crate) fn matches(&self, codes: &[u16]) -> bool {
        let q_len = self.query.len();
        if q_len == 0 {
            return true;
        }
        let mut pos = 0usize;
        // SAFETY: indexing bounded by `pos < q_len`.
        unsafe {
            for &code in codes {
                let want = *self.query.get_unchecked(pos);
                if code == want {
                    pos += 1;
                    if pos == q_len {
                        return true;
                    }
                } else {
                    let range = self.intervals.get_unchecked(pos);
                    let code_u32 = u32::from(code);
                    return code_u32 >= range.start && code_u32 < range.end;
                }
            }
        }
        // Ran out of row tokens before finishing the query → mismatch
        // unless we'd already returned `true` above.
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use super::*;
    use crate::DEFAULT_DICT12_CONFIG;
    use crate::decode::OwnedDecodeInputs;
    use crate::onpair_compress;

    fn build_inputs(strings: &[&str]) -> OwnedDecodeInputs {
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(s.as_bytes())),
            DType::Utf8(Nullability::NonNullable),
        );
        let arr =
            onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        OwnedDecodeInputs::collect(arr.as_view(), &mut ctx).unwrap()
    }

    fn row_codes(inputs: &OwnedDecodeInputs, r: usize) -> &[u16] {
        let lo = inputs.codes_offsets[r] as usize;
        let hi = inputs.codes_offsets[r + 1] as usize;
        &inputs.codes[lo..hi]
    }

    #[test]
    fn prefix_matches_decoded_truth() {
        let strings: &[&str] = &[
            "https://example.com/items/0001",
            "https://example.com/items/0002",
            "https://example.com/users/abc",
            "ftp://other.example.com/x",
            "http",
            "https",
            "h",
            "",
        ];
        let inputs = build_inputs(strings);
        let dv = inputs.view();

        for &prefix in &[
            &b"https://"[..],
            b"https://example.com/items/",
            b"ftp://",
            b"https",
            b"https:",
            b"missing",
            b"h",
            b"http",
            b"e",
        ] {
            let dfa = PrefixAutomaton::build(&dv, prefix);
            for (r, s) in strings.iter().enumerate() {
                let want = s.as_bytes().starts_with(prefix);
                let got = match dfa.as_ref() {
                    Some(d) => d.matches(row_codes(&inputs, r)),
                    None => false,
                };
                assert_eq!(
                    got, want,
                    "prefix={:?} row={s:?}",
                    std::str::from_utf8(prefix)
                );
            }
        }
    }

}
