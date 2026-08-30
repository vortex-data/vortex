// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain prefix search.
//!
//! A row starts with the needle iff its codes match the needle's token sequence
//! up to a single divergence point, where the row's token is a *longer* token
//! that begins with the remaining needle bytes (the needle's tail token absorbed
//! into a longer one). The needle is prepared once into a [`PrefixQuery`]: its
//! token sequence plus, per token position, the [`TokenRange`] of dictionary
//! tokens that extend the rest of the needle — the valid-divergence interval.
//! [`starts_with`] then scans a row's codes holding the match position in a
//! local; no decode and no dictionary at scan time (the intervals baked it in).

use super::lookup::prefix_range;
use super::tokenize::tokenize;
use crate::core::dictionary::DictionaryView;
use crate::core::types::{Token, TokenRange};

/// A needle prepared for prefix search: its canonical token sequence plus a
/// valid-divergence interval per token position. Built once against a
/// dictionary, shared (by `&`) across rows; holds no scan state.
#[derive(Debug, Clone)]
pub struct PrefixQuery {
    /// The needle's tokens (its greedy segmentation).
    tokens: Vec<Token>,
    /// `intervals[i]` = the tokens that begin with the needle bytes from token
    /// `i`'s start onward. A row token in this range at position `i` covers the
    /// whole remaining needle, so the row starts with the needle.
    intervals: Vec<TokenRange>,
}

impl PrefixQuery {
    /// Prepare `prefix` for prefix search against the sorted `dict`.
    pub fn new<V: DictionaryView>(prefix: &[u8], dict: V) -> Self {
        let tokens = tokenize(prefix, dict);
        let mut intervals = Vec::with_capacity(tokens.len());
        // intervals[i] ranges over the needle suffix starting at token i's byte.
        let mut byte_off = 0usize;
        for &t in &tokens {
            intervals.push(prefix_range(dict, &prefix[byte_off..]));
            byte_off += dict.token_len(t);
        }
        Self { tokens, intervals }
    }
}

/// Whether the row `codes` starts with the needle of `query`.
///
/// Finds the first token where the row diverges from the needle, then makes a single decision.
pub fn starts_with(codes: &[Token], query: &PrefixQuery) -> bool {
    let needle = &query.tokens;
    let min_len = codes.len().min(needle.len());

    // First position where the row diverges from the needle, or `min_len` if it
    // agrees that far. Pre-sliced so the compare loop carries no bounds checks.
    let diff = std::iter::zip(&codes[..min_len], &needle[..min_len])
        .position(|(c, t)| c != t)
        .unwrap_or(min_len);

    if diff == needle.len() {
        return true; // every needle token matched (row is at least as long); also empty needle
    }
    if diff < codes.len() {
        // Diverged before the row ended: valid iff the row's token there extends
        // the rest of the needle.
        return query.intervals[diff].contains(codes[diff]);
    }
    false // the row ended before the needle was satisfied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::Dictionary;
    use crate::{Column, DEFAULT_CONFIG, compress};

    fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
    }

    /// Decode row `k` to bytes via the into-buffer API, for the oracle.
    fn decode_row(view: crate::ColumnView<'_, u32>, k: usize) -> Vec<u8> {
        let mut buf =
            vec![std::mem::MaybeUninit::uninit(); view.row_decoded_len(k) + crate::DECODE_PADDING];
        // SAFETY: buffer sized for row `k`; view from a trusted column.
        let w = unsafe { view.decompress_row_into(k, &mut buf) };
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) }.to_vec()
    }

    /// Driving `starts_with` over every row must agree with a brute-force
    /// decode-and-`starts_with` oracle. The query is built from both the compact
    /// and the wide dictionary to confirm construction is representation-agnostic.
    fn check(rows: &[&[u8]], prefixes: &[&[u8]]) {
        let col = compress_rows(rows);
        let view = col.view();
        let wide = view.dict.to_wide();
        for &prefix in prefixes {
            let want: Vec<usize> = (0..view.num_rows())
                .filter(|&k| decode_row(view, k).starts_with(prefix))
                .collect();

            for query in [
                PrefixQuery::new(prefix, view.dict),
                PrefixQuery::new(prefix, wide.as_view()),
            ] {
                let got: Vec<usize> = (0..view.num_rows())
                    .filter(|&k| starts_with(view.row_codes(k), &query))
                    .collect();
                assert_eq!(got, want, "prefix {prefix:?}");
            }
        }
    }

    #[test]
    fn empty_prefix_matches_all_rows() {
        let rows: &[&[u8]] = &[b"a", b"", b"abc"];
        check(rows, &[b""]);
    }

    #[test]
    fn whole_string_and_exact_prefixes() {
        let rows: &[&[u8]] = &[b"alpha", b"alpine", b"beta", b"al"];
        check(
            rows,
            &[b"al", b"alp", b"alpha", b"beta", b"b", b"alphas", b"z"],
        );
    }

    #[test]
    fn prefix_ending_inside_a_token() {
        // Repetitive rows build multi-byte tokens; these prefixes end mid-token,
        // exercising the divergence-interval branch (the boundary token).
        let rows: &[&[u8]] = &[b"abcdef", b"abcxyz", b"abxxxx", b"abc", b"ab"];
        check(rows, &[b"a", b"ab", b"abc", b"abcd", b"abcx", b"abz"]);
    }

    #[test]
    fn matches_brute_force_on_repetitive_corpus() {
        use crate::test_corpus::user_strings;
        let corpus: Vec<Vec<u8>> = user_strings(50)
            .into_iter()
            .map(String::into_bytes)
            .collect();
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        let prefixes: &[&[u8]] = &[
            b"",
            b"h",
            b"https",
            b"https://www.",
            b"https://www.example.com/",
            b"ftp://",
            b"zzz",
        ];
        check(&rows, prefixes);
    }

    #[test]
    fn matches_brute_force_on_binary_corpus() {
        use crate::test_corpus::binary_strings;
        let corpus = binary_strings(40, 24, 19);
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        // Probe with each row's own leading bytes plus a few unlikely prefixes.
        let mut prefixes: Vec<&[u8]> = rows.iter().map(|r| &r[..r.len().min(3)]).collect();
        prefixes.push(b"");
        prefixes.push(b"\x00\x00");
        check(&rows, &prefixes);
    }
}
