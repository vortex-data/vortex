// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sorted-dictionary prefix lookup.
//!
//! Because the dictionary is in strict bytewise-lexicographic order, the set of
//! tokens sharing a given byte-prefix is a contiguous run of token ids. Both
//! operations here narrow such a run one byte at a time.

use crate::core::dictionary::DictionaryView;
use crate::core::types::{Token, TokenRange};

/// Narrow `range` to the tokens whose byte at index `k` equals `target` (and are
/// therefore longer than `k`), assuming every token in `range` already agrees on
/// bytes `0..k`. Returns the contiguous sub-range, or [`TokenRange::EMPTY`] if
/// none match.
///
/// Two binary searches over the sorted dictionary: the lower bound of the tokens
/// whose byte `k` is `>= target`, then the upper bound of those `<= target`.
/// Tokens shorter than `k + 1` sort before any that has byte `k`, so the
/// `token_len(t) <= k` guard both excludes them and keeps `token(t)[k]` in
/// bounds (it is only evaluated once that guard is false).
#[inline]
pub(super) fn narrow<V: DictionaryView>(
    dict: V,
    range: TokenRange,
    k: usize,
    target: u8,
) -> TokenRange {
    let tlen = |t: Token| -> usize { dict.token_len(t) };
    let byte_at = |t: Token| -> u8 { dict.token(t)[k] };

    // Lower bound: first token in range with byte[k] >= target.
    let (mut lo, mut hi) = (range.begin, range.last);
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if tlen(mid) <= k || byte_at(mid) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if tlen(lo) <= k || byte_at(lo) != target {
        return TokenRange::EMPTY;
    }
    let first = lo;

    // Upper bound: first token with byte[k] > target, stepped back to the last
    // with byte[k] == target. `lo` already holds `first`.
    hi = range.last;
    while lo < hi {
        let mid = lo + ((hi - lo) >> 1);
        if tlen(mid) <= k || byte_at(mid) <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let last = if tlen(lo) > k && byte_at(lo) > target {
        lo - 1
    } else {
        lo
    };

    TokenRange { begin: first, last }
}

/// The contiguous range of token ids that have `needle` as a byte-prefix — every
/// token at least `needle.len()` bytes long whose first `needle.len()` bytes
/// equal `needle` — or [`TokenRange::EMPTY`] if none.
///
/// An empty `needle` matches the whole dictionary. A `needle` longer than
/// [`MAX_TOKEN_SIZE`](crate::MAX_TOKEN_SIZE) yields the empty range (no token is
/// that long).
pub fn prefix_range<V: DictionaryView>(dict: V, needle: &[u8]) -> TokenRange {
    let n = dict.num_tokens();
    if n == 0 {
        return TokenRange::EMPTY;
    }
    let mut range = TokenRange {
        begin: 0,
        last: (n - 1) as Token,
    };
    for (k, &target) in needle.iter().enumerate() {
        range = narrow(dict, range, k, target);
        if range.is_empty() {
            return TokenRange::EMPTY;
        }
    }
    range
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, pad_raw};

    /// A sorted, read-padded compact dictionary of all 256 single bytes plus the
    /// given multi-byte tokens, returned alongside the id-ordered token list.
    fn build(extra: &[&[u8]]) -> (CompactDictionary, Vec<Vec<u8>>) {
        let mut toks: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        for t in extra {
            toks.push(t.to_vec());
        }
        toks.sort();
        toks.dedup();

        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in &toks {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        pad_raw(&mut bytes, &offsets);
        let dict = CompactDictionary::from_raw(bytes, offsets);
        (dict, toks)
    }

    /// Brute-force prefix range over the sorted token list (contiguous, since
    /// sorted): `(first_id, last_id)` of tokens starting with `needle`, or None.
    fn brute(toks: &[Vec<u8>], needle: &[u8]) -> Option<(usize, usize)> {
        let ids: Vec<usize> = toks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.starts_with(needle))
            .map(|(i, _)| i)
            .collect();
        ids.first().map(|&f| (f, *ids.last().unwrap()))
    }

    fn check(needle: &[u8], extra: &[&[u8]]) {
        let (dict, toks) = build(extra);
        let want = brute(&toks, needle);

        for r in [
            prefix_range(dict.as_view(), needle),
            prefix_range(dict.to_wide().as_view(), needle),
        ] {
            match want {
                None => assert!(r.is_empty(), "needle {needle:?} should be empty"),
                Some((f, l)) => {
                    assert_eq!(
                        (r.begin as usize, r.last as usize),
                        (f, l),
                        "needle {needle:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn single_byte_prefix() {
        // Every token starting with 'a', incl the bare "a" and the longer ones.
        check(b"a", &[b"ab", b"abc", b"abcd", b"bc", b"ca"]);
    }

    #[test]
    fn multi_byte_prefix_is_contiguous() {
        check(b"ab", &[b"ab", b"abc", b"abcd", b"bc", b"ca"]);
        check(b"abc", &[b"ab", b"abc", b"abcd", b"bc", b"ca"]);
    }

    #[test]
    fn exact_token_is_its_own_prefix() {
        check(b"abcd", &[b"ab", b"abc", b"abcd"]);
    }

    #[test]
    fn absent_prefix_is_empty() {
        check(b"xy", &[b"ab", b"abc"]); // no token starts with "xy"
        check(b"abce", &[b"ab", b"abc", b"abcd"]); // "abc" present but not "abce"
    }

    #[test]
    fn empty_needle_spans_whole_dictionary() {
        let (dict, toks) = build(&[b"ab"]);
        let r = prefix_range(dict.as_view(), b"");
        assert_eq!((r.begin as usize, r.last as usize), (0, toks.len() - 1));
    }
}
