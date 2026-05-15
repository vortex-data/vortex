// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/detail/tokenize.h`.
//
// Greedy longest-match tokenisation of a byte string against a sorted
// dictionary. The C++ original drives a binary search at each prefix length
// directly over the sorted dictionary. We re-use [`LongestPrefixMatcher`]
// built from the dictionary instead — it gives the same result (greedy
// longest match) and is O(MAX_TOKEN_SIZE) per token regardless of dict size.
//
// Precondition: the dictionary must contain every single-byte token
// (guaranteed by [`crate::train`]). This makes tokenisation total.

use crate::dict::Dictionary;
use crate::lpm::LongestPrefixMatcher;
use crate::types::Token;

/// Tokenise `text` against `dict` via greedy longest match.
pub fn tokenize(text: &[u8], dict: &Dictionary) -> Vec<Token> {
    if text.is_empty() {
        return Vec::new();
    }
    let lpm = LongestPrefixMatcher::from_dictionary(dict);
    tokenize_with(text, &lpm)
}

/// Tokenise using a pre-built LPM (skip the build cost when tokenising
/// many strings against the same dictionary).
pub fn tokenize_with(text: &[u8], lpm: &LongestPrefixMatcher) -> Vec<Token> {
    let mut out = Vec::with_capacity(text.len());
    let mut pos = 0;
    while pos < text.len() {
        let (t, len) = lpm.find_longest_match(&text[pos..]);
        out.push(t);
        pos += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FixedThreshold, ThresholdSpec, TrainingConfig};
    use crate::test_corpus::{
        binary_strings, make_raw, random_ascii_strings, single_byte_strings, user_strings,
    };
    use crate::trainer::{TrainResult, train};

    fn reconstruct(tokens: &[Token], dict: &Dictionary) -> Vec<u8> {
        let mut out = Vec::new();
        for &t in tokens {
            out.extend_from_slice(dict.data(t));
        }
        out
    }

    fn train_dict<S: AsRef<[u8]>>(corpus: &[S]) -> Dictionary {
        let raw = make_raw(corpus);
        let cfg = TrainingConfig { seed: Some(42), ..Default::default() };
        let TrainResult { dict, .. } = train(&raw.data, &raw.offsets, raw.n, &cfg);
        dict
    }

    // ── Empty input ───────────────────────────────────────────────────────

    #[test]
    fn empty_string_returns_no_tokens() {
        let d = train_dict(&user_strings(10));
        assert!(tokenize(b"", &d).is_empty());
    }

    // ── Single byte ───────────────────────────────────────────────────────

    #[test]
    fn single_byte_produces_one_token() {
        let d = train_dict(&user_strings(10));
        let t = tokenize(b"x", &d);
        assert_eq!(t.len(), 1);
        assert_eq!(reconstruct(&t, &d), b"x");
    }

    // ── Round-trip ─────────────────────────────────────────────────────────

    #[test]
    fn reconstruction_matches_input() {
        let corpus = user_strings(50);
        let d = train_dict(&corpus);
        for s in &corpus {
            let t = tokenize(s.as_bytes(), &d);
            assert_eq!(reconstruct(&t, &d), s.as_bytes(), "row {s}");
        }
    }

    #[test]
    fn reconstruction_with_random_strings() {
        let corpus = random_ascii_strings(100, 50, 77);
        let d = train_dict(&corpus);
        let unseen = random_ascii_strings(20, 40, 999);
        for s in &unseen {
            let t = tokenize(s, &d);
            assert_eq!(reconstruct(&t, &d), *s);
        }
    }

    #[test]
    fn reconstruction_with_binary_strings() {
        let corpus = binary_strings(50, 30, 13);
        let d = train_dict(&corpus);
        for s in &corpus {
            let t = tokenize(s, &d);
            assert_eq!(reconstruct(&t, &d), *s);
        }
    }

    // ── Greedy longest match ──────────────────────────────────────────────

    #[test]
    fn greedy_longest_match_compresses() {
        let corpus: Vec<&str> = (0..100).map(|_| "aabb").collect();
        let raw = make_raw(&corpus);
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        let TrainResult { dict, .. } = train(&raw.data, &raw.offsets, raw.n, &cfg);
        let tokens = tokenize(b"aabb", &dict);
        assert!(tokens.len() < 4, "greedy tokenisation should compress");
        assert_eq!(reconstruct(&tokens, &dict), b"aabb");
    }

    // ── All 256 base tokens ───────────────────────────────────────────────

    #[test]
    fn all_256_bytes_tokenisable_via_base_tokens() {
        let d = train_dict(&single_byte_strings());
        for b in 0u16..=255 {
            let s = [b as u8];
            let t = tokenize(&s, &d);
            assert_eq!(t.len(), 1, "byte {b} not tokenised");
            assert_eq!(reconstruct(&t, &d), &s[..], "byte {b} mismatch");
        }
    }

    // ── Bound: token count <= byte count ──────────────────────────────────

    #[test]
    fn token_count_never_exceeds_string_length() {
        let corpus = user_strings(50);
        let d = train_dict(&corpus);
        for s in &corpus {
            let t = tokenize(s.as_bytes(), &d);
            assert!(t.len() <= s.len(), "more tokens than bytes for {s}");
        }
    }

    // ── Consistency with parser ───────────────────────────────────────────

    #[test]
    fn tokenize_matches_parser_output() {
        let corpus = user_strings(50);
        let raw = make_raw(&corpus);
        let cfg = TrainingConfig { seed: Some(42), ..Default::default() };
        let TrainResult { dict, lpm } = train(&raw.data, &raw.offsets, raw.n, &cfg);
        for s in &corpus {
            let tokens_a = tokenize(s.as_bytes(), &dict);
            // Tokenise directly via the trained LPM.
            let tokens_b = tokenize_with(s.as_bytes(), &lpm);
            assert_eq!(tokens_a, tokens_b, "disagreement on {s}");
        }
    }
}
