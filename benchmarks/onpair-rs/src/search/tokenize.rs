// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Greedy longest-match tokenisation of a byte string against the dictionary.

use super::lookup::narrow;
use crate::core::dictionary::DictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};

/// Greedy longest-match tokenisation of `text` against the sorted dictionary,
/// matching the encoder's own segmentation.
///
/// At each position it takes the longest dictionary token that is a prefix of
/// the remaining bytes, narrowing an inclusive `[begin, last]` window of
/// candidate token ids one byte at a time via binary search over the sorted
/// dictionary.
///
/// # When to use this
/// For extensive parsing the encoder's longest-prefix matcher
/// ([`Parser`](crate::Parser)) is the right structure — it amortizes its build
/// cost across the whole input. Reach for `tokenize` when you only need to encode
/// a little (a query needle, say), where rebuilding that matcher would cost more
/// than it saves.
///
/// # Precondition
/// `dict` is conformant: **sorted** (strict bytewise-lexicographic order) and
/// **complete** (all 256 single-byte tokens present). These properties are
/// guaranteed for any dictionary trained by [`Parser::train`](crate::Parser::train)
/// or passed through [`CompactDictionary::validate`](crate::CompactDictionary::validate).
/// Completeness makes every byte encodable, so the loop always consumes at least
/// one byte and terminates. See [`crate::search`] for the general search
/// precondition.
pub fn tokenize<V: DictionaryView>(text: &[u8], dict: V) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(text.len());

    let mut pos = 0;
    while pos < text.len() {
        let max_len = (text.len() - pos).min(MAX_TOKEN_SIZE);

        let mut best: Token = 0;
        // Candidate window over the sorted dictionary, narrowed byte by byte.
        let mut range = TokenRange {
            begin: 0,
            last: (dict.num_tokens() - 1) as Token,
        };

        for k in 0..max_len {
            let next = narrow(dict, range, k, text[pos + k]);
            if next.is_empty() {
                break;
            }
            // The shortest token in range sorts first; length exactly k+1 means
            // it is an exact match of the bytes consumed so far.
            if dict.token_len(next.begin) == k + 1 {
                best = next.begin;
            }
            range = next;
        }

        tokens.push(best);
        pos += dict.token_len(best);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary, pad_raw};
    use crate::test_corpus::{binary_strings, make_raw, random_ascii_strings, user_strings};
    use crate::{DECODE_PADDING, DEFAULT_CONFIG, MAX_TOKEN_SIZE, Parser, decode_into, decoded_len};

    /// Decode `codes` against `dict` into a fresh `Vec` (test helper; the crate
    /// exposes only the into-buffer decoder).
    fn decode_to_vec<V: DictionaryView>(codes: &[Token], dict: V) -> Vec<u8> {
        let n = decoded_len(codes, dict);
        let mut out = Vec::with_capacity(n + DECODE_PADDING);
        // SAFETY: a built/padded test dict is read-padded, codes are in range, and
        // the buffer carries DECODE_PADDING headroom.
        let w = unsafe { decode_into(codes, dict, out.spare_capacity_mut()) };
        // SAFETY: the decoder initialized exactly `w` leading bytes.
        unsafe { out.set_len(w) };
        out
    }

    /// A sorted, complete (all 256 single bytes), read-padded compact dictionary
    /// built from the given `extra` multi-byte tokens. Returns it alongside the
    /// id-ordered token list (token id == index), for the reference tokenizer.
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

    /// Naive greedy longest-match against an id-ordered token list — the
    /// definition `tokenize` optimizes with binary search.
    fn reference(text: &[u8], toks: &[Vec<u8>]) -> Vec<Token> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let (mut best_id, mut best_len) = (0usize, 0usize);
            for (id, t) in toks.iter().enumerate() {
                if t.len() <= MAX_TOKEN_SIZE
                    && t.len() <= text.len() - pos
                    && t.len() > best_len
                    && text[pos..pos + t.len()] == t[..]
                {
                    best_id = id;
                    best_len = t.len();
                }
            }
            out.push(best_id as Token);
            pos += best_len.max(1); // complete dictionary ⇒ best_len >= 1
        }
        out
    }

    /// `tokenize` over both representations agrees with the reference, and the
    /// emitted tokens decode back to the input.
    fn check(text: &[u8], extra: &[&[u8]]) {
        let (dict, toks) = build(extra);
        let want = reference(text, &toks);

        let compact = dict.as_view();
        assert_eq!(tokenize(text, compact), want, "compact tokens");

        let wide = dict.to_wide();
        assert_eq!(tokenize(text, wide.as_view()), want, "wide tokens");

        assert_eq!(decode_to_vec(&want, compact).as_slice(), text, "round-trip");
    }

    #[test]
    fn empty_text_yields_no_tokens() {
        let (dict, _) = build(&[]);
        assert!(tokenize(b"", dict.as_view()).is_empty());
    }

    #[test]
    fn single_byte_only_dictionary() {
        // No multi-byte tokens: every position is one single-byte token.
        check(b"hello, world!", &[]);
    }

    #[test]
    fn longest_match_prefers_multibyte_token() {
        // "ab" present ⇒ "abab" is two tokens, not four single bytes.
        let (dict, toks) = build(&[b"ab"]);
        let ab = toks.iter().position(|t| t == b"ab").unwrap() as Token;
        assert_eq!(tokenize(b"abab", dict.as_view()), vec![ab, ab]);
    }

    #[test]
    fn greedy_over_overlapping_tokens() {
        let extra: &[&[u8]] = &[b"ab", b"abc", b"abcd", b"bc", b"ca"];
        check(b"ababcabcabcd", extra);
    }

    #[test]
    fn matches_reference_on_mixed_corpus() {
        let extra: &[&[u8]] = &[
            b"th", b"the", b"he", b"er", b"ing", b"in", b"on", b"onpair", b"pair",
        ];
        check(b"the onpair thing is interesting in the morning", extra);
    }

    #[test]
    fn handles_max_length_token() {
        let long = [b'z'; MAX_TOKEN_SIZE];
        let extra: &[&[u8]] = &[&long, b"zz"];
        let text = vec![b'z'; MAX_TOKEN_SIZE + 2]; // forces a 2nd token after the full-width one
        check(&text, extra);
    }

    // ── Trained-dictionary tests ──────────────────────────────────────────────
    //
    // The strongest oracle: a real trained dictionary, with `tokenize` checked
    // against the encoder's own segmentation. The encoder finds the longest
    // match with a hash matcher, `tokenize` with binary search over the sorted
    // dictionary — they must agree, because reproducing the encoder's tokens is
    // exactly tokenize's contract.

    fn to_bytes(strings: Vec<String>) -> Vec<Vec<u8>> {
        strings.into_iter().map(String::into_bytes).collect()
    }

    /// Train a parser on `train`, then for every string in `inputs` assert that
    /// `tokenize`: (a) reproduces the encoder's stored codes, (b) agrees across
    /// the compact and wide representations, (c) round-trips to the input, and
    /// (d) never emits more tokens than bytes. `inputs` may differ from `train`
    /// (unseen strings).
    fn check_against_encoder(train: &[Vec<u8>], inputs: &[Vec<u8>]) {
        let traw = make_raw(train);
        let parser = Parser::train(&traw.data, &traw.offsets, DEFAULT_CONFIG).unwrap();
        let dict = parser.dict.as_view();
        let wide = parser.dict.to_wide();

        // The encoder's segmentation is what it actually stored as each row's codes.
        let iraw = make_raw(inputs);
        let col = parser.parse(&iraw.data, &iraw.offsets).unwrap();
        let view = col.view();

        for (i, s) in inputs.iter().enumerate() {
            let toks = tokenize(s, dict);
            assert_eq!(
                toks.as_slice(),
                view.row_codes(i),
                "row {i}: tokenize vs encoder"
            );
            assert_eq!(
                tokenize(s, wide.as_view()),
                toks,
                "row {i}: wide vs compact"
            );
            assert_eq!(
                decode_to_vec(&toks, dict).as_slice(),
                s.as_slice(),
                "row {i}: round-trip"
            );
            assert!(toks.len() <= s.len(), "row {i}: more tokens than bytes");
        }
    }

    #[test]
    fn matches_encoder_on_repetitive_corpus() {
        // URL-shaped strings train rich multi-byte tokens — the merge case.
        let corpus = to_bytes(user_strings(50));
        check_against_encoder(&corpus, &corpus);
    }

    #[test]
    fn matches_encoder_on_binary_strings() {
        // Full 0..=255 byte range — exercises high bytes in the sorted-dict search.
        let corpus = binary_strings(50, 30, 13);
        check_against_encoder(&corpus, &corpus);
    }

    #[test]
    fn matches_encoder_on_unseen_strings() {
        // Train on one random corpus, tokenize a disjoint one: completeness makes
        // any input encodable, and the segmentation must still match the encoder.
        let train = random_ascii_strings(100, 50, 77);
        let unseen = random_ascii_strings(20, 40, 999);
        check_against_encoder(&train, &unseen);
    }

    #[test]
    fn every_byte_value_tokenizes_to_its_base_token() {
        // Completeness: all 256 single-byte base tokens are present, so every byte
        // (including 0x00 and the high 0x80..=0xFF range) is one base token.
        let corpus = to_bytes(user_strings(20));
        let raw = make_raw(&corpus);
        let parser = Parser::train(&raw.data, &raw.offsets, DEFAULT_CONFIG).unwrap();
        let dict = parser.dict.as_view();
        for b in 0u16..256 {
            let s = [b as u8];
            let toks = tokenize(&s, dict);
            assert_eq!(toks.len(), 1, "byte {b}: expected exactly one token");
            assert_eq!(dict.token(toks[0]), &s, "byte {b}: reconstruction mismatch");
        }
    }
}
