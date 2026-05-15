// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/encoding/parsing/parser.h` and `parser.cpp`.
//
// Drives the LongestPrefixMatcher over every input string, writes the
// resulting token IDs into a `Store` via `BitWriter`, and records per-string
// token-count boundaries.

use crate::bits::BitWriter;
use crate::lpm::LongestPrefixMatcher;
use crate::store::Store;
use crate::types::BitWidth;

/// Encode all `n` strings into `store` using `lpm`.
///
/// `offsets` has length `n + 1`; string `i` occupies
/// `data[offsets[i]..offsets[i + 1]]`. On entry `store` is reset.
pub fn parse(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    lpm: &LongestPrefixMatcher,
    bits: BitWidth,
    store: &mut Store,
) {
    store.bit_width = bits;
    store.packed.clear();
    store.boundaries.clear();
    store.boundaries.reserve(n + 1);
    store.boundaries.push(0);

    let mut writer = BitWriter::new(store);
    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0u32);

    for i in 0..n {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        let mut pos = s;
        while pos < e {
            let (tok, mlen) = lpm.find_longest_match(&data[pos..e]);
            writer.write(tok);
            pos += mlen;
        }
        boundaries.push(writer.tokens_written() as u32);
    }
    drop(writer); // flush packed words + sentinel
    store.boundaries = boundaries;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::unpack_codes_to_u16;
    use crate::config::{FixedThreshold, ThresholdSpec, TrainingConfig};
    use crate::dict::Dictionary;
    use crate::trainer::{TrainResult, train};
    use crate::types::Token;

    use crate::test_corpus::{
        alternating_strings as make_alternating_strings, binary_strings as make_binary_strings,
        homogeneous_strings as make_homogeneous_strings, make_raw,
        mixed_length_strings as make_mixed_length_strings,
        random_ascii_strings as make_random_strings, user_strings as make_user_strings,
    };

    fn make_base_dict() -> Dictionary {
        let mut d = Dictionary::default();
        d.offsets.push(0);
        for i in 0u16..=255 {
            d.bytes.push(i as u8);
            d.offsets.push(d.bytes.len() as u32);
        }
        d
    }

    /// Decode all tokens for row `idx` against `dict`. Equivalent of the
    /// C++ `decode_tokens` helper in test_parser.cpp.
    fn decode_tokens(store: &Store, dict: &Dictionary, idx: usize) -> Vec<u8> {
        let begin = store.boundaries[idx] as usize;
        let end = store.boundaries[idx + 1] as usize;
        let codes = unpack_codes_to_u16(&store.packed, end, store.bit_width as u32);
        let mut out = Vec::new();
        for &c in &codes[begin..end] {
            out.extend_from_slice(dict.data(c as Token));
        }
        out
    }

    fn expected_packed_words(n: usize, bits: BitWidth) -> usize {
        (n * bits as usize).div_ceil(64)
    }

    fn roundtrip_all<S: AsRef<[u8]>>(strings: &[S], bits: BitWidth, seed: u64) -> bool {
        if strings.is_empty() {
            return true;
        }
        let raw = make_raw(strings);
        let cfg = TrainingConfig {
            bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(seed),
        };
        let TrainResult { dict, lpm } = train(&raw.data, &raw.offsets, raw.n, &cfg);
        let mut store = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, bits, &mut store);
        for i in 0..strings.len() {
            let decoded = decode_tokens(&store, &dict, i);
            if decoded != strings[i].as_ref() {
                return false;
            }
        }
        true
    }

    const WIDTHS: &[BitWidth] = &[9, 10, 11, 12, 13, 14, 15, 16];

    // ── Degenerate inputs ──────────────────────────────────────────────────

    #[test]
    fn zero_strings_produces_one_boundary() {
        let lpm = LongestPrefixMatcher::new();
        let mut store = Store::default();
        parse(&[], &[0], 0, &lpm, 16, &mut store);
        assert_eq!(store.boundaries, vec![0u32]);
        assert!(store.packed.is_empty());
        assert_eq!(store.bit_width, 16);
    }

    #[test]
    fn single_empty_string_produces_two_zero_boundaries() {
        let lpm = LongestPrefixMatcher::new();
        let mut store = Store::default();
        parse(&[], &[0, 0], 1, &lpm, 16, &mut store);
        assert_eq!(store.boundaries, vec![0u32, 0]);
        assert_eq!(store.num_tokens(), 0);
        assert!(store.packed.is_empty());
    }

    #[test]
    fn many_empty_strings_all_boundaries_are_zero() {
        let lpm = LongestPrefixMatcher::new();
        let offsets = vec![0u32; 51];
        let mut store = Store::default();
        parse(&[], &offsets, 50, &lpm, 16, &mut store);
        assert_eq!(store.boundaries.len(), 51);
        for b in &store.boundaries {
            assert_eq!(*b, 0);
        }
        assert_eq!(store.num_tokens(), 0);
        assert!(store.packed.is_empty());
    }

    // ── Structural invariants over all bit widths ──────────────────────────

    #[test]
    fn boundary_count_is_n_plus_one() {
        for &bits in WIDTHS {
            let lpm = LongestPrefixMatcher::new();
            let raw = make_raw(&make_user_strings(20));
            let mut store = Store::default();
            parse(&raw.data, &raw.offsets, raw.n, &lpm, bits, &mut store);
            assert_eq!(store.boundaries.len(), raw.n + 1);
            assert_eq!(store.bit_width, bits);
        }
    }

    #[test]
    fn boundaries_are_monotonic() {
        for &bits in WIDTHS {
            let lpm = LongestPrefixMatcher::new();
            let raw = make_raw(&make_random_strings(25, 40, 7));
            let mut store = Store::default();
            parse(&raw.data, &raw.offsets, raw.n, &lpm, bits, &mut store);
            for i in 1..store.boundaries.len() {
                assert!(
                    store.boundaries[i] >= store.boundaries[i - 1],
                    "non-monotonic at index {i}"
                );
            }
        }
    }

    #[test]
    fn last_boundary_equals_total_token_count() {
        for &bits in WIDTHS {
            let lpm = LongestPrefixMatcher::new();
            let raw = make_raw(&make_random_strings(15, 30, 99));
            let mut store = Store::default();
            parse(&raw.data, &raw.offsets, raw.n, &lpm, bits, &mut store);
            assert_eq!(*store.boundaries.last().unwrap() as usize, store.num_tokens());
        }
    }

    #[test]
    fn packed_size_consistent_with_token_count() {
        for &bits in WIDTHS {
            let lpm = LongestPrefixMatcher::new();
            let raw = make_raw(&make_user_strings(20));
            let mut store = Store::default();
            parse(&raw.data, &raw.offsets, raw.n, &lpm, bits, &mut store);
            assert_eq!(store.packed.len(), expected_packed_words(store.num_tokens(), bits) + 1);
        }
    }

    // ── Round-trip with base-tokens LPM ────────────────────────────────────

    #[test]
    fn base_tokens_single_known_string() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let expected = "Hello, World!";
        let raw = make_raw(&[expected]);
        let mut store = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, 16, &mut store);
        assert_eq!(decode_tokens(&store, &d, 0), expected.as_bytes());
    }

    #[test]
    fn base_tokens_all_single_byte_values() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let strings: Vec<Vec<u8>> = (0u16..=255).map(|i| vec![i as u8]).collect();
        let raw = make_raw(&strings);
        let mut store = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, 16, &mut store);
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(decode_tokens(&store, &d, i), *s, "mismatch for byte value {i}");
        }
    }

    #[test]
    fn base_tokens_multiple_strings() {
        let lpm = LongestPrefixMatcher::new();
        let d = make_base_dict();
        let strings = make_random_strings(30, 20, 2024);
        let raw = make_raw(&strings);
        let mut store = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, 16, &mut store);
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(decode_tokens(&store, &d, i), *s, "decode mismatch at string {i}");
        }
    }

    // ── Trained LPM produces multi-byte tokens ─────────────────────────────

    #[test]
    fn trained_lpm_produces_multi_byte_tokens() {
        let strings = make_homogeneous_strings(50, 40, b'a');
        let raw = make_raw(&strings);
        let cfg = TrainingConfig {
            bits: 16,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
        };
        let TrainResult { dict: _, lpm } = train(&raw.data, &raw.offsets, raw.n, &cfg);
        let mut store = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, 16, &mut store);
        let tokens_0 = store.boundaries[1] - store.boundaries[0];
        assert!(tokens_0 < 40, "parser did not use any multi-byte tokens");
    }

    // ── Round-trip with trained LPM across all bit widths ──────────────────

    #[test]
    fn roundtrip_user_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_user_strings(50), bits, 42));
        }
    }

    #[test]
    fn roundtrip_random_ascii_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_random_strings(60, 50, 1337), bits, 42));
        }
    }

    #[test]
    fn roundtrip_binary_strings_with_nul_bytes() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_binary_strings(40, 30, 777), bits, 42));
        }
    }

    #[test]
    fn roundtrip_homogeneous_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_homogeneous_strings(30, 40, b'a'), bits, 42));
        }
    }

    #[test]
    fn roundtrip_alternating_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_alternating_strings(30, 40), bits, 42));
        }
    }

    #[test]
    fn roundtrip_mixed_length_strings() {
        for &bits in WIDTHS {
            assert!(roundtrip_all(&make_mixed_length_strings(80, 100, 31415), bits, 42));
        }
    }
}
