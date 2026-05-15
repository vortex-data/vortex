// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Pure-Rust decoder. Mirrors the FFI surface
// `onpair_column_decompress` (point) and the bulk `decode_all` family from
// `include/onpair/decoding/`.
//
// The hot loop is one `u16` code load + one `(offset, length)` dictionary
// lookup + one `extend_from_slice`. The trainer's `pad_for_decoder()` call
// guarantees that every token's first byte is followed by at least
// `MAX_TOKEN_SIZE` readable bytes, so callers that prefer SIMD-style
// fixed-width over-copy can do so safely against `Column::dict_bytes_padded`.

use crate::bit_unpack::read_bits_lsb;
use crate::dict::Dictionary;
use crate::store::Store;
use crate::types::Token;

/// Append the bytes of the token sequence `codes` to `out` using `dict`.
#[inline]
pub fn decode_codes(dict: &Dictionary, codes: &[Token], out: &mut Vec<u8>) {
    for &c in codes {
        let s = dict.offsets[c as usize] as usize;
        let e = dict.offsets[c as usize + 1] as usize;
        out.extend_from_slice(&dict.bytes[s..e]);
    }
}

/// Iterate the unpacked token IDs for row `row_id` in `store`.
///
/// Yields one `Token` per packed slot; suitable for the predicate hot path
/// when we only need codes, not bytes.
#[inline]
pub fn row_codes(store: &Store, row_id: usize) -> impl Iterator<Item = Token> + '_ {
    let bits = store.bit_width as u32;
    let span = store.string_span(row_id);
    (span.begin..span.end).map(move |i| read_bits_lsb(&store.packed, (i as usize) * (bits as usize), bits))
}

/// Decompress row `row_id` into `out`, clearing it first.
pub fn decompress_row(dict: &Dictionary, store: &Store, row_id: usize, out: &mut Vec<u8>) {
    out.clear();
    let bits = store.bit_width as u32;
    let span = store.string_span(row_id);
    let n = (span.end - span.begin) as usize;
    out.reserve(n * 4); // rough heuristic, average ≥ 1 byte per token
    for i in span.begin..span.end {
        let code = read_bits_lsb(&store.packed, (i as usize) * (bits as usize), bits);
        let s = dict.offsets[code as usize] as usize;
        let e = dict.offsets[code as usize + 1] as usize;
        out.extend_from_slice(&dict.bytes[s..e]);
    }
}

/// Decode every row into a flat byte buffer plus Arrow-style `n + 1` offsets.
/// Returns `(bytes, offsets)` where `bytes[offsets[i]..offsets[i+1]]` is row
/// `i`'s content.
pub fn decode_all(dict: &Dictionary, store: &Store) -> (Vec<u8>, Vec<u32>) {
    let n = store.num_strings();
    let mut bytes = Vec::with_capacity(store.num_tokens() * 2);
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0u32);
    let bits = store.bit_width as u32;
    for row in 0..n {
        let span = store.string_span(row);
        for i in span.begin..span.end {
            let code = read_bits_lsb(&store.packed, (i as usize) * (bits as usize), bits);
            let s = dict.offsets[code as usize] as usize;
            let e = dict.offsets[code as usize + 1] as usize;
            bytes.extend_from_slice(&dict.bytes[s..e]);
        }
        offsets.push(bytes.len() as u32);
    }
    (bytes, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FixedThreshold, ThresholdSpec, TrainingConfig};
    use crate::parser::parse;
    use crate::test_corpus::*;
    use crate::trainer::{TrainResult, train};
    use crate::types::BitWidth;

    fn compress<S: AsRef<[u8]>>(strings: &[S], bits: BitWidth, seed: u64) -> (Dictionary, Store) {
        let raw = make_raw(strings);
        let cfg = TrainingConfig {
            bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(seed),
        };
        let TrainResult { dict, lpm } = train(&raw.data, &raw.offsets, raw.n, &cfg);
        let mut store = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, bits, &mut store);
        (dict, store)
    }

    const WIDTHS: &[BitWidth] = &[9, 10, 11, 12, 13, 14, 15, 16];

    // ── decompress_row ────────────────────────────────────────────────────

    #[test]
    fn decompress_row_matches_input() {
        let strings = user_strings(20);
        for &bits in WIDTHS {
            let (dict, store) = compress(&strings, bits, 42);
            let mut buf = Vec::new();
            for (i, s) in strings.iter().enumerate() {
                decompress_row(&dict, &store, i, &mut buf);
                assert_eq!(buf, s.as_bytes(), "bits={bits} row={i}");
            }
        }
    }

    #[test]
    fn decompress_row_clears_existing_contents() {
        let strings = ["hello", "world"];
        let (dict, store) = compress(&strings, 12, 1);
        let mut buf = b"GARBAGE".to_vec();
        decompress_row(&dict, &store, 0, &mut buf);
        assert_eq!(buf, b"hello");
    }

    #[test]
    fn decompress_row_handles_empty_string() {
        let strings: Vec<&[u8]> = vec![b"", b"x", b""];
        let (dict, store) = compress(&strings, 12, 1);
        let mut buf = Vec::new();
        decompress_row(&dict, &store, 0, &mut buf);
        assert!(buf.is_empty());
        decompress_row(&dict, &store, 1, &mut buf);
        assert_eq!(buf, b"x");
        decompress_row(&dict, &store, 2, &mut buf);
        assert!(buf.is_empty());
    }

    // ── decode_all ────────────────────────────────────────────────────────

    #[test]
    fn decode_all_matches_input() {
        let strings: Vec<Vec<u8>> = user_strings(40).into_iter().map(|s| s.into_bytes()).collect();
        for &bits in WIDTHS {
            let (dict, store) = compress(&strings, bits, 7);
            let (bytes, offsets) = decode_all(&dict, &store);
            assert_eq!(offsets.len(), strings.len() + 1);
            for (i, s) in strings.iter().enumerate() {
                let lo = offsets[i] as usize;
                let hi = offsets[i + 1] as usize;
                assert_eq!(&bytes[lo..hi], &s[..], "bits={bits} row={i}");
            }
        }
    }

    #[test]
    fn decode_all_binary_corpus() {
        let strings = binary_strings(40, 30, 777);
        let (dict, store) = compress(&strings, 14, 11);
        let (bytes, offsets) = decode_all(&dict, &store);
        for (i, s) in strings.iter().enumerate() {
            let lo = offsets[i] as usize;
            let hi = offsets[i + 1] as usize;
            assert_eq!(&bytes[lo..hi], &s[..]);
        }
    }

    #[test]
    fn decode_all_empty_corpus_returns_single_zero_offset() {
        let strings: Vec<&[u8]> = vec![];
        let (dict, store) = compress(&strings, 12, 1);
        let (bytes, offsets) = decode_all(&dict, &store);
        assert!(bytes.is_empty());
        assert_eq!(offsets, vec![0u32]);
    }

    #[test]
    fn decode_all_homogeneous() {
        let strings = homogeneous_strings(30, 40, b'a');
        let (dict, store) = compress(&strings, 12, 1);
        let (bytes, offsets) = decode_all(&dict, &store);
        for (i, s) in strings.iter().enumerate() {
            let lo = offsets[i] as usize;
            let hi = offsets[i + 1] as usize;
            assert_eq!(&bytes[lo..hi], &s[..]);
        }
    }

    #[test]
    fn decode_all_alternating() {
        let strings = alternating_strings(30, 40);
        let (dict, store) = compress(&strings, 12, 1);
        let (bytes, offsets) = decode_all(&dict, &store);
        for (i, s) in strings.iter().enumerate() {
            let lo = offsets[i] as usize;
            let hi = offsets[i + 1] as usize;
            assert_eq!(&bytes[lo..hi], &s[..]);
        }
    }

    #[test]
    fn decode_all_mixed_length() {
        let strings = mixed_length_strings(80, 60, 314);
        let (dict, store) = compress(&strings, 12, 1);
        let (bytes, offsets) = decode_all(&dict, &store);
        for (i, s) in strings.iter().enumerate() {
            let lo = offsets[i] as usize;
            let hi = offsets[i + 1] as usize;
            assert_eq!(&bytes[lo..hi], &s[..]);
        }
    }

    // ── row_codes ─────────────────────────────────────────────────────────

    #[test]
    fn row_codes_iter_matches_decompress() {
        let strings = ["abcabcabc", "different_row", "abcabc"];
        let (dict, store) = compress(&strings, 12, 1);
        for (i, s) in strings.iter().enumerate() {
            let mut reconstructed = Vec::new();
            for code in row_codes(&store, i) {
                let lo = dict.offsets[code as usize] as usize;
                let hi = dict.offsets[code as usize + 1] as usize;
                reconstructed.extend_from_slice(&dict.bytes[lo..hi]);
            }
            assert_eq!(reconstructed, s.as_bytes());
        }
    }

    // ── decode_codes ──────────────────────────────────────────────────────

    #[test]
    fn decode_codes_concatenates_token_bytes() {
        let strings = ["abc"];
        let (dict, store) = compress(&strings, 12, 1);
        let codes: Vec<Token> = row_codes(&store, 0).collect();
        let mut out = Vec::new();
        decode_codes(&dict, &codes, &mut out);
        assert_eq!(out, b"abc");
    }
}
