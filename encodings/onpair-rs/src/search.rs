// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Predicate kernels over a compressed `Column`. Each kernel produces an
// LSB-first packed bitmap (one bit per row, `(n + 7) / 8` bytes long),
// matching the layout that `vortex-onpair-sys::Column::equals_bitmap` and
// friends produce — so downstream consumers can swap implementations
// without changing their bitmap-handling code.
//
// Implementation: decode each row into a reusable buffer and run a
// byte-level predicate. This is a deliberate simplification of the C++
// compressed-domain token-automaton approach (KMP / AC over the token
// stream with per-token byte-level partial matches). The observable result
// is identical; the speed difference matters only at very high row counts,
// at which point downstream callers should use `vortex-onpair`'s
// SIMD-friendly `decode_rows_unchecked` + their own predicate kernels —
// the layout produced here feeds those directly.

use aho_corasick::AhoCorasick;

use crate::decoder::decompress_row;
use crate::dict::Dictionary;
use crate::store::Store;

/// Allocate an empty bitmap that can hold `n_rows` bits packed LSB-first.
#[inline]
pub fn empty_bitmap(n_rows: usize) -> Vec<u8> {
    vec![0u8; n_rows.div_ceil(8)]
}

#[inline]
fn set_bit(bits: &mut [u8], i: usize) {
    bits[i / 8] |= 1u8 << (i % 8);
}

/// Read a single bit from an LSB-packed bitmap.
#[inline]
pub fn get_bit(bits: &[u8], i: usize) -> bool {
    (bits[i / 8] >> (i % 8)) & 1 == 1
}

/// `WHERE col = needle` — one bit per row, LSB-first packing.
pub fn equals_bitmap(dict: &Dictionary, store: &Store, needle: &[u8]) -> Vec<u8> {
    run_predicate(dict, store, |row| row == needle)
}

/// `col LIKE 'needle%'`.
pub fn starts_with_bitmap(dict: &Dictionary, store: &Store, needle: &[u8]) -> Vec<u8> {
    run_predicate(dict, store, |row| row.starts_with(needle))
}

/// `col LIKE '%needle%'`. Uses `memchr::memmem` for single-needle byte search.
pub fn contains_bitmap(dict: &Dictionary, store: &Store, needle: &[u8]) -> Vec<u8> {
    if needle.is_empty() {
        // Every row contains the empty string (vacuously true).
        let n = store.num_strings();
        let mut bits = empty_bitmap(n);
        for i in 0..n {
            set_bit(&mut bits, i);
        }
        return bits;
    }
    let finder = memchr::memmem::Finder::new(needle);
    run_predicate(dict, store, |row| finder.find(row).is_some())
}

/// `col LIKE '%a%' OR '%b%' OR ...` — multi-pattern substring match using
/// Aho-Corasick. One pass per row over the union of `needles`.
///
/// An empty `needles` slice produces an all-zero bitmap.
pub fn multi_pattern_bitmap(
    dict: &Dictionary,
    store: &Store,
    needles: &[&[u8]],
) -> Vec<u8> {
    if needles.is_empty() {
        return empty_bitmap(store.num_strings());
    }
    let ac = AhoCorasick::new(needles).expect("aho-corasick: build automaton");
    run_predicate(dict, store, |row| ac.is_match(row))
}

/// Internal: decode every row into a reusable buffer and call `pred`.
fn run_predicate<F>(dict: &Dictionary, store: &Store, mut pred: F) -> Vec<u8>
where
    F: FnMut(&[u8]) -> bool,
{
    let n = store.num_strings();
    let mut bits = empty_bitmap(n);
    let mut buf = Vec::with_capacity(64);
    for i in 0..n {
        decompress_row(dict, store, i, &mut buf);
        if pred(&buf) {
            set_bit(&mut bits, i);
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FixedThreshold, ThresholdSpec, TrainingConfig};
    use crate::parser::parse;
    use crate::test_corpus::*;
    use crate::trainer::{TrainResult, train};
    use crate::types::BitWidth;

    fn compress<S: AsRef<[u8]>>(strings: &[S], bits: BitWidth) -> (Dictionary, Store) {
        let raw = make_raw(strings);
        let cfg = TrainingConfig {
            bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
        };
        let TrainResult { dict, lpm } = train(&raw.data, &raw.offsets, raw.n, &cfg);
        let mut store = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, bits, &mut store);
        (dict, store)
    }

    fn expected_bitmap<S: AsRef<[u8]>>(strings: &[S], pred: impl Fn(&[u8]) -> bool) -> Vec<u8> {
        let mut bits = empty_bitmap(strings.len());
        for (i, s) in strings.iter().enumerate() {
            if pred(s.as_ref()) {
                set_bit(&mut bits, i);
            }
        }
        bits
    }

    // ── empty_bitmap / set_bit / get_bit ──────────────────────────────────

    #[test]
    fn empty_bitmap_length_is_ceil_div_8() {
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 100] {
            assert_eq!(empty_bitmap(n).len(), n.div_ceil(8), "n={n}");
        }
    }

    #[test]
    fn set_and_get_bit() {
        let mut b = empty_bitmap(20);
        for i in [0usize, 7, 8, 15, 16, 19] {
            set_bit(&mut b, i);
            assert!(get_bit(&b, i), "bit {i}");
        }
        for i in [1, 2, 9, 18] {
            assert!(!get_bit(&b, i), "bit {i} must be clear");
        }
    }

    // ── equals_bitmap ─────────────────────────────────────────────────────

    #[test]
    fn equals_finds_exact_rows() {
        let strings = ["foo", "bar", "foo", "baz", "FOO"];
        let (d, s) = compress(&strings, 12);
        let bits = equals_bitmap(&d, &s, b"foo");
        let truth = expected_bitmap(&strings, |row| row == b"foo");
        assert_eq!(bits, truth);
    }

    #[test]
    fn equals_no_match_returns_all_zeros() {
        let strings = ["a", "b", "c"];
        let (d, s) = compress(&strings, 12);
        let bits = equals_bitmap(&d, &s, b"never");
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn equals_with_empty_needle() {
        let strings: Vec<&[u8]> = vec![b"", b"a", b""];
        let (d, s) = compress(&strings, 12);
        let bits = equals_bitmap(&d, &s, b"");
        assert!(get_bit(&bits, 0));
        assert!(!get_bit(&bits, 1));
        assert!(get_bit(&bits, 2));
    }

    #[test]
    fn equals_across_all_bit_widths() {
        let strings = ["alpha", "beta", "alpha"];
        for bw in 9u8..=16 {
            let (d, s) = compress(&strings, bw);
            let bits = equals_bitmap(&d, &s, b"alpha");
            assert!(get_bit(&bits, 0), "bw={bw}");
            assert!(!get_bit(&bits, 1), "bw={bw}");
            assert!(get_bit(&bits, 2), "bw={bw}");
        }
    }

    // ── starts_with_bitmap ────────────────────────────────────────────────

    #[test]
    fn starts_with_finds_prefix_rows() {
        let strings = ["alpha_one", "beta_two", "alpha_three", "beta_one"];
        let (d, s) = compress(&strings, 12);
        let bits = starts_with_bitmap(&d, &s, b"alpha_");
        let truth = expected_bitmap(&strings, |r| r.starts_with(b"alpha_"));
        assert_eq!(bits, truth);
    }

    #[test]
    fn starts_with_empty_needle_matches_all() {
        let strings = ["x", "y", "z"];
        let (d, s) = compress(&strings, 12);
        let bits = starts_with_bitmap(&d, &s, b"");
        for i in 0..3 {
            assert!(get_bit(&bits, i));
        }
    }

    #[test]
    fn starts_with_no_match_returns_all_zeros() {
        let strings = ["abc", "def"];
        let (d, s) = compress(&strings, 12);
        let bits = starts_with_bitmap(&d, &s, b"zzz");
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn starts_with_full_string_match() {
        let strings = ["abc", "abcd", "ab"];
        let (d, s) = compress(&strings, 12);
        let bits = starts_with_bitmap(&d, &s, b"abc");
        assert!(get_bit(&bits, 0));
        assert!(get_bit(&bits, 1));
        assert!(!get_bit(&bits, 2));
    }

    // ── contains_bitmap ───────────────────────────────────────────────────

    #[test]
    fn contains_finds_substring_rows() {
        let strings = ["xfoox", "foo", "abc", "yfoo"];
        let (d, s) = compress(&strings, 12);
        let bits = contains_bitmap(&d, &s, b"foo");
        let truth = expected_bitmap(&strings, |r| {
            r.windows(3).any(|w| w == b"foo")
        });
        assert_eq!(bits, truth);
    }

    #[test]
    fn contains_empty_needle_matches_all() {
        let strings = ["a", "b", "c"];
        let (d, s) = compress(&strings, 12);
        let bits = contains_bitmap(&d, &s, b"");
        for i in 0..3 {
            assert!(get_bit(&bits, i));
        }
    }

    #[test]
    fn contains_no_match_returns_all_zeros() {
        let strings = ["abc", "def"];
        let (d, s) = compress(&strings, 12);
        let bits = contains_bitmap(&d, &s, b"zzz");
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn contains_needle_at_boundary() {
        let strings = ["foo_other", "other_foo", "foo"];
        let (d, s) = compress(&strings, 12);
        let bits = contains_bitmap(&d, &s, b"foo");
        for i in 0..3 {
            assert!(get_bit(&bits, i), "row {i}");
        }
    }

    // ── multi_pattern_bitmap (Aho-Corasick) ───────────────────────────────

    #[test]
    fn multi_pattern_union_of_substring_matches() {
        let strings = ["admin_001", "guest_999", "user_007", "noop"];
        let (d, s) = compress(&strings, 12);
        let bits = multi_pattern_bitmap(&d, &s, &[b"admin", b"guest"]);
        let truth = expected_bitmap(&strings, |r| {
            r.windows(5).any(|w| w == b"admin" || w == b"guest")
        });
        assert_eq!(bits, truth);
    }

    #[test]
    fn multi_pattern_empty_needles_returns_zeros() {
        let strings = ["a", "b"];
        let (d, s) = compress(&strings, 12);
        let bits = multi_pattern_bitmap(&d, &s, &[]);
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn multi_pattern_overlapping_needles() {
        // "user_admin_001" contains both "user" and "admin".
        let strings = ["user_admin_001", "user_007", "admin_only"];
        let (d, s) = compress(&strings, 12);
        let bits = multi_pattern_bitmap(&d, &s, &[b"user", b"admin"]);
        for i in 0..3 {
            assert!(get_bit(&bits, i), "row {i}");
        }
    }

    #[test]
    fn multi_pattern_single_needle_equals_contains() {
        let strings = ["hello world", "goodbye world", "hello there"];
        let (d, s) = compress(&strings, 12);
        let multi = multi_pattern_bitmap(&d, &s, &[b"hello"]);
        let single = contains_bitmap(&d, &s, b"hello");
        assert_eq!(multi, single);
    }
}
