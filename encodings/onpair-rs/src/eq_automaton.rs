// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/eq_automaton.h`.
//
// Token-level automaton for SQL `col = value`. Tokenises the query once
// against the column's dictionary, then a step is a single bounds check +
// `u16` compare. `is_dead()` becomes true the moment a token diverges from
// the query — the scan loop skips the rest of the row.

use crate::automaton::TokenAutomaton;
use crate::dict::Dictionary;
use crate::tokenize::tokenize;
use crate::types::Token;

pub struct EqAutomaton {
    query: Vec<Token>,
    pos: usize,
    failed: bool,
}

impl EqAutomaton {
    /// Build the automaton against `dict`. Empty `value` matches only rows
    /// with zero tokens (empty strings).
    pub fn new(value: &[u8], dict: &Dictionary) -> Self {
        Self {
            query: tokenize(value, dict),
            pos: 0,
            failed: false,
        }
    }

    /// Number of tokens the query produced.
    pub fn query_length(&self) -> usize {
        self.query.len()
    }
}

impl TokenAutomaton for EqAutomaton {
    #[inline]
    fn step(&mut self, t: Token) {
        // failed |= (pos >= len) || (t != query[pos])
        self.failed |= self.pos >= self.query.len() || t != self.query[self.pos];
        self.pos += 1;
    }

    #[inline]
    fn is_accepted(&self) -> bool {
        !self.failed && self.pos == self.query.len()
    }

    #[inline]
    fn reset(&mut self) {
        self.pos = 0;
        self.failed = false;
    }

    #[inline]
    fn is_dead(&self) -> bool {
        self.failed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::config::{DEFAULT_DICT12_CONFIG, OnPairTrainingConfig};
    use crate::test_corpus::{make_raw, random_ascii_strings, user_strings};

    fn make_column<S: AsRef<[u8]>>(strings: &[S]) -> Column {
        make_column_bits(strings, 14)
    }

    fn make_column_bits<S: AsRef<[u8]>>(strings: &[S], bits: u32) -> Column {
        let raw = make_raw(strings);
        let cfg = OnPairTrainingConfig { bits, threshold: 0.5, seed: 42 };
        Column::compress(&raw.data, &raw.offsets_u64, cfg).unwrap()
    }

    fn brute_eq<S: AsRef<[u8]>>(strings: &[S], needle: &[u8]) -> Vec<usize> {
        strings
            .iter()
            .enumerate()
            .filter(|(_, s)| s.as_ref() == needle)
            .map(|(i, _)| i)
            .collect()
    }

    // ── Basic correctness ─────────────────────────────────────────────────

    #[test]
    fn single_match() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"def", &col.dict_for_test());
        assert_eq!(col.scan(eq), vec![1]);
    }

    #[test]
    fn no_match() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"xyz", &col.dict_for_test());
        assert!(col.scan(eq).is_empty());
    }

    #[test]
    fn multiple_identical_strings() {
        let data = ["abc", "abc", "def", "abc"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abc", &col.dict_for_test());
        assert_eq!(col.scan(eq), vec![0, 1, 3]);
    }

    #[test]
    fn empty_value_matches_only_empty_strings() {
        let data: Vec<&[u8]> = vec![b"", b"abc", b"", b"def", b""];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"", &col.dict_for_test());
        assert_eq!(col.scan(eq), vec![0, 2, 4]);
    }

    #[test]
    fn empty_value_no_empty_strings() {
        let data = ["abc", "def"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"", &col.dict_for_test());
        assert!(col.scan(eq).is_empty());
    }

    #[test]
    fn prefix_of_value_does_not_match() {
        let data = ["abc", "abcd", "abcde"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abc", &col.dict_for_test());
        assert_eq!(col.scan(eq), vec![0]);
    }

    #[test]
    fn suffix_of_value_does_not_match() {
        let data = ["bc", "abc", "xabc"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abc", &col.dict_for_test());
        assert_eq!(col.scan(eq), vec![1]);
    }

    #[test]
    fn value_longer_than_all_strings() {
        let data = ["a", "b", "c"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abcdefgh", &col.dict_for_test());
        assert!(col.scan(eq).is_empty());
    }

    // ── Rescannable ───────────────────────────────────────────────────────

    #[test]
    fn rescannable_same_column() {
        let data = ["abc", "def", "abc"];
        let col = make_column(&data);
        let mut eq = EqAutomaton::new(b"abc", &col.dict_for_test());
        let r1 = col.scan(&mut eq);
        let r2 = col.scan(&mut eq);
        assert_eq!(r1, r2);
    }

    // ── All bit widths ────────────────────────────────────────────────────

    #[test]
    fn works_across_bit_widths() {
        let data = ["abc", "def", "abc", "ghi"];
        for bw in 9u32..=16 {
            let col = make_column_bits(&data, bw);
            let eq = EqAutomaton::new(b"abc", &col.dict_for_test());
            assert_eq!(col.scan(eq), vec![0, 2], "bw={bw}");
        }
    }

    // ── Large corpus cross-validation ─────────────────────────────────────

    #[test]
    fn large_corpus_cross_validation() {
        let data = random_ascii_strings(200, 30, 123);
        let col = make_column(&data);
        let dict = col.dict_for_test();
        for qi in (0..data.len()).step_by(40) {
            let q = &data[qi];
            let eq = EqAutomaton::new(q, &dict);
            assert_eq!(col.scan(eq), brute_eq(&data, q));
        }
    }

    // ── Empty column ──────────────────────────────────────────────────────

    #[test]
    fn empty_column_returns_empty() {
        let strings: Vec<&[u8]> = vec![];
        let raw = make_raw(&strings);
        let col = Column::compress(&raw.data, &raw.offsets_u64, DEFAULT_DICT12_CONFIG).unwrap();
        let eq = EqAutomaton::new(b"abc", &col.dict_for_test());
        assert!(col.scan(eq).is_empty());
    }

    // ── Cross-validation against brute force on user strings ──────────────

    #[test]
    fn consistency_with_brute_force() {
        let data = user_strings(50);
        let col = make_column(&data);
        let dict = col.dict_for_test();
        let q = "https://www.example.com/page";
        let eq = EqAutomaton::new(q.as_bytes(), &dict);
        assert_eq!(col.scan(eq), brute_eq(&data, q.as_bytes()));
        let eq2 = EqAutomaton::new(b"missing-needle", &dict);
        assert!(col.scan(eq2).is_empty());
    }
}
