// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/prefix_automaton.h`.
//
// Token-level automaton for SQL `col LIKE 'prefix%'`.
//
// At each query position `i` we record two things:
//   query[i]    — the expected next token
//   range[i]    — the dictionary range of tokens whose bytes begin with the
//                 remaining suffix of the prefix at that position
//
// On `step(t)`:
//   * if `t == query[i]`, advance.
//   * else if `t` is inside `range[i]`, the row's i-th token diverges from
//     the query but still extends the prefix legally — accept.
//   * else reject.
//
// `is_dead` becomes true the moment we accept or reject; the row never has
// to be decompressed.

use crate::automaton::TokenAutomaton;
use crate::dict::Dictionary;
use crate::tokenize::tokenize;
use crate::types::{Token, TokenRange};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Matching,
    Accepted,
    Rejected,
}

pub struct PrefixAutomaton {
    query: Vec<Token>,
    intervals: Vec<TokenRange>,
    pos: usize,
    status: Status,
}

impl PrefixAutomaton {
    pub fn new(prefix: &[u8], dict: &Dictionary) -> Self {
        let query = tokenize(prefix, dict);
        let q_len = query.len();
        let mut intervals = vec![TokenRange::default(); q_len];

        if q_len == 0 {
            return Self { query, intervals, pos: 0, status: Status::Accepted };
        }

        // For each token position, precompute the dictionary prefix range
        // that any *divergent* token would have to lie in. Walk the prefix
        // string and at each step ask "which tokens have these remaining
        // bytes as a prefix of their bytes?".
        let mut cur = 0usize;
        for (i, &tok) in query.iter().enumerate() {
            intervals[i] = dict.prefix_range(&prefix[cur..]);
            cur += dict.token_size(tok);
        }

        Self { query, intervals, pos: 0, status: Status::Matching }
    }

    pub fn query_length(&self) -> usize {
        self.query.len()
    }
}

impl TokenAutomaton for PrefixAutomaton {
    #[inline]
    fn step(&mut self, t: Token) {
        if self.status != Status::Matching {
            return;
        }
        if t != self.query[self.pos] {
            self.status = if self.intervals[self.pos].contains(t) {
                Status::Accepted
            } else {
                Status::Rejected
            };
            return;
        }
        self.pos += 1;
        if self.pos == self.query.len() {
            self.status = Status::Accepted;
        }
    }

    #[inline]
    fn is_accepted(&self) -> bool {
        self.status == Status::Accepted
    }

    #[inline]
    fn reset(&mut self) {
        self.pos = 0;
        self.status = if self.query.is_empty() {
            Status::Accepted
        } else {
            Status::Matching
        };
    }

    #[inline]
    fn is_dead(&self) -> bool {
        self.status != Status::Matching
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

    fn brute_prefix<S: AsRef<[u8]>>(strings: &[S], prefix: &[u8]) -> Vec<usize> {
        strings
            .iter()
            .enumerate()
            .filter(|(_, s)| s.as_ref().starts_with(prefix))
            .map(|(i, _)| i)
            .collect()
    }

    // ── Basic ─────────────────────────────────────────────────────────────

    #[test]
    fn basic_prefix_match() {
        let data = [
            "user_000001",
            "user_000002",
            "admin_001",
            "user_000003",
            "guest_001",
            "admin_002",
        ];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"user_", &col.dict_for_test());
        assert_eq!(col.scan(pa), vec![0, 1, 3]);
    }

    #[test]
    fn admin_prefix() {
        let data = ["user_000001", "admin_001", "admin_002", "guest_001"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"admin", &col.dict_for_test());
        assert_eq!(col.scan(pa), vec![1, 2]);
    }

    #[test]
    fn no_matches() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"xyz", &col.dict_for_test());
        assert!(col.scan(pa).is_empty());
    }

    #[test]
    fn exact_match() {
        let data = ["abc", "abcd", "abcde"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"abc", &col.dict_for_test());
        assert_eq!(col.scan(pa).len(), 3);
    }

    #[test]
    fn prefix_longer_than_string() {
        let data = ["ab", "abc", "abcd"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"abcde", &col.dict_for_test());
        assert!(col.scan(pa).is_empty());
    }

    #[test]
    fn single_char_prefix() {
        let data = ["abc", "axe", "bcd", "apple"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"a", &col.dict_for_test());
        assert_eq!(col.scan(pa).len(), 3);
    }

    // ── Empty prefix ──────────────────────────────────────────────────────

    #[test]
    fn empty_prefix_matches_all() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"", &col.dict_for_test());
        assert_eq!(col.scan(pa).len(), 3);
    }

    // ── User strings ──────────────────────────────────────────────────────

    #[test]
    fn user_strings_prefix() {
        let data = user_strings(50);
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"https://", &col.dict_for_test());
        // user_strings corpus rotates through six bases; five start with https://.
        let expected = brute_prefix(&data, b"https://");
        assert_eq!(col.scan(pa), expected);
    }

    // ── Rescannable ───────────────────────────────────────────────────────

    #[test]
    fn rescannable_same_column() {
        let data = ["user_001", "admin_001", "user_002"];
        let col = make_column(&data);
        let mut pa = PrefixAutomaton::new(b"user_", &col.dict_for_test());
        let r1 = col.scan(&mut pa);
        let r2 = col.scan(&mut pa);
        assert_eq!(r1, r2);
    }

    // ── All bit widths ────────────────────────────────────────────────────

    #[test]
    fn works_across_bit_widths() {
        let data = ["user_001", "admin_001", "user_002", "user_003"];
        for bw in 9u32..=16 {
            let col = make_column_bits(&data, bw);
            let pa = PrefixAutomaton::new(b"user_", &col.dict_for_test());
            assert_eq!(col.scan(pa).len(), 3, "bw={bw}");
        }
    }

    // ── Divergence-inside-token ───────────────────────────────────────────

    #[test]
    fn prefix_boundary_within_token() {
        let data = ["user_001", "useful", "umbrella"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"use", &col.dict_for_test());
        assert_eq!(col.scan(pa).len(), 2);
    }

    // ── Cross-validation ──────────────────────────────────────────────────

    #[test]
    fn consistency_with_brute_force() {
        let data = random_ascii_strings(200, 30, 123);
        let col = make_column(&data);
        for prefix in [b"a" as &[u8], b"ab", b"z", b"xx"] {
            let pa = PrefixAutomaton::new(prefix, &col.dict_for_test());
            assert_eq!(col.scan(pa), brute_prefix(&data, prefix), "prefix={prefix:?}");
        }
    }

    // ── Empty column ──────────────────────────────────────────────────────

    #[test]
    fn empty_column_returns_empty() {
        let strings: Vec<&[u8]> = vec![];
        let raw = make_raw(&strings);
        let col = Column::compress(&raw.data, &raw.offsets_u64, DEFAULT_DICT12_CONFIG).unwrap();
        let pa = PrefixAutomaton::new(b"abc", &col.dict_for_test());
        assert!(col.scan(pa).is_empty());
    }

    #[test]
    fn empty_string_matches_empty_prefix() {
        let data: Vec<&[u8]> = vec![b"", b"abc", b""];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"", &col.dict_for_test());
        assert_eq!(col.scan(pa).len(), 3);
    }
}
