// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/token_automaton.h`.
//
// Token-level automata that consume the bit-packed token stream of an
// `onpair_lib::Column` directly. The scan loop in [`Column::scan`] resets
// the automaton at each row, feeds every token, and inspects
// `is_accepted()` once after the last token (or after `is_dead()` becomes
// true, whichever is first).
//
// Build composite predicates via [`and`], [`or`], [`not`]; the wrappers
// also implement [`TokenAutomaton`], so they nest. Every concrete
// automaton must implement `step` / `is_accepted` / `reset`; the default
// `is_dead` returns `false`, which is correct for any automaton that
// never finalises before the end of the row.

use crate::types::Token;

/// Token-by-token streaming predicate. Reset once per row, stepped on every
/// token, read for the final verdict.
pub trait TokenAutomaton {
    fn step(&mut self, t: Token);
    fn is_accepted(&self) -> bool;
    fn reset(&mut self);
    /// `true` once the verdict cannot change regardless of remaining
    /// tokens. The scan loop uses this to skip the rest of a row.
    fn is_dead(&self) -> bool {
        false
    }
}

impl<A: TokenAutomaton + ?Sized> TokenAutomaton for &mut A {
    #[inline]
    fn step(&mut self, t: Token) {
        (**self).step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        (**self).is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        (**self).reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        (**self).is_dead()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Combinators — Negated / And / Or.
// ─────────────────────────────────────────────────────────────────────────────

/// `!A` — flips `is_accepted`. `is_dead` is forwarded unchanged.
pub struct Negated<A>(pub A);

impl<A: TokenAutomaton> TokenAutomaton for Negated<A> {
    #[inline]
    fn step(&mut self, t: Token) {
        self.0.step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        !self.0.is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        self.0.is_dead()
    }
}

/// `A AND B` — both must accept. Both step on every token. Early-exits
/// when either inner becomes dead in a state that proves rejection.
pub struct And<A, B>(pub A, pub B);

impl<A: TokenAutomaton, B: TokenAutomaton> TokenAutomaton for And<A, B> {
    #[inline]
    fn step(&mut self, t: Token) {
        self.0.step(t);
        self.1.step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        self.0.is_accepted() && self.1.is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        (self.0.is_dead() && !self.0.is_accepted())
            || (self.1.is_dead() && !self.1.is_accepted())
    }
}

/// `A OR B` — either may accept. Both step on every token. Early-exits
/// when either inner becomes dead in a state that proves acceptance.
pub struct Or<A, B>(pub A, pub B);

impl<A: TokenAutomaton, B: TokenAutomaton> TokenAutomaton for Or<A, B> {
    #[inline]
    fn step(&mut self, t: Token) {
        self.0.step(t);
        self.1.step(t);
    }
    #[inline]
    fn is_accepted(&self) -> bool {
        self.0.is_accepted() || self.1.is_accepted()
    }
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }
    #[inline]
    fn is_dead(&self) -> bool {
        (self.0.is_dead() && self.0.is_accepted())
            || (self.1.is_dead() && self.1.is_accepted())
    }
}

/// `not(a)` constructs a [`Negated`] wrapper.
pub fn not<A: TokenAutomaton>(a: A) -> Negated<A> {
    Negated(a)
}

/// `and(a, b)` constructs an [`And`] wrapper.
pub fn and<A: TokenAutomaton, B: TokenAutomaton>(a: A, b: B) -> And<A, B> {
    And(a, b)
}

/// `or(a, b)` constructs an [`Or`] wrapper.
pub fn or<A: TokenAutomaton, B: TokenAutomaton>(a: A, b: B) -> Or<A, B> {
    Or(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny test automaton that accepts a fixed token-id once seen.
    struct AcceptsToken {
        target: Token,
        seen: bool,
    }
    impl AcceptsToken {
        fn new(t: Token) -> Self {
            Self { target: t, seen: false }
        }
    }
    impl TokenAutomaton for AcceptsToken {
        fn step(&mut self, t: Token) {
            if t == self.target {
                self.seen = true;
            }
        }
        fn is_accepted(&self) -> bool {
            self.seen
        }
        fn reset(&mut self) {
            self.seen = false;
        }
        fn is_dead(&self) -> bool {
            self.seen
        }
    }

    fn drive<A: TokenAutomaton>(mut a: A, tokens: &[Token]) -> bool {
        a.reset();
        for &t in tokens {
            a.step(t);
            if a.is_dead() {
                break;
            }
        }
        a.is_accepted()
    }

    #[test]
    fn accepts_token_basic() {
        assert!(drive(AcceptsToken::new(7), &[1, 2, 3, 7, 9]));
        assert!(!drive(AcceptsToken::new(7), &[1, 2, 3, 8]));
    }

    #[test]
    fn negation_inverts() {
        assert!(!drive(not(AcceptsToken::new(7)), &[1, 7, 2]));
        assert!(drive(not(AcceptsToken::new(7)), &[1, 8, 2]));
    }

    #[test]
    fn and_requires_both() {
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(and(a, b), &[1, 2, 3]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(!drive(and(a, b), &[1, 3]));
    }

    #[test]
    fn or_requires_either() {
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(or(a, b), &[1, 9]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(or(a, b), &[9, 2]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(!drive(or(a, b), &[3, 4, 5]));
    }

    #[test]
    fn nested_and_not() {
        // A AND NOT B
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(drive(and(a, not(b)), &[1, 3]));
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        assert!(!drive(and(a, not(b)), &[1, 2, 3]));
    }

    #[test]
    fn references_implement_trait() {
        let mut a = AcceptsToken::new(7);
        let result = drive(&mut a, &[1, 7]);
        assert!(result);
        // Inner state remains accepting after.
        assert!(a.is_accepted());
    }

    #[test]
    fn or_dead_when_accepted() {
        // Once an Or component accepts and is dead, the combinator is dead.
        let a = AcceptsToken::new(1);
        let b = AcceptsToken::new(2);
        let mut comb = or(a, b);
        comb.reset();
        comb.step(1);
        assert!(comb.is_dead());
        assert!(comb.is_accepted());
    }
}
// Port of `include/onpair/search/automata/eq_automaton.h`.
//
// Token-level automaton for SQL `col = value`. Tokenises the query once
// against the column's dictionary, then a step is a single bounds check +
// `u16` compare. `is_dead()` becomes true the moment a token diverges from
// the query — the scan loop skips the rest of the row.

use crate::dict::Dictionary;
use crate::tokenize::tokenize;

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
mod tests2 {
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
        let eq = EqAutomaton::new(b"def", &col.dictionary().clone());
        assert_eq!(col.scan(eq), vec![1]);
    }

    #[test]
    fn no_match() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"xyz", &col.dictionary().clone());
        assert!(col.scan(eq).is_empty());
    }

    #[test]
    fn multiple_identical_strings() {
        let data = ["abc", "abc", "def", "abc"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abc", &col.dictionary().clone());
        assert_eq!(col.scan(eq), vec![0, 1, 3]);
    }

    #[test]
    fn empty_value_matches_only_empty_strings() {
        let data: Vec<&[u8]> = vec![b"", b"abc", b"", b"def", b""];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"", &col.dictionary().clone());
        assert_eq!(col.scan(eq), vec![0, 2, 4]);
    }

    #[test]
    fn empty_value_no_empty_strings() {
        let data = ["abc", "def"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"", &col.dictionary().clone());
        assert!(col.scan(eq).is_empty());
    }

    #[test]
    fn prefix_of_value_does_not_match() {
        let data = ["abc", "abcd", "abcde"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abc", &col.dictionary().clone());
        assert_eq!(col.scan(eq), vec![0]);
    }

    #[test]
    fn suffix_of_value_does_not_match() {
        let data = ["bc", "abc", "xabc"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abc", &col.dictionary().clone());
        assert_eq!(col.scan(eq), vec![1]);
    }

    #[test]
    fn value_longer_than_all_strings() {
        let data = ["a", "b", "c"];
        let col = make_column(&data);
        let eq = EqAutomaton::new(b"abcdefgh", &col.dictionary().clone());
        assert!(col.scan(eq).is_empty());
    }

    // ── Rescannable ───────────────────────────────────────────────────────

    #[test]
    fn rescannable_same_column() {
        let data = ["abc", "def", "abc"];
        let col = make_column(&data);
        let mut eq = EqAutomaton::new(b"abc", &col.dictionary().clone());
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
            let eq = EqAutomaton::new(b"abc", &col.dictionary().clone());
            assert_eq!(col.scan(eq), vec![0, 2], "bw={bw}");
        }
    }

    // ── Large corpus cross-validation ─────────────────────────────────────

    #[test]
    fn large_corpus_cross_validation() {
        let data = random_ascii_strings(200, 30, 123);
        let col = make_column(&data);
        let dict = col.dictionary().clone();
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
        let eq = EqAutomaton::new(b"abc", &col.dictionary().clone());
        assert!(col.scan(eq).is_empty());
    }

    // ── Cross-validation against brute force on user strings ──────────────

    #[test]
    fn consistency_with_brute_force() {
        let data = user_strings(50);
        let col = make_column(&data);
        let dict = col.dictionary().clone();
        let q = "https://www.example.com/page";
        let eq = EqAutomaton::new(q.as_bytes(), &dict);
        assert_eq!(col.scan(eq), brute_eq(&data, q.as_bytes()));
        let eq2 = EqAutomaton::new(b"missing-needle", &dict);
        assert!(col.scan(eq2).is_empty());
    }
}
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

use crate::types::TokenRange;

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
mod tests3 {
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
        let pa = PrefixAutomaton::new(b"user_", &col.dictionary().clone());
        assert_eq!(col.scan(pa), vec![0, 1, 3]);
    }

    #[test]
    fn admin_prefix() {
        let data = ["user_000001", "admin_001", "admin_002", "guest_001"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"admin", &col.dictionary().clone());
        assert_eq!(col.scan(pa), vec![1, 2]);
    }

    #[test]
    fn no_matches() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"xyz", &col.dictionary().clone());
        assert!(col.scan(pa).is_empty());
    }

    #[test]
    fn exact_match() {
        let data = ["abc", "abcd", "abcde"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"abc", &col.dictionary().clone());
        assert_eq!(col.scan(pa).len(), 3);
    }

    #[test]
    fn prefix_longer_than_string() {
        let data = ["ab", "abc", "abcd"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"abcde", &col.dictionary().clone());
        assert!(col.scan(pa).is_empty());
    }

    #[test]
    fn single_char_prefix() {
        let data = ["abc", "axe", "bcd", "apple"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"a", &col.dictionary().clone());
        assert_eq!(col.scan(pa).len(), 3);
    }

    // ── Empty prefix ──────────────────────────────────────────────────────

    #[test]
    fn empty_prefix_matches_all() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"", &col.dictionary().clone());
        assert_eq!(col.scan(pa).len(), 3);
    }

    // ── User strings ──────────────────────────────────────────────────────

    #[test]
    fn user_strings_prefix() {
        let data = user_strings(50);
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"https://", &col.dictionary().clone());
        // user_strings corpus rotates through six bases; five start with https://.
        let expected = brute_prefix(&data, b"https://");
        assert_eq!(col.scan(pa), expected);
    }

    // ── Rescannable ───────────────────────────────────────────────────────

    #[test]
    fn rescannable_same_column() {
        let data = ["user_001", "admin_001", "user_002"];
        let col = make_column(&data);
        let mut pa = PrefixAutomaton::new(b"user_", &col.dictionary().clone());
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
            let pa = PrefixAutomaton::new(b"user_", &col.dictionary().clone());
            assert_eq!(col.scan(pa).len(), 3, "bw={bw}");
        }
    }

    // ── Divergence-inside-token ───────────────────────────────────────────

    #[test]
    fn prefix_boundary_within_token() {
        let data = ["user_001", "useful", "umbrella"];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"use", &col.dictionary().clone());
        assert_eq!(col.scan(pa).len(), 2);
    }

    // ── Cross-validation ──────────────────────────────────────────────────

    #[test]
    fn consistency_with_brute_force() {
        let data = random_ascii_strings(200, 30, 123);
        let col = make_column(&data);
        for prefix in [b"a" as &[u8], b"ab", b"z", b"xx"] {
            let pa = PrefixAutomaton::new(prefix, &col.dictionary().clone());
            assert_eq!(col.scan(pa), brute_prefix(&data, prefix), "prefix={prefix:?}");
        }
    }

    // ── Empty column ──────────────────────────────────────────────────────

    #[test]
    fn empty_column_returns_empty() {
        let strings: Vec<&[u8]> = vec![];
        let raw = make_raw(&strings);
        let col = Column::compress(&raw.data, &raw.offsets_u64, DEFAULT_DICT12_CONFIG).unwrap();
        let pa = PrefixAutomaton::new(b"abc", &col.dictionary().clone());
        assert!(col.scan(pa).is_empty());
    }

    #[test]
    fn empty_string_matches_empty_prefix() {
        let data: Vec<&[u8]> = vec![b"", b"abc", b""];
        let col = make_column(&data);
        let pa = PrefixAutomaton::new(b"", &col.dictionary().clone());
        assert_eq!(col.scan(pa).len(), 3);
    }
}
