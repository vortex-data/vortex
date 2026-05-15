// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/automata/kmp_automaton.h`.
//
// Token-level KMP automaton for SQL `col LIKE '%pattern%'`.
//
// Construction (mirrors the C++ original):
//   1. Byte-level KMP failure table over the pattern.
//   2. Base pass — for every dictionary token `t`, evolve KMP from state 0
//      through `t`'s bytes and record the exit state in `base[t]`.
//   3. Sparse pass — for every non-zero entry state `j`, find dictionary
//      tokens whose exit state from `j` differs from `base[t]`. These are
//      stored as sorted-by-token sparse range tables grouped per entry
//      state.
//
// `step(t)`:
//   * if `state > 0`, scan the sparse ranges for the current state. If `t`
//     lies inside one of them, jump to its target state.
//   * otherwise the transition is `base[t]`.
//
// `is_dead()` becomes true once `state == pattern_length` (a match has been
// observed). Pattern length is capped at 255 bytes because states are
// stored as `u8`.

use crate::automaton::TokenAutomaton;
use crate::dict::Dictionary;
use crate::types::{Token, TokenRange};

type State = u8;

#[derive(Clone, Copy, Debug)]
struct SparseTransition {
    range: TokenRange,
    target: State,
}

pub struct KmpAutomaton {
    match_state: State,
    state: State,
    /// `base[token] = KMP exit state after consuming token's bytes from state 0`.
    base: Vec<State>,
    /// Flattened sparse transitions: transitions for entry state `s` live
    /// at `sparse[offsets[s]..offsets[s+1]]`.
    sparse: Vec<SparseTransition>,
    offsets: Vec<u16>,
}

impl KmpAutomaton {
    pub fn new(pattern: &[u8], dict: &Dictionary) -> Self {
        let m = pattern.len();
        assert!(m <= u8::MAX as usize, "KmpAutomaton pattern must be at most 255 bytes");
        let num_tokens = dict.num_tokens();

        if m == 0 {
            return Self {
                match_state: 0,
                state: 0,
                base: vec![0; num_tokens],
                sparse: Vec::new(),
                offsets: vec![0u16; 2],
            };
        }
        let match_state = m as State;

        // ── 1. KMP failure table ───────────────────────────────────────────
        let fail = build_failure(pattern);

        // Step the KMP DFA over a byte string starting from `s`. Once we
        // reach `m` (match), subsequent bytes are absorbed and we stay at m.
        let step_bytes = |mut s: State, data: &[u8]| -> State {
            for &c in data {
                if s as usize == m {
                    return match_state;
                }
                while s > 0 && pattern[s as usize] != c {
                    s = fail[(s - 1) as usize];
                }
                if pattern[s as usize] == c {
                    s += 1;
                }
            }
            s
        };

        // ── 2. Base pass ──────────────────────────────────────────────────
        let mut base = vec![0u8; num_tokens];
        let p0 = pattern[0];
        for t in 0..num_tokens {
            let tok = dict.data(t as Token);
            if !tok.contains(&p0) {
                base[t] = 0;
                continue;
            }
            base[t] = step_bytes(0, tok);
        }

        // ── 3. Sparse pass — dual-KMP trie traversal ───────────────────────
        let mut sparse: Vec<SparseTransition> = Vec::new();
        let mut offsets = vec![0u16; m + 1];

        let mut work = SparseBuilder { dict, base: &base, sparse: &mut sparse, range_start: 0 };

        let mut relevant_chars: Vec<u8> = Vec::with_capacity(m);

        for j in 1..(m as State) {
            work.range_start = work.sparse.len();
            offsets[j as usize] = work.range_start as u16;

            // Bytes that could cause a different KMP transition from j vs
            // from 0: exactly the pattern bytes along the failure chain
            // j → fail[j-1] → ... → 0.
            relevant_chars.clear();
            let mut s = j;
            while s > 0 {
                relevant_chars.push(pattern[s as usize]);
                s = fail[(s - 1) as usize];
            }
            relevant_chars.sort_unstable();
            relevant_chars.dedup();

            for &byte in &relevant_chars {
                let range = dict.prefix_range(&[byte]);
                if range.empty() {
                    continue;
                }
                let kj = step_bytes(j, &[byte]);
                let k0 = step_bytes(0, &[byte]);
                work.traverse(range, 1, kj, k0, match_state, &step_bytes);
            }
        }

        offsets[m] = sparse.len() as u16;

        Self { match_state, state: 0, base, sparse, offsets }
    }

    /// Pattern length (== matching state, capped at 255).
    pub fn pattern_length(&self) -> usize {
        self.match_state as usize
    }

    /// Total number of sparse transitions emitted at construction.
    pub fn sparse_range_count(&self) -> usize {
        self.sparse.len()
    }
}

impl TokenAutomaton for KmpAutomaton {
    #[inline]
    fn step(&mut self, t: Token) {
        if self.state == self.match_state {
            return;
        }
        if self.state > 0 {
            let lo = self.offsets[self.state as usize] as usize;
            let hi = self.offsets[(self.state as usize) + 1] as usize;
            // Sparse table is sorted by range.begin and ranges don't overlap.
            for r in &self.sparse[lo..hi] {
                if t < r.range.begin {
                    break;
                }
                if t <= r.range.last {
                    self.state = r.target;
                    return;
                }
            }
        }
        self.state = self.base[t as usize];
    }

    #[inline]
    fn is_accepted(&self) -> bool {
        self.state == self.match_state
    }

    #[inline]
    fn reset(&mut self) {
        self.state = 0;
    }

    #[inline]
    fn is_dead(&self) -> bool {
        self.state == self.match_state
    }
}

/// Internal builder split out so we can recurse while holding a `&mut
/// Vec<SparseTransition>`. The closure-based recursion in the C++ original
/// translates poorly to Rust's borrow checker; an explicit struct works
/// fine.
struct SparseBuilder<'a> {
    dict: &'a Dictionary,
    base: &'a [State],
    sparse: &'a mut Vec<SparseTransition>,
    /// Index in `sparse` where the current entry state's ranges began.
    /// Used to merge adjacent same-target ranges within one state group.
    range_start: usize,
}

impl SparseBuilder<'_> {
    fn emit(&mut self, range: TokenRange, target: State) {
        if self.sparse.len() > self.range_start {
            let last = self.sparse.last_mut().unwrap();
            if last.target == target && (last.range.last as u32) + 1 == range.begin as u32 {
                last.range.last = range.last;
                return;
            }
        }
        self.sparse.push(SparseTransition { range, target });
    }

    fn traverse<F>(
        &mut self,
        tr: TokenRange,
        depth: usize,
        kmp_j: State,
        kmp_0: State,
        match_state: State,
        step_bytes: &F,
    ) where
        F: Fn(State, &[u8]) -> State,
    {
        if kmp_j == kmp_0 || tr.empty() {
            return;
        }

        // Full match from kmp_j: override tokens whose base != match_state.
        if kmp_j == match_state {
            let mut i = tr.begin;
            while i <= tr.last {
                if self.base[i as usize] != match_state {
                    let start = i;
                    while i <= tr.last && self.base[i as usize] != match_state {
                        i = i.wrapping_add(1);
                        if i == 0 {
                            // overflow guard for u16
                            break;
                        }
                    }
                    self.emit(
                        TokenRange { begin: start, last: i.wrapping_sub(1) },
                        match_state,
                    );
                } else {
                    if i == tr.last {
                        break;
                    }
                    i = i.wrapping_add(1);
                }
            }
            return;
        }

        // Leaf tokens (token length == depth) all share exit state kmp_j.
        let mut cur = tr.begin;
        while cur <= tr.last && self.dict.token_size(cur) == depth {
            if cur == tr.last {
                self.emit(TokenRange { begin: tr.begin, last: cur }, kmp_j);
                return;
            }
            cur = cur.wrapping_add(1);
        }
        if cur > tr.begin {
            self.emit(TokenRange { begin: tr.begin, last: cur - 1 }, kmp_j);
        }
        if cur > tr.last {
            return;
        }

        // Recurse into subtrees partitioned by byte at `depth`.
        while cur <= tr.last {
            let c = self.dict.data(cur)[depth];
            let mut sub_hi = cur;
            while sub_hi < tr.last && self.dict.data(sub_hi + 1)[depth] == c {
                sub_hi += 1;
            }
            self.traverse(
                TokenRange { begin: cur, last: sub_hi },
                depth + 1,
                step_bytes(kmp_j, &[c]),
                step_bytes(kmp_0, &[c]),
                match_state,
                step_bytes,
            );
            if sub_hi == tr.last {
                break;
            }
            cur = sub_hi + 1;
        }
    }
}

/// Standard KMP failure function over `pattern`.
fn build_failure(pattern: &[u8]) -> Vec<State> {
    let m = pattern.len();
    let mut fail = vec![0u8; m];
    let mut len: State = 0;
    let mut i = 1usize;
    while i < m {
        if pattern[i] == pattern[len as usize] {
            len += 1;
            fail[i] = len;
            i += 1;
        } else if len > 0 {
            len = fail[(len - 1) as usize];
        } else {
            fail[i] = 0;
            i += 1;
        }
    }
    fail
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

    fn brute_contains<S: AsRef<[u8]>>(strings: &[S], needle: &[u8]) -> Vec<usize> {
        if needle.is_empty() {
            return (0..strings.len()).collect();
        }
        strings
            .iter()
            .enumerate()
            .filter(|(_, s)| s.as_ref().windows(needle.len()).any(|w| w == needle))
            .map(|(i, _)| i)
            .collect()
    }

    // ── Empty pattern ─────────────────────────────────────────────────────

    #[test]
    fn empty_pattern_matches_all() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"", &col.dictionary().clone());
        assert_eq!(col.scan(kmp).len(), 3);
    }

    // ── Basic substring search ────────────────────────────────────────────

    #[test]
    fn basic_substring_match() {
        let data = ["hello world", "foo bar", "hello there", "world hello", "xyz"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"hello", &col.dictionary().clone());
        assert_eq!(col.scan(kmp), vec![0, 2, 3]);
    }

    #[test]
    fn pattern_at_beginning() {
        let data = ["abc_def", "xyz_abc", "abc"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"abc", &col.dictionary().clone());
        assert_eq!(col.scan(kmp).len(), 3);
    }

    #[test]
    fn pattern_at_end() {
        let data = ["hello_xyz", "abc_xyz", "no_match"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"xyz", &col.dictionary().clone());
        assert_eq!(col.scan(kmp), vec![0, 1]);
    }

    #[test]
    fn no_matches() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"xyz", &col.dictionary().clone());
        assert!(col.scan(kmp).is_empty());
    }

    #[test]
    fn exact_string_match() {
        let data = ["abc", "abcd", "ab"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"abc", &col.dictionary().clone());
        assert_eq!(col.scan(kmp), vec![0, 1]);
    }

    #[test]
    fn single_char_pattern() {
        let data = ["abc", "def", "axe"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"a", &col.dictionary().clone());
        assert_eq!(col.scan(kmp), vec![0, 2]);
    }

    // ── KMP failure-function stress ───────────────────────────────────────

    #[test]
    fn overlapping_pattern_in_string() {
        let data = ["aaaa", "ab", "ba"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"aa", &col.dictionary().clone());
        assert_eq!(col.scan(kmp), vec![0]);
    }

    #[test]
    fn kmp_failure_function_stress() {
        // Pattern "abab" has LPS [0,0,1,2].
        let data = ["ababab", "abab", "abba", "baba"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"abab", &col.dictionary().clone());
        assert_eq!(col.scan(kmp), vec![0, 1]);
    }

    // ── Cross-validation against brute force ──────────────────────────────

    #[test]
    fn cross_validation_with_brute_force() {
        let data = random_ascii_strings(100, 30, 42);
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"ab", &col.dictionary().clone());
        assert_eq!(col.scan(kmp), brute_contains(&data, b"ab"));
    }

    #[test]
    fn cross_validation_url_corpus() {
        let data = user_strings(60);
        let col = make_column(&data);
        for needle in [&b"example"[..], b"https", b"docs", b"missing", b"://"] {
            let kmp = KmpAutomaton::new(needle, &col.dictionary().clone());
            assert_eq!(col.scan(kmp), brute_contains(&data, needle), "needle={needle:?}");
        }
    }

    // ── All bit widths ────────────────────────────────────────────────────

    #[test]
    fn works_across_bit_widths() {
        let data = ["the quick brown fox", "lazy dog", "quick fox"];
        for bw in 9u32..=16 {
            let col = make_column_bits(&data, bw);
            let kmp = KmpAutomaton::new(b"quick", &col.dictionary().clone());
            assert_eq!(col.scan(kmp), vec![0, 2], "bw={bw}");
        }
    }

    // ── Pattern longer than any string ────────────────────────────────────

    #[test]
    fn pattern_longer_than_strings() {
        let data = ["ab", "cd"];
        let col = make_column(&data);
        let kmp = KmpAutomaton::new(b"abcdefghij", &col.dictionary().clone());
        assert!(col.scan(kmp).is_empty());
    }

    // ── Empty column ──────────────────────────────────────────────────────

    #[test]
    fn empty_column_returns_empty() {
        let strings: Vec<&[u8]> = vec![];
        let raw = make_raw(&strings);
        let col = Column::compress(&raw.data, &raw.offsets_u64, DEFAULT_DICT12_CONFIG).unwrap();
        let kmp = KmpAutomaton::new(b"abc", &col.dictionary().clone());
        assert!(col.scan(kmp).is_empty());
    }

    // ── Rescannable ───────────────────────────────────────────────────────

    #[test]
    fn rescannable() {
        let data = ["abc", "def", "abc_xyz"];
        let col = make_column(&data);
        let mut kmp = KmpAutomaton::new(b"abc", &col.dictionary().clone());
        let r1 = col.scan(&mut kmp);
        let r2 = col.scan(&mut kmp);
        assert_eq!(r1, r2);
    }

    // ── Equivalence with byte-level contains_bitmap ───────────────────────

    #[test]
    fn equivalent_to_contains_bitmap() {
        let data = user_strings(80);
        let col = make_column(&data);
        for needle in [&b"example"[..], b"https", b"docs"] {
            let kmp = KmpAutomaton::new(needle, &col.dictionary().clone());
            let token_result = col.scan(kmp);
            let bitmap = col.contains_bitmap(needle);
            let bitmap_result: Vec<usize> = (0..data.len())
                .filter(|&i| (bitmap[i / 8] >> (i % 8)) & 1 == 1)
                .collect();
            assert_eq!(token_result, bitmap_result, "needle={needle:?}");
        }
    }
}
