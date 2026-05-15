// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/search/aho_corasick_trie.h` and
// `include/onpair/search/automata/aho_corasick_automaton.h`.
//
// Token-level multi-pattern substring match (SQL `col LIKE '%a%' OR '%b%'
// OR ...`). The byte-level Aho-Corasick trie is built first; an eager
// dual-traversal then projects it onto the dictionary's token alphabet,
// giving the same base/sparse decomposition as `KmpAutomaton`.
//
// `step` is a single sparse-range scan + table lookup. `is_dead` becomes
// true the moment any pattern matches.

use crate::automaton::TokenAutomaton;
use crate::dict::Dictionary;
use crate::types::{Token, TokenRange};

// ─────────────────────────────────────────────────────────────────────────────
// AhoCorasickTrie — byte-level
// ─────────────────────────────────────────────────────────────────────────────

pub type AcState = u16;

const NULL_STATE: AcState = u16::MAX;
const ROOT_STATE: AcState = 0;

pub struct AhoCorasickTrie {
    /// Edge labels concatenated by parent state; sorted within each state's
    /// child group (SoA Arrow layout).
    edge_labels: Vec<u8>,
    edge_targets: Vec<AcState>,
    child_offsets: Vec<u16>,
    /// Failure link per state.
    fail: Vec<AcState>,
    /// Marks state as "matches at least one pattern".
    accepting: Vec<bool>,
    num_states: usize,
    num_patterns: usize,
}

impl AhoCorasickTrie {
    pub fn new(patterns: &[&[u8]]) -> Self {
        let num_patterns = patterns.len();

        // ── 1. Build trie as first-child / next-sibling temporary nodes ────
        struct Node {
            c: u8,
            first_child: AcState,
            next_sibling: AcState,
        }
        let mut nodes: Vec<Node> = vec![Node {
            c: 0,
            first_child: NULL_STATE,
            next_sibling: NULL_STATE,
        }];
        let mut accepting = vec![false];

        for pat in patterns {
            if pat.is_empty() {
                accepting[ROOT_STATE as usize] = true;
                continue;
            }
            let mut cur = ROOT_STATE;
            for &b in *pat {
                // Walk the (sorted) sibling list.
                let mut child = nodes[cur as usize].first_child;
                let mut prev = NULL_STATE;
                while child != NULL_STATE && nodes[child as usize].c < b {
                    prev = child;
                    child = nodes[child as usize].next_sibling;
                }
                if child == NULL_STATE || nodes[child as usize].c != b {
                    let new_node = nodes.len() as AcState;
                    nodes.push(Node { c: b, first_child: NULL_STATE, next_sibling: child });
                    accepting.push(false);
                    if prev == NULL_STATE {
                        nodes[cur as usize].first_child = new_node;
                    } else {
                        nodes[prev as usize].next_sibling = new_node;
                    }
                    child = new_node;
                }
                cur = child;
            }
            accepting[cur as usize] = true;
        }

        let num_states = nodes.len();
        assert!(num_states < NULL_STATE as usize, "AhoCorasickTrie: too many states");

        // ── 2. Compact into SoA ───────────────────────────────────────────
        let mut edge_labels: Vec<u8> = Vec::with_capacity(num_states);
        let mut edge_targets: Vec<AcState> = Vec::with_capacity(num_states);
        let mut child_offsets: Vec<u16> = Vec::with_capacity(num_states + 1);

        for i in 0..num_states {
            child_offsets.push(edge_labels.len() as u16);
            let mut child = nodes[i].first_child;
            while child != NULL_STATE {
                edge_labels.push(nodes[child as usize].c);
                edge_targets.push(child);
                child = nodes[child as usize].next_sibling;
            }
        }
        child_offsets.push(edge_labels.len() as u16);

        // ── 3. Failure links via BFS ──────────────────────────────────────
        let mut fail = vec![ROOT_STATE; num_states];
        let mut bfs: Vec<AcState> = Vec::with_capacity(num_states);

        let root_start = child_offsets[ROOT_STATE as usize] as usize;
        let root_end = child_offsets[ROOT_STATE as usize + 1] as usize;
        for i in root_start..root_end {
            fail[edge_targets[i] as usize] = ROOT_STATE;
            bfs.push(edge_targets[i]);
        }

        let mut trie = Self {
            edge_labels,
            edge_targets,
            child_offsets,
            fail,
            accepting,
            num_states,
            num_patterns,
        };

        let mut qi = 0;
        while qi < bfs.len() {
            let u = bfs[qi];
            qi += 1;
            // Propagate accepting through fail chain.
            if trie.accepting[trie.fail[u as usize] as usize] {
                trie.accepting[u as usize] = true;
            }
            let lo = trie.child_offsets[u as usize] as usize;
            let hi = trie.child_offsets[u as usize + 1] as usize;
            for i in lo..hi {
                let target = trie.edge_targets[i];
                let label = trie.edge_labels[i];
                trie.fail[target as usize] = trie.advance(trie.fail[u as usize], label);
                bfs.push(target);
            }
        }

        trie
    }

    /// Advance state `u` by byte `c`, resolving failure links as needed.
    pub fn advance(&self, mut u: AcState, c: u8) -> AcState {
        loop {
            let lo = self.child_offsets[u as usize] as usize;
            let hi = self.child_offsets[u as usize + 1] as usize;
            // Edges are sorted by label.
            for i in lo..hi {
                let label = self.edge_labels[i];
                if label == c {
                    return self.edge_targets[i];
                }
                if label > c {
                    break;
                }
            }
            if u == ROOT_STATE {
                return ROOT_STATE;
            }
            u = self.fail[u as usize];
        }
    }

    pub fn is_accepting(&self, s: AcState) -> bool {
        self.accepting[s as usize]
    }

    pub fn num_states(&self) -> usize {
        self.num_states
    }

    pub fn num_patterns(&self) -> usize {
        self.num_patterns
    }

    pub fn fail_link(&self, s: AcState) -> AcState {
        self.fail[s as usize]
    }

    pub fn edge_labels(&self, s: AcState) -> &[u8] {
        let lo = self.child_offsets[s as usize] as usize;
        let hi = self.child_offsets[s as usize + 1] as usize;
        &self.edge_labels[lo..hi]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AhoCorasickAutomaton — token-level
// ─────────────────────────────────────────────────────────────────────────────

const HIT_STATE: AcState = NULL_STATE;

pub struct AhoCorasickAutomaton {
    base: Vec<AcState>,
    sparse_offsets: Vec<u32>,
    sparse_ranges: Vec<TokenRange>,
    sparse_targets: Vec<AcState>,
    state: AcState,
    hit: bool,
    all_match: bool,
}

impl AhoCorasickAutomaton {
    /// Convenience: build the trie and project it in one go.
    pub fn new(patterns: &[&[u8]], dict: &Dictionary) -> Self {
        let trie = AhoCorasickTrie::new(patterns);
        Self::from_trie(&trie, dict)
    }

    /// Project an existing trie onto the token alphabet of `dict`.
    pub fn from_trie(trie: &AhoCorasickTrie, dict: &Dictionary) -> Self {
        let all_match = trie.is_accepting(ROOT_STATE);
        let mut me = Self {
            base: Vec::new(),
            sparse_offsets: Vec::new(),
            sparse_ranges: Vec::new(),
            sparse_targets: Vec::new(),
            state: ROOT_STATE,
            hit: all_match,
            all_match,
        };
        if all_match {
            return me;
        }

        let num_states = trie.num_states();
        let num_tokens = dict.num_tokens();

        // Evolve one state through one byte; collapse to HIT if accepting.
        let evolve = |state: AcState, c: u8| -> AcState {
            if state == HIT_STATE {
                return HIT_STATE;
            }
            let next = trie.advance(state, c);
            if trie.is_accepting(next) { HIT_STATE } else { next }
        };

        // ── 1. Base pass ──────────────────────────────────────────────────
        me.base = Vec::with_capacity(num_tokens);
        for t in 0..num_tokens {
            let bytes = dict.data(t as Token);
            let mut s = ROOT_STATE;
            for &c in bytes {
                s = trie.advance(s, c);
                if trie.is_accepting(s) {
                    s = HIT_STATE;
                    break;
                }
            }
            me.base.push(s);
        }

        // ── 2. Sparse pass ────────────────────────────────────────────────
        me.sparse_offsets = vec![0u32; num_states + 1];

        let mut builder = AcSparseBuilder {
            dict,
            base: &me.base,
            ranges: &mut me.sparse_ranges,
            targets: &mut me.sparse_targets,
            range_start: 0,
        };

        let mut relevant_chars: Vec<u8> = Vec::new();

        for j in 1..(num_states as AcState) {
            builder.range_start = builder.ranges.len();
            me.sparse_offsets[j as usize] = builder.range_start as u32;

            // Collect labels along the failure chain.
            relevant_chars.clear();
            let mut u = j;
            while u != ROOT_STATE {
                for &c in trie.edge_labels(u) {
                    relevant_chars.push(c);
                }
                u = trie.fail_link(u);
            }
            relevant_chars.sort_unstable();
            relevant_chars.dedup();

            for &byte in &relevant_chars {
                // Identical transition from j and root → no exception needed.
                if trie.advance(j, byte) == trie.advance(ROOT_STATE, byte) {
                    continue;
                }
                let range = dict.prefix_range(&[byte]);
                if range.empty() {
                    continue;
                }
                builder.traverse(range, 1, evolve(j, byte), evolve(ROOT_STATE, byte), &evolve);
            }
        }
        me.sparse_offsets[num_states] = me.sparse_ranges.len() as u32;

        me
    }
}

impl TokenAutomaton for AhoCorasickAutomaton {
    #[inline]
    fn step(&mut self, t: Token) {
        if self.hit {
            return;
        }
        if self.state != ROOT_STATE {
            let lo = self.sparse_offsets[self.state as usize] as usize;
            let hi = self.sparse_offsets[self.state as usize + 1] as usize;
            for i in lo..hi {
                let r = self.sparse_ranges[i];
                if t < r.begin {
                    break;
                }
                if t <= r.last {
                    let target = self.sparse_targets[i];
                    self.hit = target == HIT_STATE;
                    self.state = target;
                    return;
                }
            }
        }
        let target = self.base[t as usize];
        self.hit = target == HIT_STATE;
        self.state = target;
    }

    #[inline]
    fn is_accepted(&self) -> bool {
        self.hit
    }

    #[inline]
    fn reset(&mut self) {
        self.state = ROOT_STATE;
        self.hit = self.all_match;
    }

    #[inline]
    fn is_dead(&self) -> bool {
        self.hit
    }
}

struct AcSparseBuilder<'a> {
    dict: &'a Dictionary,
    base: &'a [AcState],
    ranges: &'a mut Vec<TokenRange>,
    targets: &'a mut Vec<AcState>,
    range_start: usize,
}

impl AcSparseBuilder<'_> {
    fn emit(&mut self, range: TokenRange, target: AcState) {
        if self.ranges.len() > self.range_start
            && *self.targets.last().unwrap() == target
            && (self.ranges.last().unwrap().last as u32) + 1 == range.begin as u32
        {
            self.ranges.last_mut().unwrap().last = range.last;
            return;
        }
        self.ranges.push(range);
        self.targets.push(target);
    }

    fn traverse<F>(
        &mut self,
        tr: TokenRange,
        depth: usize,
        state_j: AcState,
        state_0: AcState,
        evolve: &F,
    ) where
        F: Fn(AcState, u8) -> AcState,
    {
        if state_j == state_0 || tr.empty() {
            return;
        }
        if state_j == HIT_STATE {
            let mut i = tr.begin;
            while i <= tr.last {
                if self.base[i as usize] != HIT_STATE {
                    let start = i;
                    while i <= tr.last && self.base[i as usize] != HIT_STATE {
                        if i == tr.last {
                            self.emit(TokenRange { begin: start, last: i }, HIT_STATE);
                            return;
                        }
                        i += 1;
                    }
                    self.emit(TokenRange { begin: start, last: i - 1 }, HIT_STATE);
                } else {
                    if i == tr.last {
                        break;
                    }
                    i += 1;
                }
            }
            return;
        }

        // Leaf tokens of length == depth share exit state state_j.
        let mut cur = tr.begin;
        while cur <= tr.last && self.dict.token_size(cur) == depth {
            if cur == tr.last {
                self.emit(TokenRange { begin: tr.begin, last: cur }, state_j);
                return;
            }
            cur += 1;
        }
        if cur > tr.begin {
            self.emit(TokenRange { begin: tr.begin, last: cur - 1 }, state_j);
        }
        if cur > tr.last {
            return;
        }

        while cur <= tr.last {
            let c = self.dict.data(cur)[depth];
            let mut sub_hi = cur;
            while sub_hi < tr.last && self.dict.data(sub_hi + 1)[depth] == c {
                sub_hi += 1;
            }
            self.traverse(
                TokenRange { begin: cur, last: sub_hi },
                depth + 1,
                evolve(state_j, c),
                evolve(state_0, c),
                evolve,
            );
            if sub_hi == tr.last {
                break;
            }
            cur = sub_hi + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::config::{DEFAULT_DICT12_CONFIG, OnPairTrainingConfig};
    use crate::test_corpus::{make_raw, user_strings};

    fn make_column<S: AsRef<[u8]>>(strings: &[S]) -> Column {
        make_column_bits(strings, 14)
    }

    fn make_column_bits<S: AsRef<[u8]>>(strings: &[S], bits: u32) -> Column {
        let raw = make_raw(strings);
        let cfg = OnPairTrainingConfig { bits, threshold: 0.5, seed: 42 };
        Column::compress(&raw.data, &raw.offsets_u64, cfg).unwrap()
    }

    fn contains_any(col: &Column, patterns: &[&[u8]]) -> Vec<usize> {
        let dict = col.dictionary().clone();
        let ac = AhoCorasickAutomaton::new(patterns, &dict);
        col.scan(ac)
    }

    // ── Trie sanity ────────────────────────────────────────────────────────

    #[test]
    fn trie_basic_advance() {
        let patterns: Vec<&[u8]> = vec![b"he", b"she", b"his", b"hers"];
        let t = AhoCorasickTrie::new(&patterns);
        // root --h--> ? --e--> accepting "he"
        let s_h = t.advance(ROOT_STATE, b'h');
        assert!(!t.is_accepting(s_h));
        let s_he = t.advance(s_h, b'e');
        assert!(t.is_accepting(s_he));
        // Walking "ushers" should reach an accepting state via failure links.
        let mut s = ROOT_STATE;
        for &c in b"ushers" {
            s = t.advance(s, c);
        }
        assert!(t.is_accepting(s));
    }

    #[test]
    fn trie_empty_pattern_marks_root_accepting() {
        let patterns: Vec<&[u8]> = vec![b""];
        let t = AhoCorasickTrie::new(&patterns);
        assert!(t.is_accepting(ROOT_STATE));
    }

    // ── Basic multi-pattern search ────────────────────────────────────────

    #[test]
    fn basic_multi_pattern() {
        let data = [
            "error: disk full",
            "warning: low memory",
            "info: all ok",
            "fatal: kernel panic",
            "debug: trace",
        ];
        let col = make_column(&data);
        let result = contains_any(&col, &[b"error", b"fatal"]);
        assert_eq!(result, vec![0, 3]);
    }

    #[test]
    fn single_pattern() {
        let data = ["abc", "def", "abc_xyz"];
        let col = make_column(&data);
        assert_eq!(contains_any(&col, &[b"abc"]), vec![0, 2]);
    }

    #[test]
    fn no_matches() {
        let data = ["abc", "def", "ghi"];
        let col = make_column(&data);
        assert!(contains_any(&col, &[b"xyz", b"uvw"]).is_empty());
    }

    #[test]
    fn all_strings_match() {
        let data = ["abc_def", "def_ghi", "ghi_abc"];
        let col = make_column(&data);
        let r = contains_any(&col, &[b"abc", b"def", b"ghi"]);
        assert_eq!(r.len(), 3);
    }

    // ── Empty patterns ────────────────────────────────────────────────────

    #[test]
    fn empty_pattern_matches_all() {
        let data = ["abc", "def"];
        let col = make_column(&data);
        assert_eq!(contains_any(&col, &[b""]).len(), 2);
    }

    #[test]
    fn empty_pattern_set_matches_none() {
        let data = ["abc", "def"];
        let col = make_column(&data);
        assert!(contains_any(&col, &[]).is_empty());
    }

    // ── Overlapping / prefix patterns ─────────────────────────────────────

    #[test]
    fn overlapping_patterns() {
        let data = ["abcdef", "bcde", "xyz"];
        let col = make_column(&data);
        assert_eq!(contains_any(&col, &[b"abc", b"bcd"]), vec![0, 1]);
    }

    #[test]
    fn prefix_patterns() {
        let data = ["abc", "ab", "xyz"];
        let col = make_column(&data);
        assert_eq!(contains_any(&col, &[b"ab", b"abc"]), vec![0, 1]);
    }

    // ── All bit widths ────────────────────────────────────────────────────

    #[test]
    fn works_across_bit_widths() {
        let data = ["error log", "warning log", "info log"];
        for bw in 9u32..=16 {
            let col = make_column_bits(&data, bw);
            let r = contains_any(&col, &[b"error", b"warning"]);
            assert_eq!(r.len(), 2, "bw={bw}");
        }
    }

    // ── Consistency with single-needle KMP ────────────────────────────────

    #[test]
    fn single_pattern_matches_kmp() {
        use crate::kmp::KmpAutomaton;
        let data = user_strings(50);
        let col = make_column(&data);
        let dict = col.dictionary().clone();
        let kmp = KmpAutomaton::new(b"https", &dict);
        let kmp_result = col.scan(kmp);
        let ac = AhoCorasickAutomaton::new(&[b"https"], &dict);
        let ac_result = col.scan(ac);
        assert_eq!(kmp_result, ac_result);
    }

    // ── Empty column ──────────────────────────────────────────────────────

    #[test]
    fn empty_column_returns_empty() {
        let strings: Vec<&[u8]> = vec![];
        let raw = make_raw(&strings);
        let col = Column::compress(&raw.data, &raw.offsets_u64, DEFAULT_DICT12_CONFIG).unwrap();
        let r = contains_any(&col, &[b"abc", b"def"]);
        assert!(r.is_empty());
    }
}
