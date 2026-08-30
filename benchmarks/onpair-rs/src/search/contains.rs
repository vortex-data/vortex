// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressed-domain substring search via a token-level KMP transition table.
//!
//! Substring search (`LIKE '%pattern%'`) runs byte-level KMP, but lifted to
//! tokens: one transition per *token*, not per byte. The transition function is
//! precomputed once into an immutable [`ContainsTable`]; [`contains`] then scans
//! a row's codes holding the KMP state in a local — no decode, no dictionary at
//! scan time, O(#tokens) per row.
//!
//! The table is stored sparsely. `base[token]` is the transition from state 0
//! (a dense array, one entry per token). For an entry state `s > 0` only a few
//! tokens transition differently from `base`, and — because the dictionary is
//! sorted — those tokens form a handful of contiguous [`TokenRange`]s, kept as
//! per-state exceptions. A full `(states × tokens)` matrix would store the same
//! function densely; the sparse form matches the data and stays small.

use super::lookup::prefix_range;
use crate::core::dictionary::DictionaryView;
use crate::core::types::{Token, TokenRange};

/// A KMP state: the number of leading pattern bytes matched, `0..=pattern.len()`.
/// `u8`, so the pattern is limited to 255 bytes.
type State = u8;

/// Immutable token-level KMP transition table for one pattern, built once
/// against a dictionary and shared (by `&`) across every row scanned. Holds no
/// scan state — [`contains`] keeps the KMP state in a local.
#[derive(Debug, Clone)]
pub struct ContainsTable {
    /// Accept state — the pattern's byte length `m`. `0` ⇒ empty pattern.
    accept: State,
    /// `base[token]` = KMP state after running `token`'s bytes from state 0.
    base: Vec<State>,
    /// Per-state exceptions: for entry state `s`, the tokens in
    /// `sparse[offsets[s]..offsets[s + 1]]` transition to their `target` instead
    /// of `base[token]`. Ranges within a state are ascending and disjoint.
    sparse: Vec<SparseTransition>,
    /// `offsets[s]..offsets[s + 1]` bounds state `s`'s exceptions; `len == m + 1`.
    offsets: Vec<u32>,
}

/// Tokens in `range` transition to `target` (overriding `base`).
#[derive(Debug, Clone, Copy)]
struct SparseTransition {
    range: TokenRange,
    target: State,
}

impl ContainsTable {
    /// Build the transition table for `pattern` against the sorted `dict`.
    ///
    /// # Panics
    /// If `pattern` is longer than 255 bytes (KMP states are stored as `u8`).
    pub fn new<V: DictionaryView>(pattern: &[u8], dict: V) -> Self {
        assert!(
            pattern.len() <= State::MAX as usize,
            "contains pattern exceeds 255 bytes"
        );
        let m = pattern.len();
        let num_tokens = dict.num_tokens();

        // Empty pattern: accept state 0, every token a no-op transition.
        if m == 0 {
            return Self {
                accept: 0,
                base: vec![0; num_tokens],
                sparse: Vec::new(),
                offsets: vec![0, 0],
            };
        }

        let mut build = Build {
            dict,
            p: pattern,
            m,
            fail: kmp_failure(pattern),
            base: vec![0; num_tokens],
            sparse: Vec::new(),
            offsets: vec![0u32; m + 1],
            range_start: 0,
        };
        build.base_pass();
        build.sparse_pass();

        Self {
            accept: m as State,
            base: build.base,
            sparse: build.sparse,
            offsets: build.offsets,
        }
    }

    /// KMP transition from `state` on `token`.
    ///
    /// Precondition: `state < accept` — [`contains`] stops at `accept`, so
    /// `offsets[state + 1]` is in bounds.
    #[inline]
    fn next(&self, state: State, token: Token) -> State {
        if state > 0 {
            let lo = self.offsets[state as usize] as usize;
            let hi = self.offsets[state as usize + 1] as usize;
            // Ranges ascending and disjoint ⇒ stop once past the token.
            for tr in &self.sparse[lo..hi] {
                if token < tr.range.begin {
                    break;
                }
                if token <= tr.range.last {
                    return tr.target;
                }
            }
        }
        self.base[token as usize]
    }
}

/// Whether `codes` contains the pattern of `table` as a substring — even one that
/// straddles token boundaries.
///
/// Stateless: the KMP state lives here, starting at 0 each call. Every step is a
/// table lookup; no dictionary and no decode. Early-exits on the first match.
pub fn contains(codes: &[Token], table: &ContainsTable) -> bool {
    let mut state: State = 0;
    if state == table.accept {
        return true; // empty pattern is a substring of everything
    }
    for &code in codes {
        state = table.next(state, code);
        if state == table.accept {
            return true;
        }
    }
    false
}

/// Byte-level KMP failure table: `fail[i]` is the length of the longest proper
/// prefix of `pattern[..=i]` that is also a suffix.
fn kmp_failure(pattern: &[u8]) -> Vec<State> {
    let m = pattern.len();
    let mut fail = vec![0 as State; m];
    let (mut i, mut len) = (1usize, 0usize);
    while i < m {
        if pattern[i] == pattern[len] {
            len += 1;
            fail[i] = len as State;
            i += 1;
        } else if len > 0 {
            len = fail[len - 1] as usize;
        } else {
            fail[i] = 0;
            i += 1;
        }
    }
    fail
}

/// Scratch state for building a [`ContainsTable`].
struct Build<'a, V> {
    dict: V,
    p: &'a [u8],
    m: usize,
    fail: Vec<State>,
    base: Vec<State>,
    sparse: Vec<SparseTransition>,
    offsets: Vec<u32>,
    /// Index into `sparse` where the current entry state's ranges begin (so
    /// `emit` only merges within the state it is filling).
    range_start: usize,
}

impl<V: DictionaryView> Build<'_, V> {
    /// Run `data`'s bytes through the byte-level KMP from state `s` (the accept
    /// state `m` is absorbing).
    fn step_bytes(&self, mut s: State, data: &[u8]) -> State {
        for &b in data {
            if s as usize == self.m {
                return self.m as State;
            }
            while s > 0 && self.p[s as usize] != b {
                s = self.fail[s as usize - 1];
            }
            if self.p[s as usize] == b {
                s += 1;
            }
        }
        s
    }

    /// `base[t]` = state reached by running token `t`'s bytes from state 0.
    fn base_pass(&mut self) {
        let p0 = self.p[0];
        for t in 0..self.base.len() {
            // A token without the pattern's first byte cannot leave state 0.
            let s = {
                let tok = self.dict.token(t as Token);
                if tok.contains(&p0) {
                    self.step_bytes(0, tok)
                } else {
                    0
                }
            };
            self.base[t] = s;
        }
    }

    /// Append `(range, target)`, extending the previous range if it is adjacent
    /// and shares the target. Only merges within the current entry state.
    fn emit(&mut self, range: TokenRange, target: State) {
        if self.sparse.len() > self.range_start {
            let last = self.sparse.last_mut().unwrap();
            if last.target == target && last.range.last as usize + 1 == range.begin as usize {
                last.range.last = range.last;
                return;
            }
        }
        self.sparse.push(SparseTransition { range, target });
    }

    /// Fill the sparse exceptions for every entry state `j` in `1..m`.
    fn sparse_pass(&mut self) {
        let mut relevant: Vec<u8> = Vec::new();
        for j in 1..self.m {
            self.range_start = self.sparse.len();
            self.offsets[j] = self.range_start as u32;

            // Only bytes p[s] along the failure chain j → fail[j-1] → … → 0 can
            // make state j transition differently from state 0; skip the rest.
            relevant.clear();
            let mut s = j as State;
            while s > 0 {
                relevant.push(self.p[s as usize]);
                s = self.fail[s as usize - 1];
            }
            relevant.sort_unstable();
            relevant.dedup();

            for &byte in &relevant {
                let range = prefix_range(self.dict, &[byte]);
                if range.is_empty() {
                    continue;
                }
                let kj = self.step_bytes(j as State, &[byte]);
                let k0 = self.step_bytes(0, &[byte]);
                self.traverse(range, 1, kj, k0);
            }
        }
        self.offsets[self.m] = self.sparse.len() as u32;
    }

    /// Walk the sorted-dictionary subtree `tr` at byte depth `depth`, tracking
    /// the KMP state evolved from entry state j (`kmp_j`) against the one evolved
    /// from state 0 (`kmp_0` = what `base` already records). Emits an exception
    /// for every token where they differ. Pruned where the two states coincide.
    fn traverse(&mut self, tr: TokenRange, depth: usize, kmp_j: State, kmp_0: State) {
        if kmp_j == kmp_0 || tr.is_empty() {
            return;
        }
        let (begin, last) = (tr.begin as usize, tr.last as usize);

        // Full match from state j: it stays at the accept state through the whole
        // subtree, so override any token whose `base` differs.
        if kmp_j as usize == self.m {
            let exit = self.m as State;
            let mut i = begin;
            while i <= last {
                if self.base[i] != exit {
                    let start = i;
                    while i <= last && self.base[i] != exit {
                        i += 1;
                    }
                    self.emit(
                        TokenRange {
                            begin: start as Token,
                            last: (i - 1) as Token,
                        },
                        exit,
                    );
                } else {
                    i += 1;
                }
            }
            return;
        }

        // Tokens of length == depth end here; they all exit at kmp_j.
        let mut cur = begin;
        while cur <= last && self.dict.token_len(cur as Token) == depth {
            cur += 1;
        }
        if cur > begin {
            self.emit(
                TokenRange {
                    begin: begin as Token,
                    last: (cur - 1) as Token,
                },
                kmp_j,
            );
        }
        if cur > last {
            return;
        }

        // Partition the remaining (longer) tokens by their byte at `depth` and
        // recurse into each subtree.
        while cur <= last {
            let c = self.dict.token(cur as Token)[depth];
            let mut sub_hi = cur;
            while sub_hi < last && self.dict.token((sub_hi + 1) as Token)[depth] == c {
                sub_hi += 1;
            }
            let kj = self.step_bytes(kmp_j, &[c]);
            let k0 = self.step_bytes(kmp_0, &[c]);
            self.traverse(
                TokenRange {
                    begin: cur as Token,
                    last: sub_hi as Token,
                },
                depth + 1,
                kj,
                k0,
            );
            cur = sub_hi + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::Dictionary;
    use crate::{Column, DEFAULT_CONFIG, compress};

    fn compress_rows(rows: &[&[u8]]) -> Column<u32> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            bytes.extend_from_slice(r);
            offsets.push(bytes.len() as u32);
        }
        compress(&bytes, &offsets, DEFAULT_CONFIG).unwrap()
    }

    fn byte_contains(hay: &[u8], needle: &[u8]) -> bool {
        needle.is_empty() || hay.windows(needle.len()).any(|w| w == needle)
    }

    /// Decode row `k` to bytes via the into-buffer API, for the oracle.
    fn decode_row(view: crate::ColumnView<'_, u32>, k: usize) -> Vec<u8> {
        let mut buf =
            vec![std::mem::MaybeUninit::uninit(); view.row_decoded_len(k) + crate::DECODE_PADDING];
        // SAFETY: buffer sized for row `k`; view from a trusted column.
        let w = unsafe { view.decompress_row_into(k, &mut buf) };
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), w) }.to_vec()
    }

    /// Driving `contains` over every row must agree with a brute-force
    /// decode-and-substring oracle. The table is built from both the compact and
    /// the wide dictionary to confirm construction is representation-agnostic.
    fn check(rows: &[&[u8]], patterns: &[&[u8]]) {
        let col = compress_rows(rows);
        let view = col.view();
        let wide = view.dict.to_wide();
        for &pat in patterns {
            let want: Vec<usize> = (0..view.num_rows())
                .filter(|&k| byte_contains(&decode_row(view, k), pat))
                .collect();

            for table in [
                ContainsTable::new(pat, view.dict),
                ContainsTable::new(pat, wide.as_view()),
            ] {
                let got: Vec<usize> = (0..view.num_rows())
                    .filter(|&k| contains(view.row_codes(k), &table))
                    .collect();
                assert_eq!(got, want, "pattern {pat:?}");
            }
        }
    }

    #[test]
    fn empty_pattern_matches_all_rows() {
        let rows: &[&[u8]] = &[b"a", b"", b"abc"];
        check(rows, &[b""]);
    }

    #[test]
    fn single_and_multi_token_substrings() {
        let rows: &[&[u8]] = &[b"hello world", b"world peace", b"helloworld", b"hell"];
        check(
            rows,
            &[
                b"hello",
                b"world",
                b"o w",
                b"llowo",
                b"xyz",
                b"hello world",
                b"hello world!",
            ],
        );
    }

    #[test]
    fn substrings_spanning_token_boundaries() {
        // Repetitive corpus → multi-byte tokens; patterns chosen to straddle them.
        let rows: &[&[u8]] = &[b"abcabcabc", b"xabcabcy", b"ababab", b"cab"];
        check(
            rows,
            &[b"abc", b"bca", b"cab", b"bcabca", b"abcabcabc", b"ba"],
        );
    }

    #[test]
    fn repeating_pattern_exercises_failure_links() {
        // Patterns with internal repetition stress the KMP failure function and
        // the sparse cross-token transitions.
        let rows: &[&[u8]] = &[b"aaaaab", b"aabaab", b"ababab", b"aaa"];
        check(rows, &[b"aa", b"aaa", b"aab", b"abab", b"aaaa", b"aabaa"]);
    }

    #[test]
    fn matches_brute_force_on_repetitive_corpus() {
        use crate::test_corpus::user_strings;
        let corpus: Vec<Vec<u8>> = user_strings(50)
            .into_iter()
            .map(String::into_bytes)
            .collect();
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        check(
            &rows,
            &[
                b"example", b"https", b"://", b".com", b"/page", b"ftp", b"zzz", b"w",
            ],
        );
    }

    #[test]
    fn matches_brute_force_on_binary_corpus() {
        use crate::test_corpus::binary_strings;
        let corpus = binary_strings(40, 24, 11);
        let rows: Vec<&[u8]> = corpus.iter().map(Vec::as_slice).collect();
        let patterns: &[&[u8]] = &[b"", b"\x00", b"\xff", b"\x00\x01", &[7u8], &[200u8, 201]];
        check(&rows, patterns);
    }
}
