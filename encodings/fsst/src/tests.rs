// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation, clippy::unnecessary_map_or)]

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::buffer;
use vortex_mask::Mask;

use crate::FSSTArray;
use crate::FSSTVTable;
use crate::fsst_compress;
use crate::fsst_train_compressor;

/// this function is VERY slow on miri, so we only want to run it once
pub(crate) fn build_fsst_array() -> ArrayRef {
    let mut input_array = VarBinBuilder::<i32>::with_capacity(3);
    input_array.append_value(b"The Greeks never said that the limit could not be overstepped");
    input_array.append_value(
        b"They said it existed and that whoever dared to exceed it was mercilessly struck down",
    );
    input_array.append_value(b"Nothing in present history can contradict them");
    let input_array = input_array.finish(DType::Utf8(Nullability::NonNullable));

    let compressor = fsst_train_compressor(&input_array);
    fsst_compress(input_array, &compressor).into_array()
}

#[test]
fn test_fsst_array_ops() {
    // first test the scalar_at values
    let fsst_array = build_fsst_array();
    assert_nth_scalar!(
        fsst_array,
        0,
        "The Greeks never said that the limit could not be overstepped"
    );
    assert_nth_scalar!(
        fsst_array,
        1,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );
    assert_nth_scalar!(
        fsst_array,
        2,
        "Nothing in present history can contradict them"
    );

    // test slice
    let fsst_sliced = fsst_array.slice(1..3).unwrap();
    assert!(fsst_sliced.is::<FSSTVTable>());
    assert_eq!(fsst_sliced.len(), 2);
    assert_nth_scalar!(
        fsst_sliced,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );
    assert_nth_scalar!(
        fsst_sliced,
        1,
        "Nothing in present history can contradict them"
    );

    // test take
    let indices = buffer![0, 2].into_array();
    let fsst_taken = fsst_array.take(indices).unwrap();
    assert_eq!(fsst_taken.len(), 2);
    assert_nth_scalar!(
        fsst_taken,
        0,
        "The Greeks never said that the limit could not be overstepped"
    );
    assert_nth_scalar!(
        fsst_taken,
        1,
        "Nothing in present history can contradict them"
    );

    // test filter
    let mask = Mask::from_iter([false, true, true]);

    let fsst_filtered = fsst_array.filter(mask).unwrap();

    assert_eq!(fsst_filtered.len(), 2);
    assert_nth_scalar!(
        fsst_filtered,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );

    // test to_canonical
    let canonical_array = fsst_array.to_varbinview().into_array();

    assert_arrays_eq!(fsst_array.to_array(), canonical_array);
}

// ---------------------------------------------------------------------------
// DFA-based prefix and contains matching on FSST-compressed codes.
//
// The key idea: precompute a transition table so that each FSST code
// (which decodes to 1–8 bytes) maps to a single table lookup instead
// of a per-byte inner loop.  This makes the matching loop O(|codes|)
// rather than O(|decoded_string|).
// ---------------------------------------------------------------------------

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_array::accessor::ArrayAccessor;

/// Build the KMP failure (partial-match) table for `needle`.
fn kmp_failure_table(needle: &[u8]) -> Vec<usize> {
    let mut failure = vec![0usize; needle.len()];
    let mut k = 0;
    for i in 1..needle.len() {
        while k > 0 && needle[k] != needle[i] {
            k = failure[k - 1];
        }
        if needle[k] == needle[i] {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}

/// Build a full KMP byte-level transition table.
///
/// `byte_transitions[state * 256 + byte] = next_state`
///
/// This is the classic DFA form of KMP: for every (state, byte) pair we
/// know the next state without branching through the failure chain at
/// match time.
fn kmp_byte_transitions(needle: &[u8]) -> Vec<u16> {
    let n_states = needle.len() + 1;
    let accept = needle.len() as u16;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u16; n_states * 256];
    for state in 0..n_states {
        for byte in 0..256u16 {
            if state == needle.len() {
                // Accept is absorbing.
                table[state * 256 + byte as usize] = accept;
                continue;
            }
            let mut s = state;
            loop {
                if byte as u8 == needle[s] {
                    s += 1;
                    break;
                }
                if s == 0 {
                    break;
                }
                s = failure[s - 1];
            }
            table[state * 256 + byte as usize] = s as u16;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// FsstPrefixDfa — one table-lookup per code for `starts_with`
// ---------------------------------------------------------------------------

/// DFA whose states track how many leading bytes of `prefix` have been
/// matched.  Transitions are precomputed per (state, symbol-code) so the
/// hot loop does one table lookup per FSST code.
///
/// States:
///   0 ..  prefix.len()-1  — matched that many prefix bytes
///   prefix.len()          — ACCEPT  (whole prefix matched)
///   prefix.len()+1        — FAIL    (absorbing dead state)
struct FsstPrefixDfa {
    /// `symbol_transitions[state * n_symbols + code]`
    symbol_transitions: Vec<u16>,
    /// `escape_transitions[state * 256 + escaped_byte]`
    escape_transitions: Vec<u16>,
    n_symbols: usize,
    accept_state: u16,
    fail_state: u16,
}

impl FsstPrefixDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], prefix: &[u8]) -> Self {
        let n_symbols = symbols.len();
        let accept_state = prefix.len() as u16;
        let fail_state = prefix.len() as u16 + 1;
        let n_states = prefix.len() + 2;

        let mut symbol_transitions = vec![fail_state; n_states * n_symbols];
        let mut escape_transitions = vec![fail_state; n_states * 256];

        for state in 0..n_states {
            // Accept and fail are absorbing.
            if state as u16 == accept_state {
                for code in 0..n_symbols {
                    symbol_transitions[state * n_symbols + code] = accept_state;
                }
                for b in 0..256 {
                    escape_transitions[state * 256 + b] = accept_state;
                }
                continue;
            }
            if state as u16 == fail_state {
                // Already filled with fail_state.
                continue;
            }

            // Symbol transitions: simulate matching all symbol bytes.
            for code in 0..n_symbols {
                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;
                let remaining = prefix.len() - state;
                let cmp = sym_len.min(remaining);

                if sym[..cmp] == prefix[state..state + cmp] {
                    let next = state + cmp;
                    symbol_transitions[state * n_symbols + code] = if next >= prefix.len() {
                        accept_state
                    } else {
                        next as u16
                    };
                }
                // else: stays fail_state (default)
            }

            // Escape transitions: single byte.
            for b in 0..256usize {
                if b as u8 == prefix[state] {
                    let next = state + 1;
                    escape_transitions[state * 256 + b] = if next >= prefix.len() {
                        accept_state
                    } else {
                        next as u16
                    };
                }
                // else: stays fail_state
            }
        }

        Self {
            symbol_transitions,
            escape_transitions,
            n_symbols,
            accept_state,
            fail_state,
        }
    }

    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u16;
        let mut pos = 0;

        while pos < codes.len() {
            if state == self.accept_state {
                return true;
            }
            if state == self.fail_state {
                return false;
            }

            let code = codes[pos];
            pos += 1;

            if code == ESCAPE_CODE {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + b as usize];
            } else {
                debug_assert!(
                    (code as usize) < self.n_symbols,
                    "code {code} >= n_symbols {}",
                    self.n_symbols,
                );
                state = self.symbol_transitions[state as usize * self.n_symbols + code as usize];
            }
        }

        state == self.accept_state
    }
}

// ---------------------------------------------------------------------------
// FsstContainsDfa — one table-lookup per code for substring search
// ---------------------------------------------------------------------------

/// DFA that checks whether the decoded string contains `needle`.
///
/// Built by precomputing, for each (KMP-state, symbol-code), the state
/// reached after feeding all of that symbol's bytes through the KMP
/// automaton.  Escape codes fall back to the byte-level KMP table
/// (one lookup per escaped byte, but escapes are rare).
struct FsstContainsDfa {
    /// `symbol_transitions[state * n_symbols + code]`
    symbol_transitions: Vec<u16>,
    /// `escape_transitions[state * 256 + byte]`  (= the KMP byte-level table)
    escape_transitions: Vec<u16>,
    n_symbols: usize,
    accept_state: u16,
}

impl FsstContainsDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n_symbols = symbols.len();
        let accept_state = needle.len() as u16;
        let n_states = needle.len() + 1;

        // Byte-level KMP DFA — also used directly for escape transitions.
        let byte_table = kmp_byte_transitions(needle);

        // Per-symbol transitions: simulate feeding all symbol bytes.
        let mut symbol_transitions = vec![0u16; n_states * n_symbols];
        for state in 0..n_states {
            for code in 0..n_symbols {
                if state as u16 == accept_state {
                    symbol_transitions[state * n_symbols + code] = accept_state;
                    continue;
                }

                let sym = symbols[code].to_u64().to_le_bytes();
                let sym_len = symbol_lengths[code] as usize;

                let mut s = state as u16;
                for &b in &sym[..sym_len] {
                    if s == accept_state {
                        break;
                    }
                    s = byte_table[s as usize * 256 + b as usize];
                }
                symbol_transitions[state * n_symbols + code] = s;
            }
        }

        Self {
            symbol_transitions,
            escape_transitions: byte_table,
            n_symbols,
            accept_state,
        }
    }

    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u16;
        let mut pos = 0;

        while pos < codes.len() {
            if state == self.accept_state {
                return true;
            }

            let code = codes[pos];
            pos += 1;

            if code == ESCAPE_CODE {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + b as usize];
            } else {
                debug_assert!(
                    (code as usize) < self.n_symbols,
                    "code {code} >= n_symbols {}",
                    self.n_symbols,
                );
                state = self.symbol_transitions[state as usize * self.n_symbols + code as usize];
            }
        }

        state == self.accept_state
    }
}

// ---------------------------------------------------------------------------
// Helpers that apply the DFAs across an FSSTArray
// ---------------------------------------------------------------------------

fn fsst_prefix_match(array: &FSSTArray, prefix: &[u8]) -> Vec<bool> {
    if prefix.is_empty() {
        return vec![true; array.len()];
    }
    let dfa = FsstPrefixDfa::new(
        array.symbols().as_slice(),
        array.symbol_lengths().as_slice(),
        prefix,
    );
    array.codes().with_iterator(|iter| {
        iter.map(|codes| match codes {
            Some(c) => dfa.matches(c),
            None => false,
        })
        .collect()
    })
}

fn fsst_contains_match(array: &FSSTArray, needle: &[u8]) -> Vec<bool> {
    if needle.is_empty() {
        return vec![true; array.len()];
    }
    let dfa = FsstContainsDfa::new(
        array.symbols().as_slice(),
        array.symbol_lengths().as_slice(),
        needle,
    );
    array.codes().with_iterator(|iter| {
        iter.map(|codes| match codes {
            Some(c) => dfa.matches(c),
            None => false,
        })
        .collect()
    })
}

fn make_fsst(strings: &[Option<&str>]) -> FSSTArray {
    let varbin = VarBinArray::from_iter(
        strings.iter().copied(),
        DType::Utf8(if strings.iter().any(|s| s.is_none()) {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        }),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

// ---- prefix tests ----

#[test]
fn test_prefix_basic() {
    let fsst = make_fsst(&[
        Some("http://example.com"),
        Some("http://test.org"),
        Some("ftp://files.net"),
        Some("http://vortex.dev"),
        Some("ssh://server.io"),
    ]);
    assert_eq!(
        fsst_prefix_match(&fsst, b"http"),
        [true, true, false, true, false],
    );
}

#[test]
fn test_prefix_empty() {
    let fsst = make_fsst(&[Some("abc"), Some(""), Some("xyz")]);
    assert_eq!(fsst_prefix_match(&fsst, b""), [true, true, true]);
}

#[test]
fn test_prefix_no_match() {
    let fsst = make_fsst(&[Some("abc"), Some("def"), Some("ghi")]);
    assert_eq!(fsst_prefix_match(&fsst, b"xyz"), [false, false, false]);
}

#[test]
fn test_prefix_mid_symbol_boundary() {
    let fsst = make_fsst(&[
        Some("abcdef"),
        Some("abcxyz"),
        Some("abdxyz"),
        Some("xyzabc"),
    ]);
    assert_eq!(fsst_prefix_match(&fsst, b"ab"), [true, true, true, false],);
}

#[test]
fn test_prefix_empty_strings() {
    let fsst = make_fsst(&[Some(""), Some("a"), Some(""), Some("abc")]);
    assert_eq!(fsst_prefix_match(&fsst, b"a"), [false, true, false, true],);
}

#[test]
fn test_prefix_long_repeated() {
    let fsst = make_fsst(&[
        Some("the quick brown fox jumps"),
        Some("the quick red fox sleeps"),
        Some("the slow brown dog sits"),
        Some("a totally different string"),
        Some("the quick brown fox runs"),
    ]);
    assert_eq!(
        fsst_prefix_match(&fsst, b"the quick"),
        [true, true, false, false, true],
    );
}

// ---- contains tests ----

#[test]
fn test_contains_basic() {
    let fsst = make_fsst(&[
        Some("hello world"),
        Some("say hello"),
        Some("goodbye"),
        Some("hellooo"),
    ]);
    assert_eq!(
        fsst_contains_match(&fsst, b"hello"),
        [true, true, false, true],
    );
}

#[test]
fn test_contains_empty_needle() {
    let fsst = make_fsst(&[Some("abc"), Some("")]);
    assert_eq!(fsst_contains_match(&fsst, b""), [true, true]);
}

#[test]
fn test_contains_no_match() {
    let fsst = make_fsst(&[Some("abc"), Some("def"), Some("ghi")]);
    assert_eq!(fsst_contains_match(&fsst, b"xyz"), [false, false, false],);
}

#[test]
fn test_contains_at_end() {
    let fsst = make_fsst(&[
        Some("foobar_sdf"),
        Some("sdf_start"),
        Some("mid_sdf_mid"),
        Some("nothing"),
    ]);
    assert_eq!(
        fsst_contains_match(&fsst, b"sdf"),
        [true, true, true, false],
    );
}

#[test]
fn test_contains_overlapping_pattern() {
    let fsst = make_fsst(&[Some("aaab"), Some("aab"), Some("ab"), Some("b")]);
    assert_eq!(
        fsst_contains_match(&fsst, b"aab"),
        [true, true, false, false],
    );
}

#[test]
fn test_contains_cross_symbol_boundary() {
    let fsst = make_fsst(&[
        Some("abcdefgh"),
        Some("xxcdexx"),
        Some("nothing_here"),
        Some("abcde_fgh"),
    ]);
    assert_eq!(
        fsst_contains_match(&fsst, b"cde"),
        [true, true, false, true],
    );
}

#[test]
fn test_contains_long_strings() {
    let fsst = make_fsst(&[
        Some("the quick brown fox jumps over the lazy dog"),
        Some("a]short"),
        Some("the lazy dog sleeps"),
        Some("no match here at all"),
    ]);
    assert_eq!(
        fsst_contains_match(&fsst, b"lazy dog"),
        [true, false, true, false],
    );
}

// ---- DFA correctness: verify against brute-force decompress-and-check ----

#[test]
fn test_dfa_matches_decompressed_prefix() {
    let strings: Vec<Option<&str>> = vec![
        Some("http://example.com/page/1"),
        Some("https://secure.example.com"),
        Some("ftp://files.example.com"),
        Some("http://another.site.org"),
        Some("mailto:user@example.com"),
        Some("http://x"),
        Some("h"),
        Some(""),
    ];
    let fsst = make_fsst(&strings);

    for prefix in [
        b"".as_slice(),
        b"h",
        b"ht",
        b"htt",
        b"http",
        b"http://",
        b"http://example",
    ] {
        let dfa_result = fsst_prefix_match(&fsst, prefix);
        let expected: Vec<bool> = strings
            .iter()
            .map(|s| s.map_or(false, |s| s.as_bytes().starts_with(prefix)))
            .collect();
        assert_eq!(
            dfa_result,
            expected,
            "prefix = {:?}",
            std::str::from_utf8(prefix)
        );
    }
}

#[test]
fn test_dfa_matches_decompressed_contains() {
    let strings: Vec<Option<&str>> = vec![
        Some("the quick brown fox jumps over the lazy dog"),
        Some("a lazy cat sleeps"),
        Some("nothing to see here"),
        Some("foxes are quick"),
        Some(""),
        Some("lazy"),
    ];
    let fsst = make_fsst(&strings);

    for needle in [
        b"".as_slice(),
        b"lazy",
        b"quick",
        b"fox",
        b"the",
        b"zzz",
        b"lazy dog",
    ] {
        let dfa_result = fsst_contains_match(&fsst, needle);
        let expected: Vec<bool> = strings
            .iter()
            .map(|s| {
                s.map_or(false, |s| {
                    if needle.is_empty() {
                        true
                    } else {
                        s.as_bytes().windows(needle.len()).any(|w| w == needle)
                    }
                })
            })
            .collect();
        assert_eq!(
            dfa_result,
            expected,
            "needle = {:?}",
            std::str::from_utf8(needle)
        );
    }
}
