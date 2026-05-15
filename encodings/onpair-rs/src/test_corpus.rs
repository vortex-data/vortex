// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Shared test corpus generators. These replicate the helpers in the C++
// `tests/helpers/corpus.h` so the unit tests in `trainer`, `parser`,
// `decoder`, and `search` all draw from the same data and can be compared
// head-to-head against the upstream test suite.

#![allow(dead_code)]

use rand::RngExt;
use rand::SeedableRng;

use crate::types::MAX_TOKEN_SIZE;

/// Arrow-style flat representation of a list of byte strings.
pub(crate) struct Raw {
    pub data: Vec<u8>,
    pub offsets: Vec<u32>,
    pub offsets_u64: Vec<u64>,
    pub n: usize,
}

pub(crate) fn make_raw<S: AsRef<[u8]>>(strings: &[S]) -> Raw {
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    let mut offsets_u64 = Vec::with_capacity(strings.len() + 1);
    offsets.push(0u32);
    offsets_u64.push(0u64);
    for s in strings {
        data.extend_from_slice(s.as_ref());
        offsets.push(data.len() as u32);
        offsets_u64.push(data.len() as u64);
    }
    Raw { data, offsets, offsets_u64, n: strings.len() }
}

/// URL-shaped repetitive corpus — easy BPE merge targets.
pub(crate) fn user_strings(n: usize) -> Vec<String> {
    const BASES: &[&str] = &[
        "https://www.example.com/page",
        "https://www.example.com/data",
        "https://www.test.org/page",
        "ftp://files.example.com/x",
        "https://docs.example.com/spec",
        "https://api.example.net/v1",
    ];
    (0..n).map(|i| BASES[i % BASES.len()].to_string()).collect()
}

/// All copies of the same single-character string of length `len`.
pub(crate) fn homogeneous_strings(n: usize, len: usize, ch: u8) -> Vec<Vec<u8>> {
    (0..n).map(|_| vec![ch; len]).collect()
}

/// "abab..." period-2 strings of length `len`.
pub(crate) fn alternating_strings(n: usize, len: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|_| {
            (0..len).map(|i| if i.is_multiple_of(2) { b'a' } else { b'b' }).collect()
        })
        .collect()
}

/// Random ascii lowercase strings, length 1..=max_len.
pub(crate) fn random_ascii_strings(n: usize, max_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let l = rng.random_range(1..=max_len);
            (0..l).map(|_| rng.random_range(b'a'..=b'z')).collect()
        })
        .collect()
}

/// Random bytes over the full 0..=255 range.
pub(crate) fn binary_strings(n: usize, max_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let l = rng.random_range(1..=max_len);
            (0..l).map(|_| rng.random_range(0..=255u32) as u8).collect()
        })
        .collect()
}

/// `n` fixed-length strings of length `len`, content rotating through the
/// lowercase alphabet so adjacent strings differ.
pub(crate) fn fixed_length_strings(n: usize, len: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| (0..len).map(|j| b'a' + ((i + j) as u8 % 26)).collect())
        .collect()
}

/// Random strings with length 0..=max_len — exercises empty + max-len paths.
pub(crate) fn mixed_length_strings(n: usize, max_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let l = rng.random_range(0..=max_len);
            (0..l).map(|_| rng.random_range(b'a'..=b'z')).collect()
        })
        .collect()
}

/// One row per byte value 0..=255 — exercises the base dictionary.
pub(crate) fn single_byte_strings() -> Vec<Vec<u8>> {
    (0u16..=255).map(|i| vec![i as u8]).collect()
}

/// `n` empty strings.
pub(crate) fn empty_strings(n: usize) -> Vec<Vec<u8>> {
    vec![vec![]; n]
}

/// A corpus designed to trigger longer-token discovery: `n` copies of a
/// long fixed string that is exactly `MAX_TOKEN_SIZE` bytes.
pub(crate) fn max_token_strings(n: usize) -> Vec<Vec<u8>> {
    let pat: Vec<u8> = (0..MAX_TOKEN_SIZE as u8).collect();
    vec![pat; n]
}
