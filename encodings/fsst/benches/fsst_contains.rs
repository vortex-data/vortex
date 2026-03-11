// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use fsst::ESCAPE_CODE;
use fsst::Symbol;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ToCanonical;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BitBufferMut;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// URL generator
// ---------------------------------------------------------------------------

const DOMAINS: &[&str] = &[
    "google.com",
    "facebook.com",
    "github.com",
    "stackoverflow.com",
    "amazon.com",
    "reddit.com",
    "twitter.com",
    "youtube.com",
    "wikipedia.org",
    "microsoft.com",
    "apple.com",
    "netflix.com",
    "linkedin.com",
    "cloudflare.com",
    "google.co.uk",
    "docs.google.com",
    "mail.google.com",
    "maps.google.com",
    "news.ycombinator.com",
    "arxiv.org",
];

const PATHS: &[&str] = &[
    "/index.html",
    "/about",
    "/search?q=vortex",
    "/user/profile/settings",
    "/api/v2/data",
    "/blog/2024/post",
    "/products/item/12345",
    "/docs/reference/guide",
    "/login",
    "/dashboard/analytics",
];

fn generate_urls(n: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| {
            let scheme = if rng.random_bool(0.8) {
                "https"
            } else {
                "http"
            };
            let domain = DOMAINS[rng.random_range(0..DOMAINS.len())];
            let path = PATHS[rng.random_range(0..PATHS.len())];
            format!("{scheme}://{domain}{path}")
        })
        .collect()
}

fn make_fsst_urls(n: usize) -> FSSTArray {
    let urls = generate_urls(n);
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    fsst_compress(varbin, &compressor)
}

// ---------------------------------------------------------------------------
// DFA (copied from tests — production code would share this)
// ---------------------------------------------------------------------------

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

fn kmp_byte_transitions(needle: &[u8]) -> Vec<u16> {
    let n_states = needle.len() + 1;
    let accept = needle.len() as u16;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u16; n_states * 256];
    for state in 0..n_states {
        for byte in 0..256u16 {
            if state == needle.len() {
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

struct FsstContainsDfa {
    symbol_transitions: Vec<u16>,
    escape_transitions: Vec<u16>,
    n_symbols: usize,
    accept_state: u16,
}

impl FsstContainsDfa {
    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n_symbols = symbols.len();
        let accept_state = needle.len() as u16;
        let n_states = needle.len() + 1;

        let byte_table = kmp_byte_transitions(needle);

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

    #[inline]
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
                state = self.symbol_transitions[state as usize * self.n_symbols + code as usize];
            }
        }

        state == self.accept_state
    }
}

fn dfa_contains_iterator(array: &FSSTArray, needle: &[u8]) -> Vec<bool> {
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

fn dfa_contains_direct(array: &FSSTArray, needle: &[u8]) -> BitBufferMut {
    let dfa = FsstContainsDfa::new(
        array.symbols().as_slice(),
        array.symbol_lengths().as_slice(),
        needle,
    );
    let codes = array.codes();
    let offsets = codes.offsets().to_primitive();
    let all_bytes = codes.bytes();
    let all_bytes = all_bytes.as_slice();
    let n = codes.len();

    match_each_integer_ptype!(offsets.ptype(), |T| {
        let off = offsets.as_slice::<T>();
        BitBufferMut::collect_bool(n, |i| {
            let start = off[i] as usize;
            let end = off[i + 1] as usize;
            dfa.matches(&all_bytes[start..end])
        })
    })
}

fn decompress_then_contains(array: &FSSTArray, needle: &[u8]) -> Vec<bool> {
    let decompressor = array.decompressor();
    array.codes().with_iterator(|iter| {
        iter.map(|codes| match codes {
            Some(c) => {
                let decompressed = decompressor.decompress(c);
                decompressed.windows(needle.len()).any(|w| w == needle)
            }
            None => false,
        })
        .collect()
    })
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

const N: usize = 100_000;
const NEEDLE: &[u8] = b"google";

#[divan::bench]
fn contains_dfa_iterator(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    bencher
        .with_inputs(|| &fsst)
        .bench_refs(|fsst| dfa_contains_iterator(fsst, NEEDLE));
}

#[divan::bench]
fn contains_dfa_direct(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    bencher
        .with_inputs(|| &fsst)
        .bench_refs(|fsst| dfa_contains_direct(fsst, NEEDLE));
}

#[divan::bench]
fn contains_decompress(bencher: Bencher) {
    let fsst = make_fsst_urls(N);
    bencher
        .with_inputs(|| &fsst)
        .bench_refs(|fsst| decompress_then_contains(fsst, NEEDLE));
}
