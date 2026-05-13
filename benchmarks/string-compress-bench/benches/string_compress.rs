// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Divan benches for the string-compression backends.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use string_compress_bench::backends::Backend;
use string_compress_bench::backends::Pushdown;
use string_compress_bench::backends::fsst_rs_backend::FsstRsBackend;
use string_compress_bench::datasets::{
    Corpus, json_like, long_shared_prefix, natural_words, random_bytes, short_codes,
    skewed_dictionary, url_like,
};

#[cfg(feature = "onpair-cpp")]
use string_compress_bench::backends::onpair_cpp_backend::OnPairCppBackend;
#[cfg(feature = "fsst-cpp")]
use string_compress_bench::backends::{fsst_cpp_8::FsstCpp8Backend, fsst_cpp_12::FsstCpp12Backend};
#[cfg(feature = "onpair")]
use string_compress_bench::backends::{
    onpair_backend::OnPairBackend, onpair16_backend::OnPair16Backend,
};

fn main() {
    divan::main();
}

const ROWS: usize = 2048;
const ONPAIR_THRESHOLD: u16 = 4;
const ONPAIR_CPP_BITS: u8 = 14;
const ONPAIR_CPP_SEED: u32 = 42;

fn corpora() -> Vec<Corpus> {
    vec![
        skewed_dictionary(ROWS),
        url_like(ROWS),
        random_bytes(ROWS),
        long_shared_prefix(ROWS),
        natural_words(ROWS),
        json_like(ROWS),
        short_codes(ROWS),
    ]
}

/// Generates the four bench functions for a backend whose factory takes only
/// the input strings. `$ctor` is a closure / path returning an instance of
/// the backend.
macro_rules! backend_benches {
    ($mod:ident, $ctor:expr) => {
        mod $mod {
            use super::*;

            #[divan::bench(args = corpora())]
            fn compress(bencher: Bencher, corpus: &Corpus) {
                bencher.bench(|| ($ctor)(&corpus.strings));
            }

            #[divan::bench(args = corpora())]
            fn decompress(bencher: Bencher, corpus: &Corpus) {
                let b = ($ctor)(&corpus.strings);
                bencher.bench(|| b.decompress_all());
            }

            #[divan::bench(args = corpora())]
            fn pushdown_equals(bencher: Bencher, corpus: &Corpus) {
                let b = ($ctor)(&corpus.strings);
                let needle = corpus.needles[0].clone();
                bencher.bench(|| b.equals(&needle));
            }

            #[divan::bench(args = corpora())]
            fn pushdown_contains(bencher: Bencher, corpus: &Corpus) {
                let b = ($ctor)(&corpus.strings);
                let needle = corpus.needles[0].clone();
                bencher.bench(|| b.contains(&needle));
            }

            #[divan::bench(args = corpora())]
            fn pushdown_starts_with(bencher: Bencher, corpus: &Corpus) {
                let b = ($ctor)(&corpus.strings);
                let needle = corpus.needles[0].clone();
                bencher.bench(|| b.starts_with(&needle));
            }
        }
    };
}

backend_benches!(fsst_rs, |s: &[Vec<u8>]| FsstRsBackend::train_and_compress(
    s
));

#[cfg(feature = "fsst-cpp")]
backend_benches!(fsst_cpp_8, |s: &[Vec<u8>]| {
    FsstCpp8Backend::train_and_compress(s)
});

#[cfg(feature = "fsst-cpp")]
backend_benches!(fsst_cpp_12, |s: &[Vec<u8>]| {
    FsstCpp12Backend::train_and_compress(s)
});

#[cfg(feature = "onpair")]
backend_benches!(onpair, |s: &[Vec<u8>]| OnPairBackend::train_and_compress(
    s,
    ONPAIR_THRESHOLD
));

#[cfg(feature = "onpair")]
backend_benches!(onpair16, |s: &[Vec<u8>]| {
    OnPair16Backend::train_and_compress(s, ONPAIR_THRESHOLD)
});

#[cfg(feature = "onpair-cpp")]
backend_benches!(onpair_cpp, |s: &[Vec<u8>]| {
    OnPairCppBackend::train_and_compress(s, ONPAIR_CPP_BITS, ONPAIR_CPP_SEED)
});
