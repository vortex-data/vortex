// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 2_048;

/// The per-row pattern arms compile one pattern per row in the worst case, which costs an order of
/// magnitude more per row than matching against a constant pattern. They use a smaller fixture to
/// stay inside the 1 ms per-iteration budget from `docs/developer-guide/benchmarking.md`.
const PER_ROW_PATTERN_ROWS: usize = 512;

/// Random lowercase strings of 4..=24 bytes, some with a `hello` infix.
fn strings(len: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let len_dist = Uniform::new_inclusive(4usize, 24).unwrap();
    VarBinViewArray::from_iter_str((0..len).map(|i| {
        let len = rng.sample(len_dist);
        let mut s: String = (0..len)
            .map(|_| char::from(rng.random_range(b'a'..=b'z')))
            .collect();
        if i % 7 == 0 {
            s.insert_str(len / 2, "hello");
        }
        s
    }))
    .into_array()
}

fn bench_like(bencher: Bencher, pattern: &str, options: LikeOptions) {
    let session = vortex_array::array_session();
    let array = strings(ARRAY_SIZE);
    bencher
        .with_inputs(|| {
            (
                ScalarFnArray::try_new(
                    TypedScalarFnInstance::new(Like, options).erased(),
                    vec![
                        array.clone(),
                        ConstantArray::new(pattern, ARRAY_SIZE).into_array(),
                    ],
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn like_exact(bencher: Bencher) {
    bench_like(bencher, "hello", LikeOptions::default());
}

#[divan::bench]
fn like_prefix(bencher: Bencher) {
    bench_like(bencher, "hello%", LikeOptions::default());
}

#[divan::bench]
fn like_suffix(bencher: Bencher) {
    bench_like(bencher, "%hello", LikeOptions::default());
}

#[divan::bench]
fn like_contains(bencher: Bencher) {
    bench_like(bencher, "%hello%", LikeOptions::default());
}

#[divan::bench]
fn like_regex(bencher: Bencher) {
    bench_like(bencher, "h_llo%w%d", LikeOptions::default());
}

fn bench_per_row_patterns(bencher: Bencher, patterns: ArrayRef) {
    let session = vortex_array::array_session();
    let array = strings(patterns.len());
    bencher
        .with_inputs(|| {
            (
                ScalarFnArray::try_new(
                    TypedScalarFnInstance::new(Like, LikeOptions::default()).erased(),
                    vec![array.clone(), patterns.clone()],
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn like_per_row_patterns(bencher: Bencher) {
    // A non-constant pattern child takes the per-row path; repeated patterns hit the compile cache.
    let patterns = VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|_| "hello%")).into_array();
    bench_per_row_patterns(bencher, patterns);
}

/// The cached half of the compile-cache pair: every row repeats one pattern whose five-byte shape
/// matches the distinct-pattern arm, so the two differ only in whether the cache hits. Both use
/// [`PER_ROW_PATTERN_ROWS`] rows to stay comparable.
///
/// [`like_per_row_distinct_patterns`]
#[divan::bench]
fn like_per_row_repeated_patterns(bencher: Bencher) {
    let patterns =
        VarBinViewArray::from_iter_str((0..PER_ROW_PATTERN_ROWS).map(|_| "%aaa%")).into_array();
    bench_per_row_patterns(bencher, patterns);
}

/// The per-row path with the compile cache defeated: every row carries a distinct pattern of the
/// same shape as the repeated-pattern arm, so each row pays one pattern compilation.
///
/// [`like_per_row_repeated_patterns`]
#[divan::bench]
fn like_per_row_distinct_patterns(bencher: Bencher) {
    let patterns = VarBinViewArray::from_iter_str(
        (0..PER_ROW_PATTERN_ROWS).map(|i| format!("%{}%", distinct_trigram(i))),
    )
    .into_array();
    bench_per_row_patterns(bencher, patterns);
}

/// A distinct three-letter lowercase infix per row, so every pattern has the same shape while
/// no two rows share a compiled pattern.
fn distinct_trigram(i: usize) -> String {
    let letter = |shift: usize| char::from(b'a' + u8::try_from((i >> shift) % 26).unwrap());
    [letter(0), letter(5), letter(10)].iter().collect()
}

#[divan::bench]
fn ilike_contains(bencher: Bencher) {
    bench_like(
        bencher,
        "%HELLO%",
        LikeOptions {
            negated: false,
            case_insensitive: true,
        },
    );
}
