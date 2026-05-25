// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Divan benchmarks for every decimal operation: the best split kernel vs the
//! Arrow-rs decimal kernel, so the numbers in PERFORMANCE.md are reproducible.
//!
//! Both sides are in one binary, so they share the compiled feature set. Run
//! cache-resident and pinned for stable numbers:
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" taskset -c 1 \
//!   cargo bench -p decimal-split-experiment --bench decimal_ops
//! ```
//!
//! `N` is 65536 (~2 MiB working set = L2-resident), the regime a chunked engine
//! actually runs in. Compare an `*_arrow` bench against its `*_split` sibling.

use decimal_split_experiment::aggregate;
use decimal_split_experiment::arrow_ref;
use decimal_split_experiment::compare;
use decimal_split_experiment::data;
use decimal_split_experiment::data::Magnitude;
use decimal_split_experiment::layout::SplitI128;
use decimal_split_experiment::layout::SplitI256;
use decimal_split_experiment::muldiv;
use divan::Bencher;
use divan::black_box;
use divan::counter::ItemsCount;

fn main() {
    divan::main();
}

/// L2-resident working set (~2 MiB for two i128 columns).
const N: usize = 65536;

// ---- compare (lt) ------------------------------------------------------------

#[divan::bench]
fn lt_i128_arrow(bencher: Bencher) {
    let a = arrow_ref::decimal128(&data::gen_i128(N, Magnitude::Large, 1), 38, 0);
    let b = arrow_ref::decimal128(&data::gen_i128(N, Magnitude::Large, 2), 38, 0);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::lt_decimal128(black_box(&a), black_box(&b))));
}

#[divan::bench]
fn lt_i128_split(bencher: Bencher) {
    let a = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Large, 1));
    let b = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Large, 2));
    let mut out = vec![0u8; compare::bitmap_len(N)];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| compare::lt_i128_u4(black_box(&a), black_box(&b), black_box(&mut out)));
}

/// Constant high limb (small decimals): low-limb compare only.
#[divan::bench]
fn lt_i128_split_const_hi(bencher: Bencher) {
    let a = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Small, 1));
    let b = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Small, 2));
    let mut out = vec![0u8; compare::bitmap_len(N)];
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        compare::lt_i128_const_hi(
            black_box(&a.lo),
            0,
            black_box(&b.lo),
            0,
            black_box(&mut out),
        );
    });
}

#[divan::bench]
fn lt_i256_arrow(bencher: Bencher) {
    let a = arrow_ref::decimal256(&data::gen_i256(N, Magnitude::Large, 1), 76, 0);
    let b = arrow_ref::decimal256(&data::gen_i256(N, Magnitude::Large, 2), 76, 0);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::lt_decimal256(black_box(&a), black_box(&b))));
}

#[divan::bench]
fn lt_i256_split(bencher: Bencher) {
    let a = SplitI256::from_aos(&data::gen_i256(N, Magnitude::Large, 1));
    let b = SplitI256::from_aos(&data::gen_i256(N, Magnitude::Large, 2));
    let mut out = vec![0u8; compare::bitmap_len(N)];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| compare::lt_i256(black_box(&a), black_box(&b), black_box(&mut out)));
}

// ---- sum ---------------------------------------------------------------------

#[divan::bench]
fn sum_i128_arrow(bencher: Bencher) {
    let a = arrow_ref::decimal128(&data::gen_i128(N, Magnitude::Large, 1), 38, 0);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::sum_decimal128(black_box(&a))));
}

/// Exact i256-widening sum (overflow-safe; Arrow wraps i128).
#[divan::bench]
fn sum_i128_split_widening(bencher: Bencher) {
    let a = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Large, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| black_box(aggregate::sum_i128_widening(black_box(&a))));
}

/// Constant high limb: 4-accumulator low-limb sum (skips the high stream).
#[divan::bench]
fn sum_i128_split_const_hi(bencher: Bencher) {
    let a = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Small, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| black_box(aggregate::sum_i128_lo_only_u4(black_box(&a))));
}

// ---- min / max ---------------------------------------------------------------

#[divan::bench]
fn min_i128_arrow(bencher: Bencher) {
    let a = arrow_ref::decimal128(&data::gen_i128(N, Magnitude::Large, 1), 38, 0);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::min_decimal128(black_box(&a))));
}

#[divan::bench]
fn min_i128_split(bencher: Bencher) {
    let a = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Large, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| black_box(aggregate::min_i128(black_box(&a))));
}

// ---- multiply (small operands so the product fits precision 38) --------------

#[divan::bench]
fn mul_i128_arrow(bencher: Bencher) {
    let a = arrow_ref::decimal128(&data::gen_i128(N, Magnitude::Small, 1), 38, 0);
    let b = arrow_ref::decimal128(&data::gen_i128(N, Magnitude::Small, 2), 38, 0);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::mul_decimal128(black_box(&a), black_box(&b))));
}

#[divan::bench]
fn mul_i128_split(bencher: Bencher) {
    let a = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Small, 1));
    let b = SplitI128::from_aos(&data::gen_i128(N, Magnitude::Small, 2));
    let mut out = a.zeroed_like();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| muldiv::mul_i128(black_box(&a), black_box(&b), black_box(&mut out)));
}

// ---- divide (non-zero divisors; note: different rounding semantics) ----------

#[divan::bench]
fn div_i128_arrow(bencher: Bencher) {
    let a = arrow_ref::decimal128(&data::gen_i128(N, Magnitude::Small, 1), 38, 0);
    let bvals: Vec<i128> = data::gen_i128(N, Magnitude::Small, 2)
        .into_iter()
        .map(|v| v + 1)
        .collect();
    let b = arrow_ref::decimal128(&bvals, 38, 0);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| black_box(arrow_ref::div_decimal128(black_box(&a), black_box(&b))));
}

#[divan::bench]
fn div_i128_split(bencher: Bencher) {
    let a = data::gen_i128(N, Magnitude::Small, 1);
    let b: Vec<i128> = data::gen_i128(N, Magnitude::Small, 2)
        .into_iter()
        .map(|v| v + 1)
        .collect();
    let mut out = vec![0i128; N];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| muldiv::div_i128_aos(black_box(&a), black_box(&b), black_box(&mut out)));
}
