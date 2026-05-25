// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused decode+filter vs decode-then-filter, for `col > threshold` on stack B.
//! The fused path emits a compact result (count / mask) without ever
//! materialising the f64 column, so it skips the dominant output-write cost.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench filter
//! ```

use divan::Bencher;
use divan::counter::ItemsCount;
use simd_stencil::encode::encode_b;
use simd_stencil::encode::gen_f64;
use simd_stencil::scan;

const N: usize = 1 << 23; // 64 MB column: the at-scale regime where the write dominates
const EXP: i32 = 2;
const THRESHOLD: f64 = 1000.0; // ~half the smooth column passes

fn main() {
    divan::main();
}

#[divan::bench(name = "count_gt/fused")]
fn count_fused(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| scan::fused_count_gt(&enc, THRESHOLD));
}

#[divan::bench(name = "count_gt/materialized")]
fn count_materialized(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| scan::materialized_count_gt(&enc, THRESHOLD));
}

#[divan::bench(name = "mask_gt/fused")]
fn mask_fused(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| scan::fused_mask_gt(&enc, THRESHOLD));
}

#[divan::bench(name = "mask_gt/materialized")]
fn mask_materialized(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| scan::materialized_mask_gt(&enc, THRESHOLD));
}
