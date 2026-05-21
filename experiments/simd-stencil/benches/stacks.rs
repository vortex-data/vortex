// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode throughput per stack, across every composition strategy.
//!
//! Run with full SIMD codegen so the `fastlanes` kernels and the AOT/fused
//! Rust pipelines use AVX-512:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench stacks
//! ```

use divan::Bencher;
use divan::counter::ItemsCount;
use simd_stencil::encode::encode_a;
use simd_stencil::encode::encode_b;
use simd_stencil::encode::encode_c;
use simd_stencil::encode::gen_f64;
use simd_stencil::encode::gen_u32;
use simd_stencil::patched;
use simd_stencil::strategies::aot;
use simd_stencil::strategies::fused;
use simd_stencil::strategies::materialized;
use simd_stencil::vortex_baseline;

/// ~1M elements: a `u64` column is 8 MiB, well past L2, so full-column
/// materialization pays real memory traffic.
const N: usize = 1 << 20;
const EXP: i32 = 2;
const N_RUNS: usize = 1 << 17;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------- stack A: delta(bitpacking) u32

#[divan::bench(name = "a_delta_bitpack/vortex")]
fn a_vortex(bencher: Bencher) {
    let arr = vortex_baseline::build_a(&gen_u32(N, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
}

#[divan::bench(name = "a_delta_bitpack/materialized")]
fn a_materialized(bencher: Bencher) {
    let enc = encode_a(&gen_u32(N, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| materialized::decode_a(&enc));
}

#[divan::bench(name = "a_delta_bitpack/fused")]
fn a_fused(bencher: Bencher) {
    let enc = encode_a(&gen_u32(N, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| fused::decode_a(&enc));
}

#[divan::bench(name = "a_delta_bitpack/aot")]
fn a_aot(bencher: Bencher) {
    let enc = encode_a(&gen_u32(N, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| aot::decode_a(&enc));
}

// ------------------------------------------------ stack B: alp(delta(ffor(bitpacking))) f64

#[divan::bench(name = "b_alp_delta_for_bitpack/vortex")]
fn b_vortex(bencher: Bencher) {
    let arr = vortex_baseline::build_b(&gen_f64(N, EXP, 2));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
}

#[divan::bench(name = "b_alp_delta_for_bitpack/materialized")]
fn b_materialized(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| materialized::decode_b(&enc));
}

#[divan::bench(name = "b_alp_delta_for_bitpack/fused")]
fn b_fused(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| fused::decode_b(&enc));
}

#[divan::bench(name = "b_alp_delta_for_bitpack/patched")]
fn b_patched(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| patched::decode_b(&enc));
}

#[divan::bench(name = "b_alp_delta_for_bitpack/aot")]
fn b_aot(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| aot::decode_b(&enc));
}

// ------------------------------------------- stack C: rle(alp(delta(ffor(bitpacking)))) f64

#[divan::bench(name = "c_rle_alp_delta_for_bitpack/materialized")]
fn c_materialized(bencher: Bencher) {
    let enc = encode_c(N_RUNS, EXP, 11);
    bencher
        .counter(ItemsCount::new(enc.n_logical))
        .bench(|| materialized::decode_c(&enc));
}

#[divan::bench(name = "c_rle_alp_delta_for_bitpack/fused")]
fn c_fused(bencher: Bencher) {
    let enc = encode_c(N_RUNS, EXP, 11);
    bencher
        .counter(ItemsCount::new(enc.n_logical))
        .bench(|| fused::decode_c(&enc));
}

#[divan::bench(name = "c_rle_alp_delta_for_bitpack/patched")]
fn c_patched(bencher: Bencher) {
    let enc = encode_c(N_RUNS, EXP, 11);
    bencher
        .counter(ItemsCount::new(enc.n_logical))
        .bench(|| patched::decode_c(&enc));
}

#[divan::bench(name = "c_rle_alp_delta_for_bitpack/aot")]
fn c_aot(bencher: Bencher) {
    let enc = encode_c(N_RUNS, EXP, 11);
    bencher
        .counter(ItemsCount::new(enc.n_logical))
        .bench(|| aot::decode_c(&enc));
}
