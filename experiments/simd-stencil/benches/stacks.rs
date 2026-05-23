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

// Floor baseline: the data is already fully decompressed; "decode" is just
// allocating the output and copying the canonical values (memory-bandwidth bound).
// No technique can beat this — it is the lower bound every decoder pays.

#[divan::bench(name = "a_delta_bitpack/fully_decompressed")]
fn a_floor(bencher: Bencher) {
    let values = gen_u32(N, 1);
    bencher.counter(ItemsCount::new(N)).bench(|| values.clone());
}

#[divan::bench(name = "b_alp_delta_for_bitpack/fully_decompressed")]
fn b_floor(bencher: Bencher) {
    let values = gen_f64(N, EXP, 2);
    bencher.counter(ItemsCount::new(N)).bench(|| values.clone());
}

// ---------------------------------------------------------------- stack A: delta(bitpacking) u32

#[divan::bench(name = "a_delta_bitpack/vortex_shallow")]
fn a_vortex_shallow(bencher: Bencher) {
    // Vortex's own compressor leaves the deltas uncompressed (no bitpacking).
    let arr = vortex_baseline::build_a(&gen_u32(N, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
}

#[divan::bench(name = "a_delta_bitpack/vortex_same_stack")]
fn a_vortex_same_stack(bencher: Bencher) {
    // Genuine delta(bitpacking): the same stack the prototype decodes.
    let arr = vortex_baseline::build_a_same_stack(&gen_u32(N, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
}

#[divan::bench(name = "a_delta_bitpack/vortex_canonical")]
fn a_vortex_canonical(bencher: Bencher) {
    // All layers in Vortex, ending in execute::<RecursiveCanonical>.
    let arr = vortex_baseline::build_a_same_stack(&gen_u32(N, 1));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode_canonical(&arr));
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

#[divan::bench(name = "b_alp_delta_for_bitpack/vortex_regular")]
fn b_vortex(bencher: Bencher) {
    // Vortex's own ALP encoding (shallow: ALP over an uncompressed/bit-packed child).
    let arr = vortex_baseline::build_b(&gen_f64(N, EXP, 2));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
}

#[divan::bench(name = "b_alp_delta_for_bitpack/vortex_same_stack")]
fn b_vortex_same_stack(bencher: Bencher) {
    // Genuine alp(delta(ffor(bitpacking))): the same full stack, decoded per-layer.
    let arr = vortex_baseline::build_b_full_same_stack(&gen_f64(N, EXP, 2));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
}

#[divan::bench(name = "b_alp_delta_for_bitpack/vortex_canonical")]
fn b_vortex_canonical(bencher: Bencher) {
    // All layers in Vortex, ending in execute::<RecursiveCanonical>.
    let arr = vortex_baseline::build_b_full_same_stack(&gen_f64(N, EXP, 2));
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode_canonical(&arr));
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

// ------------------ stack B integer core: delta(ffor(bitpacking)) -> i64, same-work vs Vortex

#[divan::bench(name = "b_core_delta_for_bitpack/vortex_same_stack")]
fn b_core_vortex(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    let arr = vortex_baseline::build_b_core_same_stack(&enc.digits);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
}

#[divan::bench(name = "b_core_delta_for_bitpack/fused")]
fn b_core_fused(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| fused::decode_b_core(&enc));
}

#[divan::bench(name = "b_core_delta_for_bitpack/aot")]
fn b_core_aot(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| aot::decode_b_core(&enc));
}

#[divan::bench(name = "b_core_delta_for_bitpack/vortex_shallow")]
fn b_core_vortex_shallow(bencher: Bencher) {
    // Regular Vortex: its Delta encoder leaves the digit-deltas uncompressed.
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    let arr = vortex_baseline::build_b_core_shallow(&enc.digits);
    bencher
        .counter(ItemsCount::new(N))
        .bench(|| vortex_baseline::decode(&arr));
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

#[divan::bench(name = "c_rle_alp_delta_for_bitpack/vortex_regular")]
fn c_vortex_regular(bencher: Bencher) {
    // Regular Vortex: RunEnd encoding of the logical column.
    let enc = encode_c(N_RUNS, EXP, 11);
    let arr = vortex_baseline::build_c_regular(&enc.values);
    bencher
        .counter(ItemsCount::new(enc.n_logical))
        .bench(|| vortex_baseline::decode(&arr));
}
