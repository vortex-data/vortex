// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused decode vs Vortex's `execute::<RecursiveCanonical>` across array sizes,
//! to show how the fusion advantage scales as the column grows past cache.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench sweep
//! ```
//!
//! Stack B `alp(delta(ffor(bitpacking)))`, `f64`. Each size is in elements;
//! the decoded column is `8 * elements` bytes.

use divan::Bencher;
use divan::counter::ItemsCount;
use simd_stencil::encode::encode_b;
use simd_stencil::encode::gen_f64;
use simd_stencil::strategies::fused;
use simd_stencil::vortex_baseline;

const EXP: i32 = 2;

// 8 KiB → 64 MiB of decoded f64 (elements = bytes / 8).
const SIZES: [usize; 7] = [
    1 << 10, // 8 KB
    1 << 13, // 64 KB
    1 << 16, // 512 KB
    1 << 18, // 2 MB
    1 << 20, // 8 MB
    1 << 22, // 32 MB
    1 << 23, // 64 MB
];

fn main() {
    divan::main();
}

#[divan::bench(args = SIZES)]
fn fully_decompressed(bencher: Bencher, n: usize) {
    // Floor: just allocate + copy the canonical values. Every decoder must at
    // least write this f64 output, so it caps the achievable speedup.
    let values = gen_f64(n, EXP, 2);
    bencher.counter(ItemsCount::new(n)).bench(|| values.clone());
}

#[divan::bench(args = SIZES)]
fn fused(bencher: Bencher, n: usize) {
    let enc = encode_b(&gen_f64(n, EXP, 2), EXP);
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| fused::decode_b(&enc));
}

#[divan::bench(args = SIZES)]
fn vortex_canonical(bencher: Bencher, n: usize) {
    let arr = vortex_baseline::build_b_full_same_stack(&gen_f64(n, EXP, 2));
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| vortex_baseline::decode_canonical(&arr));
}
