// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Body-stitched copy-and-patch vs AOT vs per-op, for a chained elementwise
//! (affine) decode tail. Shows that stitching op *bodies* into one loop reaches
//! AOT throughput, while emitting one stencil per op (the `per_op` baseline,
//! which materialises between ops) lags well behind.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench stitch
//! ```

use divan::Bencher;
use divan::counter::ItemsCount;
use simd_stencil::stitched::affine_aot;
use simd_stencil::stitched::affine_per_op;

const N: usize = 1 << 20;
const TILE: usize = 1024;

fn main() {
    divan::main();
}

/// A 6-op affine tail — enough chained ops that per-op materialization hurts.
fn ops() -> Vec<(f64, f64)> {
    vec![
        (1.5, -3.25),
        (0.5, 100.0),
        (2.0, 0.125),
        (-1.0, 7.0),
        (1.25, -0.5),
        (0.875, 4.0),
    ]
}

fn input() -> Vec<f64> {
    (0..N).map(|i| (i as f64) * 0.013 - 7.0).collect()
}

/// Fair AOT upper bound: the 6 ops baked in as constants, so LLVM unrolls and
/// vectorizes the whole tail into one AVX-512 pass (every combination compiled
/// ahead of time).
#[inline(always)]
fn affine_aot_const(src: &[f64], dst: &mut [f64]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let mut x = *s;
        x = x.mul_add(1.5, -3.25);
        x = x.mul_add(0.5, 100.0);
        x = x.mul_add(2.0, 0.125);
        x = x.mul_add(-1.0, 7.0);
        x = x.mul_add(1.25, -0.5);
        x = x.mul_add(0.875, 4.0);
        *d = x;
    }
}

#[divan::bench(name = "affine_tail/aot_const")]
fn aot_const(bencher: Bencher) {
    let src = input();
    bencher.counter(ItemsCount::new(N)).bench(|| {
        let mut dst = vec![0f64; N];
        affine_aot_const(&src, &mut dst);
        dst
    });
}

/// Naive interpreter: ops in a runtime slice, so the compiler cannot vectorize
/// the per-element fold. Shows why "just loop over the plan" is not enough.
#[divan::bench(name = "affine_tail/aot_dynamic")]
fn aot_dynamic(bencher: Bencher) {
    let ops = ops();
    let src = input();
    bencher.counter(ItemsCount::new(N)).bench(|| {
        let mut dst = vec![0f64; N];
        affine_aot(&ops, &src, &mut dst);
        dst
    });
}

#[divan::bench(name = "affine_tail/per_op_materialized")]
fn per_op(bencher: Bencher) {
    let ops = ops();
    let src = input();
    bencher.counter(ItemsCount::new(N)).bench(|| {
        let mut dst = vec![0f64; N];
        affine_per_op(&ops, &src, &mut dst);
        dst
    });
}

#[cfg(all(target_arch = "x86_64", unix))]
#[divan::bench(name = "affine_tail/stitched")]
fn stitched(bencher: Bencher) {
    use simd_stencil::stitched::StitchedAffine;

    let ops = ops();
    let src = input();
    bencher.counter(ItemsCount::new(N)).bench(|| {
        // Build the stitched pipeline (the "JIT") then run it tile-by-tile.
        let pipe = StitchedAffine::build(&ops);
        let mut dst = vec![0f64; N];
        for t in 0..(N / TILE) {
            // SAFETY: each chunk is exactly one 1024-element tile.
            unsafe {
                pipe.run_tile(src[t * TILE..].as_ptr(), dst[t * TILE..].as_mut_ptr());
            }
        }
        dst
    });
}
