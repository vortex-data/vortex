// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! "JIT" build latency for the copy-and-patch path.
//!
//! The claim under test: emitting a specialized stencil at run time costs about
//! a `memcpy` (here, mmap + copy + 8-byte patch + mprotect), i.e. sub-microsecond,
//! versus the seconds an actual recompile would take to reach the same AOT
//! quality.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench dispatch
//! ```

fn main() {
    divan::main();
}

#[cfg(all(target_arch = "x86_64", unix))]
mod patched_build {
    use divan::Bencher;
    use divan::black_box;
    use simd_stencil::patched::AlpScaleStencil;

    /// Build one patched stencil: mmap a page, copy the template, patch the
    /// scale immediate, flip the page to executable.
    #[divan::bench(name = "build_patched_stencil")]
    fn build(bencher: Bencher) {
        bencher.bench(|| AlpScaleStencil::build(black_box(0.01)));
    }

    /// Build then run the stencil over a single 1024-element tile, the smallest
    /// useful unit of "compile + execute".
    #[divan::bench(name = "build_and_run_one_tile")]
    fn build_and_run(bencher: Bencher) {
        let digits = [12345i64; 1024];
        bencher.bench(|| {
            let mut out = [0f64; 1024];
            let stencil = AlpScaleStencil::build(black_box(0.01));
            // SAFETY: `digits` and `out` are both full 1024-element tiles.
            unsafe { stencil.run_tile(digits.as_ptr(), out.as_mut_ptr()) };
            black_box(out[0])
        });
    }
}
