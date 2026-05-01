// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::time::Duration;

use criterion::Criterion;

/// Benchmark input sizes.
///
/// On codspeed, only the 100M variant runs — kernels under ~200 µs
/// (i.e. the 10M cases) swing 15-45% across ephemeral GPU instances,
/// drowning real regressions in noise. Locally both sizes run.
#[cfg(not(codspeed))]
pub const BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M"), (100_000_000, "100M")];
#[cfg(codspeed)]
pub const BENCH_SIZES: &[(usize, &str)] = &[(100_000_000, "100M")];

/// Returns a [`Criterion`] configuration tuned for CUDA benchmarks.
///
/// All benchmarks use `iter_custom` with precise CUDA event timing.
/// criterion's iteration planner estimates `iters` from **wall time** during
/// warmup, which includes GPU context setup and memory copies — not just
/// the kernel. Setting `measurement_time = 1ns` forces `iters = 1` so
/// each sample is exactly one `iter_custom` call returning GPU-timed duration.
/// Stability comes from a high `sample_size` (many independent launches)
/// rather than many iterations per sample.
///
/// `warm_up_time` is set to 500 ms — long enough to JIT-compile PTX,
/// boost GPU clocks, and warm caches, while keeping total runtime under
/// 2 minutes even for the largest benchmark binary (~18 benchmarks).
///
/// `sample_size` is 10: with 100M inputs the kernels are long enough
/// (>500 µs) that within-run variance is low. Cross-run stability
/// comes from the large input size, not from averaging many samples.
pub(super) fn cuda_bench_config() -> Criterion {
    let sample_size = 10;

    Criterion::default()
        .without_plots()
        .sample_size(sample_size)
        // Enough for PTX JIT, GPU clock boost, and cache warming.
        .warm_up_time(Duration::from_millis(500))
        // Forces `iters = 1`: criterion's planner estimates iteration cost
        // from wall time (which includes GPU context setup), not the
        // GPU-timed duration returned by `iter_custom`. A real
        // measurement_time would cause wildly inflated iteration counts.
        .measurement_time(Duration::from_nanos(1))
}
