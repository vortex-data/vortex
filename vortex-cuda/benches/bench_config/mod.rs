// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::time::Duration;

use criterion::Criterion;

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
/// `warm_up_time` runs at least one full iteration before sampling, giving
/// the GPU a chance to reach steady state (clock boost, cache warming).
/// If a single launch exceeds the warm-up budget, criterion still completes
/// it before moving on.
pub(super) fn cuda_bench_config() -> Criterion {
    // Number of independent kernel launches.
    let sample_size = 10;

    Criterion::default()
        .without_plots()
        .sample_size(sample_size)
        // One ns is enough to JIT-compile kernels and warm GPU caches.
        // Criterion always finishes the in-flight iteration even if this
        // budget is exceeded.
        .warm_up_time(Duration::from_nanos(1))
        // Forces `iters = 1`: criterion's planner estimates iteration cost
        // from wall time (which includes GPU context setup), not the
        // GPU-timed duration returned by `iter_custom`. A real
        // measurement_time would cause wildly inflated iteration counts.
        .measurement_time(Duration::from_nanos(1))
}
