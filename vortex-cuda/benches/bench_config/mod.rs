// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::time::Duration;

use criterion::Criterion;

/// Returns a [`Criterion`] configuration tuned for CUDA benchmarks.
///
/// On CI (`CI` env var is set), uses more samples and a real warmup to
/// produce stable results for CodSpeed. Locally, keeps the minimal
/// config so `cargo bench` stays fast.
pub(super) fn cuda_bench_config() -> Criterion {
    if std::env::var("CI").is_ok() {
        Criterion::default()
            .without_plots()
            .sample_size(25)
            .warm_up_time(Duration::from_secs(3))
            .measurement_time(Duration::from_secs(5))
            .nresamples(100)
    } else {
        Criterion::default()
            .without_plots()
            .sample_size(10)
            .warm_up_time(Duration::from_nanos(1))
            .measurement_time(Duration::from_nanos(1))
            .nresamples(10)
    }
}
