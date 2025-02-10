#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use criterion::{criterion_group, criterion_main, Criterion};
use num_traits::ToPrimitive;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_runend::RunEndArray;
use vortex_runend::_benchmarking::{filter_run_end, take_indices_unchecked};

const LENS: [usize; 2] = [1000, 100_000];

fn run_end_filter(c: &mut Criterion) {
    evenly_spaced(c);
}

/// Create RunEnd arrays where the runs are equal size, and the filter mask is evenly spaced.
fn evenly_spaced(c: &mut Criterion) {
    let mut group = c.benchmark_group("evenly_spaced");

    for &n in LENS.iter().rev() {
        for run_step in [1usize << 2, 1 << 4, 1 << 8, 1 << 16] {
            let ends = (0..=n)
                .step_by(run_step)
                .map(|x| x as u64)
                .collect::<Buffer<_>>()
                .into_array();
            let run_count = ends.len() - 1;
            let values = (0..ends.len())
                .map(|x| x as u64)
                .collect::<Buffer<_>>()
                .into_array();
            let array = RunEndArray::try_new(ends, values).unwrap();

            for filter_density in [0.001, 0.01, 0.015, 0.020, 0.025, 0.030] {
                let mask = Mask::from_indices(
                    array.len(),
                    // In this case, the benchmarks don't seem to change whether we evenly spread
                    // the mask values or like here we pack them into the beginning of the mask.
                    (0..array.len())
                        .take(
                            (filter_density * array.len() as f64)
                                .round()
                                .to_usize()
                                .unwrap(),
                        )
                        .collect(),
                );

                if mask.true_count() == 0 {
                    // Can skip these
                    continue;
                }

                // Compute the ratio of true_count to run_count
                let ratio = mask.true_count() as f64 / run_count as f64;

                group.bench_function(
                    format!(
                        "take_indices n: {}, run_count: {}, true_count: {}, ratio: {}",
                        n,
                        run_count,
                        mask.true_count(),
                        ratio
                    ),
                    |b| {
                        b.iter(|| {
                            take_indices_unchecked(&array, mask.values().unwrap().indices())
                                .unwrap()
                        });
                    },
                );
                group.bench_function(
                    format!(
                        "filter_run_end n: {}, run_count: {}, true_count: {}, ratio: {}",
                        n,
                        run_count,
                        mask.true_count(),
                        ratio
                    ),
                    |b| {
                        b.iter(|| filter_run_end(&array, &mask).unwrap());
                    },
                );
            }
        }
    }
}

criterion_group!(benches, run_end_filter);
criterion_main!(benches);
