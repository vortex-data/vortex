#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng as _};
use vortex_array::array::PrimitiveArray;
use vortex_array::stats::Stat;
use vortex_array::IntoArrayData;
use vortex_buffer::Buffer;
use vortex_runend::RunEndArray;

const LENS: [usize; 2] = [1000, 100_000];

/// Create RunEnd arrays where the runs are equal size, and the null_count mask is evenly spaced.
fn run_end_null_count(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut group = c.benchmark_group("run_end_null_count");

    for &n in LENS.iter().rev() {
        for run_step in [1usize << 2, 1 << 4, 1 << 8, 1 << 16] {
            let ends = (0..=n)
                .step_by(run_step)
                .map(|x| x as u64)
                .collect::<Buffer<_>>()
                .into_array();
            let run_count = ends.len() - 1;
            for valid_density in [0.01, 0.1, 0.5] {
                let values = PrimitiveArray::from_option_iter(
                    (0..ends.len()).map(|x| rng.gen_bool(valid_density).then_some(x as u64)),
                )
                .into_array();
                let array = RunEndArray::try_new(ends.clone(), values)
                    .unwrap()
                    .into_array();

                group.bench_function(
                    format!(
                        "null_count_run_end n: {}, run_count: {}, valid_density: {}",
                        n, run_count, valid_density
                    ),
                    |b| {
                        b.iter(|| {
                            black_box(
                                array
                                    .encoding()
                                    .compute_statistics(&array, Stat::NullCount)
                                    .unwrap(),
                            )
                        });
                    },
                );
            }
        }
    }
}

criterion_group!(benches, run_end_null_count);
criterion_main!(benches);
