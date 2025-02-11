#![allow(clippy::unwrap_used)]

use std::iter::Iterator;

use criterion::{criterion_group, criterion_main, Criterion};
use itertools::repeat_n;
use vortex_array::array::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{IntoArray, IntoCanonical};
use vortex_buffer::Buffer;
use vortex_runend::compress::runend_encode;
use vortex_runend::RunEndArray;

const LENS: [usize; 2] = [1000, 10_000];

fn run_end_compress(c: &mut Criterion) {
    evenly_spaced(c);
}

/// Create RunEnd arrays where the runs are equal size.
fn evenly_spaced(c: &mut Criterion) {
    let mut group = c.benchmark_group("run end array");

    for &n in LENS.iter().rev() {
        for run_step in [1usize << 2, 1 << 4, 1 << 8, 1 << 16] {
            let run_count = (0..=n).step_by(run_step).collect::<Vec<_>>().len();
            let values = PrimitiveArray::new(
                (0..=n)
                    .step_by(run_step)
                    .enumerate()
                    .flat_map(|(idx, x)| repeat_n(idx as u64, x))
                    .collect::<Buffer<_>>(),
                Validity::NonNullable,
            );
            group.bench_function(
                format!("compress n: {}, run_count: {}", n, run_count),
                |b| {
                    b.iter_with_setup(|| values.clone(), |values| runend_encode(&values).unwrap());
                },
            );

            group.bench_function(
                format!("decompress n: {}, run_count: {}", n, run_count),
                |b| {
                    let (ends, values) = runend_encode(&values).unwrap();
                    b.iter_with_setup(
                        || RunEndArray::try_new(ends.clone().into_array(), values.clone()).unwrap(),
                        |runend_array| runend_array.into_canonical().unwrap(),
                    );
                },
            );
        }
    }
}

criterion_group!(benches, run_end_compress);
criterion_main!(benches);
