#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_runend::RunEndArray;
use vortex_runend::_benchmarking::{filter_run_end, take_indices_unchecked};

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize, f64)] = &[
    // length, run_step, filter_density
    (1000, 4, 0.005),
    (1000, 4, 0.01),
    (1000, 4, 0.03),
    (1000, 16, 0.005),
    (1000, 16, 0.01),
    (1000, 16, 0.03),
    (1000, 256, 0.005),
    (1000, 256, 0.01),
    (1000, 256, 0.03),
    (10_000, 4, 0.005),
    (10_000, 4, 0.01),
    (10_000, 4, 0.03),
    (10_000, 16, 0.005),
    (10_000, 16, 0.01),
    (10_000, 16, 0.03),
    (10_000, 256, 0.005),
    (10_000, 256, 0.01),
    (10_000, 256, 0.03),
];

#[divan::bench(args = BENCH_ARGS)]
fn take_indices(bencher: Bencher, (n, run_step, filter_density): (usize, usize, f64)) {
    let (array, mask) = fixture(n, run_step, filter_density).unwrap();

    let indices = mask.values().unwrap().indices();

    bencher
        .with_inputs(|| (&array, indices))
        .bench_refs(|(array, indices)| take_indices_unchecked(array, indices).unwrap());
}

#[divan::bench(args = BENCH_ARGS)]
fn filter_runend(bencher: Bencher, (n, run_step, filter_density): (usize, usize, f64)) {
    let (array, mask) = fixture(n, run_step, filter_density).unwrap();

    bencher
        .with_inputs(|| (&array, &mask))
        .bench_refs(|(array, mask)| filter_run_end(array, mask).unwrap());
}

fn fixture(n: usize, run_step: usize, filter_density: f64) -> Option<(RunEndArray, Mask)> {
    let ends = (0..=n)
        .step_by(run_step)
        .map(|x| x as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let values = (0..ends.len())
        .map(|x| x as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let array = RunEndArray::try_new(ends, values).unwrap();

    let mask = Mask::from_indices(
        array.len(),
        (0..array.len())
            .take((filter_density * array.len() as f64).round() as usize)
            .collect(),
    );

    if mask.true_count() == 0 {
        return None;
    }

    Some((array, mask))
}
