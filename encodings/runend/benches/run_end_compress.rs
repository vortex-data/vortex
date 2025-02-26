#![allow(clippy::unwrap_used)]

use divan::Bencher;
use itertools::repeat_n;
use vortex_array::Array;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEndArray;
use vortex_runend::compress::runend_encode;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    (1000, 4),
    (1000, 16),
    (1000, 256),
    (10_000, 4),
    (10_000, 16),
    (10_000, 256),
];

#[divan::bench(args = BENCH_ARGS)]
fn compress(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..=length)
            .step_by(run_step)
            .enumerate()
            .flat_map(|(idx, x)| repeat_n(idx as u64, x))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    bencher
        .with_inputs(|| values.clone())
        .bench_refs(|values| runend_encode(values).unwrap());
}

#[divan::bench(args = BENCH_ARGS)]
fn decompress(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..=length)
            .step_by(run_step)
            .enumerate()
            .flat_map(|(idx, x)| repeat_n(idx as u64, x))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );
    let (ends, values) = runend_encode(&values).unwrap();
    let runend_array = RunEndArray::try_new(ends.into_array(), values).unwrap();

    bencher
        .with_inputs(|| runend_array.to_array())
        .bench_values(|array| array.to_canonical().unwrap());
}

#[divan::bench(args = BENCH_ARGS)]
fn take_from_primitive(bencher: Bencher, (array_size, run_length): (usize, usize)) {
    let source_array = PrimitiveArray::from_iter(0..array_size as i32).into_array();

    // Create run-end indices that select values with uniform run lengths
    let num_runs = array_size / run_length;
    let runs = (0..num_runs).collect::<Vec<_>>();

    // Create ends array - each run is run_length long
    let ends = runs
        .iter()
        .map(|&i| ((i + 1) * run_length) as u64)
        .collect::<Vec<_>>();

    // Create values array - each run selects a value
    let values = runs
        .iter()
        .map(|&i| ((i * run_length) / 2) as u32)
        .collect::<Vec<_>>();

    let ends_array = PrimitiveArray::from_iter(ends).into_array();
    let values_array = PrimitiveArray::from_iter(values).into_array();

    let indices = RunEndArray::try_new(ends_array, values_array).unwrap();

    bencher
        .with_inputs(|| (&indices, &source_array))
        .bench_refs(|(indices, array)| take(indices, array).unwrap());
}
