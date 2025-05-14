#![allow(clippy::unwrap_used)]

use divan::Bencher;
use itertools::repeat_n;
use num_traits::PrimInt;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::take;
use vortex_array::validity::Validity;
use vortex_array::{Array, IntoArray};
use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
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
        (0..length)
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

#[divan::bench(types = [u8, u16, u32, u64], args = BENCH_ARGS)]
fn decompress<T: NativePType + PrimInt>(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .enumerate()
            .flat_map(|(idx, x)| {
                repeat_n(
                    T::from(idx % T::max_value().to_usize().unwrap()).unwrap(),
                    x,
                )
            })
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
#[allow(clippy::cast_possible_truncation)]
fn take_indices(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .enumerate()
            .flat_map(|(idx, x)| repeat_n(idx as u64, x))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let source_array = PrimitiveArray::from_iter(0..(length as i32)).into_array();
    let (ends, values) = runend_encode(&values).unwrap();
    let runend_array = RunEndArray::try_new(ends.into_array(), values).unwrap();

    bencher
        .with_inputs(|| (source_array.clone(), runend_array.to_array()))
        .bench_refs(|(array, indices)| take(array, indices).unwrap());
}
