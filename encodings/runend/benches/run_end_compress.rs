// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use itertools::repeat_n;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::warm_up_vtables;
use vortex_array::dtype::IntegerPType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_runend::RunEndArray;
use vortex_runend::compress::runend_encode;

fn main() {
    warm_up_vtables();
    divan::main();
}

const BENCH_ARGS: &[(usize, usize)] = &[
    (1000, 4),
    (1000, 16),
    (1000, 256),
    (10_000, 4),
    (10_000, 16),
    (10_000, 256),
    (10_000, 1024),
    (100_000, 4),
    (100_000, 16),
    (100_000, 256),
    (100_000, 1024),
    (100_000, 4096),
    (1_000_000, 4),
    (1_000_000, 16),
    (1_000_000, 256),
    (1_000_000, 1024),
    (1_000_000, 4096),
    (1_000_000, 8192),
];

#[divan::bench(args = BENCH_ARGS)]
fn compress(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .flat_map(|idx| repeat_n(idx as u64, run_step))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    bencher
        .with_inputs(|| &values)
        .bench_refs(|values| runend_encode(values));
}

#[divan::bench(types = [u8, u16, u32, u64], args = BENCH_ARGS)]
fn decompress<T: IntegerPType>(bencher: Bencher, (length, run_step): (usize, usize)) {
    let ends = (0..=length)
        .step_by(run_step)
        .map(|x| x as u64)
        .collect::<Buffer<_>>()
        .into_array();

    let values = (0..ends.len())
        .map(|x| T::from(x % T::max_value().to_usize().unwrap()).unwrap())
        .collect::<Buffer<_>>()
        .into_array();

    let run_end_array = RunEndArray::new(ends, values);
    let array = run_end_array.to_array();

    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| array.to_canonical());
}

#[divan::bench(args = BENCH_ARGS)]
#[allow(clippy::cast_possible_truncation)]
fn take_indices(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .flat_map(|idx| repeat_n(idx as u64, run_step))
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let source_array = PrimitiveArray::from_iter(0..(length as i32)).into_array();
    let (ends, values) = runend_encode(&values);
    let runend_array = RunEndArray::try_new(ends.into_array(), values)
        .unwrap()
        .to_array();

    bencher
        .with_inputs(|| {
            (
                &source_array,
                &runend_array,
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, indices, execution_ctx)| {
            array
                .take(indices.to_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        });
}
