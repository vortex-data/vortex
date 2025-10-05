// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use itertools::repeat_n;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dtype::IntegerPType;
use vortex_fastlanes::RLEArray;

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
        .with_inputs(|| values.clone())
        .bench_refs(|values| RLEArray::encode(values).unwrap());
}

#[divan::bench(types = [u8, u16, u32, u64], args = BENCH_ARGS)]
fn decompress<T: IntegerPType>(bencher: Bencher, (length, run_step): (usize, usize)) {
    let values = PrimitiveArray::new(
        (0..length)
            .step_by(run_step)
            .flat_map(|idx| {
                repeat_n(
                    T::from(idx % T::max_value().to_usize().unwrap()).unwrap(),
                    run_step,
                )
            })
            .collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let rle_array = RLEArray::encode(&values).unwrap();

    bencher
        .with_inputs(|| rle_array.clone())
        .bench_values(|rle_array| rle_array.to_canonical());
}
