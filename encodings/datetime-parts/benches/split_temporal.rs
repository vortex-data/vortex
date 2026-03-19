// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_datetime_parts::split_temporal;

fn main() {
    divan::main();
}

fn make_temporal(n: usize) -> TemporalArray {
    // Timestamps in microseconds, spanning several days.
    let data: Buffer<i64> = (0..n as i64)
        .map(|v| v * 1_000_000 + 86_400_000_000)
        .collect();
    let array = PrimitiveArray::new(data, Validity::NonNullable).into_array();
    TemporalArray::new_timestamp(array, TimeUnit::Microseconds, None)
}

#[divan::bench(args = [1024, 65_536, 1_048_576])]
fn split_temporal_bench(bencher: Bencher, n: usize) {
    let temporal = make_temporal(n);
    bencher
        .with_inputs(|| temporal.clone())
        .bench_values(|t| split_temporal(t).unwrap())
}
