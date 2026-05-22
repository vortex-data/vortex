// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use rand::prelude::*;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::stats::Stat;

fn main() {
    divan::main();
}

const N: usize = 100_000;

#[divan::bench]
fn sum_i32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let data: Vec<i32> = (0..N).map(|_| rng.random_range(-1000..1000)).collect();
    bencher
        .with_inputs(|| PrimitiveArray::from_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_as::<i64>(Stat::Sum, &mut LEGACY_SESSION.create_execution_ctx())
        });
}

#[divan::bench]
fn sum_u32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(2);
    let data: Vec<u32> = (0..N).map(|_| rng.random_range(0..2000)).collect();
    bencher
        .with_inputs(|| PrimitiveArray::from_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_as::<u64>(Stat::Sum, &mut LEGACY_SESSION.create_execution_ctx())
        });
}

#[divan::bench]
fn sum_i64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(3);
    let data: Vec<i64> = (0..N).map(|_| rng.random_range(-1000..1000)).collect();
    bencher
        .with_inputs(|| PrimitiveArray::from_iter(data.iter().copied()).into_array())
        .bench_refs(|a| {
            a.statistics()
                .compute_as::<i64>(Stat::Sum, &mut LEGACY_SESSION.create_execution_ctx())
        });
}
