// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::stats::Stat;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    LazyLock::force(&VALUES);
    LazyLock::force(&LEADING_NULLS);
    LazyLock::force(&SCATTERED_NULLS);
    divan::main();
}

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const N: usize = 15_000;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// The statistics a writer computes for a zone of an integer column.
const STATS: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::Sum,
    Stat::NullCount,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
];

/// Sorted values with duplicates, so that no statistic finishes early.
static VALUES: LazyLock<Buffer<i64>> = LazyLock::new(|| {
    let mut rng = StdRng::seed_from_u64(1);
    let mut values: Vec<i64> = (0..N).map(|_| rng.random_range(0..1_000_000)).collect();
    values.sort_unstable();
    Buffer::from(values)
});

/// The leading tenth of the values is null, so that the values stay sorted.
static LEADING_NULLS: LazyLock<Validity> =
    LazyLock::new(|| Validity::from_iter((0..N).map(|i| i >= N / 10)));

/// A tenth of the values is null, at random.
static SCATTERED_NULLS: LazyLock<Validity> = LazyLock::new(|| {
    let mut rng = StdRng::seed_from_u64(2);
    Validity::from_iter((0..N).map(|_| rng.random_bool(0.9)))
});

/// Returns a new array, without cached statistics, that shares the values and validity.
fn array(validity: &Validity) -> ArrayRef {
    PrimitiveArray::new(VALUES.clone(), validity.clone()).into_array()
}

fn bench(bencher: Bencher, validity: &'static Validity, stats: &'static [Stat]) {
    bencher
        .with_inputs(|| (array(validity), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| a.statistics().compute_all(stats, ctx).unwrap());
}

#[divan::bench]
fn all_stats_non_null(bencher: Bencher) {
    bench(bencher, &Validity::NonNullable, STATS);
}

#[divan::bench]
fn all_stats_nullable(bencher: Bencher) {
    bench(bencher, &LEADING_NULLS, STATS);
}

#[divan::bench]
fn all_stats_scattered_nulls(bencher: Bencher) {
    bench(bencher, &SCATTERED_NULLS, STATS);
}

#[divan::bench]
fn bounds_and_sum_scattered_nulls(bencher: Bencher) {
    bench(
        bencher,
        &SCATTERED_NULLS,
        &[Stat::Min, Stat::Max, Stat::Sum],
    );
}

#[divan::bench]
fn min_max_scattered_nulls(bencher: Bencher) {
    bench(bencher, &SCATTERED_NULLS, &[Stat::Min, Stat::Max]);
}

#[divan::bench]
fn is_sorted_nullable(bencher: Bencher) {
    bench(bencher, &LEADING_NULLS, &[Stat::IsSorted]);
}

#[divan::bench(args = [Stat::Min, Stat::Sum, Stat::IsSorted, Stat::IsStrictSorted, Stat::IsConstant])]
fn single_non_null(bencher: Bencher, stat: Stat) {
    bencher
        .with_inputs(|| {
            (
                array(&Validity::NonNullable),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_all(&[stat], ctx).unwrap());
}
