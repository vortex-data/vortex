// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression statistics over arrays with and without nulls.

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_compressor::stats::FloatStats;
use vortex_compressor::stats::GenerateStatsOptions;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const LEN: usize = 64 * 1024;

fn main() {
    divan::main();
}

/// Runs of up to 8 values, drawn from 1024 distinct values.
fn values() -> Buffer<f64> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = Vec::with_capacity(LEN);
    while values.len() < LEN {
        let value: u32 = rng.random_range(0..1024);
        let run = rng.random_range(1..=8);
        values.extend(std::iter::repeat_n(
            f64::from(value),
            run.min(LEN - values.len()),
        ));
    }
    Buffer::from(values)
}

/// One value in ten null, at random, so almost every chunk of 64 values has a null.
fn sparse_nulls() -> Validity {
    let mut rng = StdRng::seed_from_u64(1);
    Validity::from(BitBuffer::from_iter((0..LEN).map(|_| rng.random_bool(0.9))))
}

/// Computes the statistics of a fresh array each iteration, so that the bounds cached on the array
/// by a previous iteration are not reused. With `count_distinct_values`, every statistic is
/// computed, as when a dictionary scheme is enabled.
fn bench_stats(bencher: Bencher, validity: Validity, count_distinct_values: bool) {
    let values = values();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::new(values.clone(), validity.clone()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| {
            FloatStats::generate_opts(
                array,
                GenerateStatsOptions {
                    count_distinct_values,
                },
                ctx,
            )
        });
}

#[divan::bench]
fn float_stats_non_null(bencher: Bencher) {
    bench_stats(bencher, Validity::NonNullable, false);
}

#[divan::bench]
fn float_stats_nullable(bencher: Bencher) {
    bench_stats(bencher, sparse_nulls(), false);
}

#[divan::bench]
fn float_stats_all_non_null(bencher: Bencher) {
    bench_stats(bencher, Validity::NonNullable, true);
}

#[divan::bench]
fn float_stats_all_nullable(bencher: Bencher) {
    bench_stats(bencher, sparse_nulls(), true);
}
