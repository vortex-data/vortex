// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression statistics over arrays of each values width, with and without nulls.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::DecimalArray;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_compressor::stats::DecimalStats;
use vortex_compressor::stats::GenerateStatsOptions;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
});

const LEN: usize = 64 * 1024;

const WIDTHS: [DecimalType; 3] = [DecimalType::I64, DecimalType::I128, DecimalType::I256];

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

/// Runs of up to 8 values, drawn from 1024 distinct values, stored as `values_type`.
fn decimals(values_type: DecimalType, validity: Validity) -> DecimalArray {
    let mut rng = StdRng::seed_from_u64(0);
    let mut values = Vec::with_capacity(LEN);
    while values.len() < LEN {
        let value: i64 = rng.random_range(0..1024) * 1_000_003;
        let run = rng.random_range(1..=8);
        values.extend(std::iter::repeat_n(value, run.min(LEN - values.len())));
    }
    match_each_decimal_value_type!(values_type, |T| {
        let buffer = values
            .iter()
            .map(|&v| <T as BigCast>::from(v).unwrap())
            .collect::<Buffer<T>>();
        DecimalArray::new(buffer, DecimalDType::new(18, 2), validity)
    })
}

/// One value in ten null, at random, so almost every chunk of 64 values has a null.
fn sparse_nulls() -> Validity {
    let mut rng = StdRng::seed_from_u64(1);
    Validity::from(BitBuffer::from_iter((0..LEN).map(|_| rng.random_bool(0.9))))
}

/// Computes every statistic, as when the decimal dictionary scheme is enabled.
fn bench_stats(bencher: Bencher, values_type: DecimalType, validity: Validity) {
    let array = decimals(values_type, validity);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            DecimalStats::generate_opts(
                &array,
                GenerateStatsOptions {
                    count_distinct_values: true,
                },
                ctx,
            )
        });
}

#[divan::bench(args = WIDTHS)]
fn decimal_stats_non_null(bencher: Bencher, values_type: DecimalType) {
    bench_stats(bencher, values_type, Validity::NonNullable);
}

#[divan::bench(args = WIDTHS)]
fn decimal_stats_nullable(bencher: Bencher, values_type: DecimalType) {
    bench_stats(bencher, values_type, sparse_nulls());
}
