// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::i256;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

// Sized to keep the CodSpeed simulation under 1ms per benchmark.
const N: usize = 15_000;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
});

#[divan::bench]
fn max_i32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);
    let data: Vec<i32> = (0..N).map(|_| rng.random::<i32>()).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i32>(ctx));
}

#[divan::bench]
fn max_i64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(2);
    let data: Vec<i64> = (0..N).map(|_| rng.random::<i64>()).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i64>(ctx));
}

#[derive(Clone, Copy, Debug)]
enum DecimalValidity {
    NonNullable,
    ScatteredNulls,
}

const DECIMAL_VALIDITIES: &[DecimalValidity] = &[
    DecimalValidity::NonNullable,
    DecimalValidity::ScatteredNulls,
];

fn bench_decimal_min_max<T: NativeDecimalType>(
    bencher: Bencher,
    values: Buffer<T>,
    decimal_dtype: DecimalDType,
    validity: DecimalValidity,
) {
    let validity = match validity {
        DecimalValidity::NonNullable => Validity::NonNullable,
        DecimalValidity::ScatteredNulls => {
            let mut rng = StdRng::seed_from_u64(8);
            Validity::from_iter((0..N).map(|_| rng.random_bool(0.5)))
        }
    };

    bencher
        .with_inputs(|| {
            (
                DecimalArray::new(values.clone(), decimal_dtype, validity.clone()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| min_max(array, ctx, NumericalAggregateOpts::default()));
}

#[divan::bench(args = DECIMAL_VALIDITIES)]
fn min_max_decimal_i128(bencher: Bencher, validity: DecimalValidity) {
    let mut rng = StdRng::seed_from_u64(9);
    let bound = 10i128.pow(38) - 1;
    let values = (0..N).map(|_| rng.random_range(-bound..=bound)).collect();

    bench_decimal_min_max(bencher, values, DecimalDType::new(38, 2), validity);
}

#[divan::bench(args = DECIMAL_VALIDITIES)]
fn min_max_decimal_i256(bencher: Bencher, validity: DecimalValidity) {
    let mut rng = StdRng::seed_from_u64(10);
    let factor = i256::from_i128(10i128.pow(38));
    let bound = 10i128.pow(37) - 1;
    let values = (0..N)
        .map(|_| i256::from_i128(rng.random_range(-bound..=bound)) * factor)
        .collect();

    bench_decimal_min_max(bencher, values, DecimalDType::new(76, 2), validity);
}

#[divan::bench]
fn max_f64(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(3);
    let data: Vec<f64> = (0..N).map(|_| rng.random::<f64>()).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<f64>(ctx));
}

// Clustered nulls: long valid runs broken up by null blocks (run-based path's best case).
#[divan::bench]
fn max_i32_nulls_clustered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(4);
    let data: Vec<Option<i32>> = (0..N)
        .map(|i| {
            if (i / 64) % 10 == 0 {
                None
            } else {
                Some(rng.random::<i32>())
            }
        })
        .collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i32>(ctx));
}

// Scattered nulls: ~50% random nulls producing many short runs (run-based path's worst case).
#[divan::bench]
fn max_i32_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(5);
    let data: Vec<Option<i32>> = (0..N)
        .map(|_| rng.random_bool(0.5).then(|| rng.random::<i32>()))
        .collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<i32>(ctx));
}

#[divan::bench]
fn max_f32(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(6);
    let data: Vec<f32> = (0..N).map(|_| rng.random::<f32>()).collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<f32>(ctx));
}

#[divan::bench]
fn max_f64_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(7);
    let data: Vec<Option<f64>> = (0..N)
        .map(|_| rng.random_bool(0.5).then(|| rng.random::<f64>()))
        .collect();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_option_iter(data.iter().copied()).into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| a.statistics().compute_max::<f64>(ctx));
}

// String extrema. Every view carries its value's first four bytes inline, so the shape that
// matters is how far into a value its order is settled: `distinct_prefixes` and
// `low_cardinality` never have to look past the view, while the log-shaped columns (every value
// starting `thread-`/`req-`/`This is `) tie on those bytes and have to resolve the value.
//
// Like the primitive benches, each iteration gets a fresh array: `min_max` caches `Stat::Min`
// and `Stat::Max` on the array it scanned, so a reused array would only be scanned once.

// Resolving and comparing string values costs far more per row than a primitive, so these scan a
// smaller column to keep each benchmark under 1ms in CodSpeed simulation.
const STRING_N: usize = 4_096;

fn bench_min_max(bencher: Bencher, values: Vec<Option<String>>) {
    bencher
        .with_inputs(|| {
            (
                VarBinViewArray::from_iter(
                    values.iter().map(|v| v.as_deref()),
                    DType::Utf8(Nullability::Nullable),
                )
                .into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| min_max(a, ctx, NumericalAggregateOpts::default()));
}

/// Long values with distinct leading bytes: the prefix rejects almost every element.
#[divan::bench]
fn max_utf8_view_distinct_prefixes(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(6);
    bench_min_max(
        bencher,
        (0..STRING_N)
            .map(|_| {
                Some(format!(
                    "{:016x}-{:016x}",
                    rng.random::<u64>(),
                    rng.random::<u64>()
                ))
            })
            .collect(),
    );
}

/// Log levels: short inlined values over a five-value dictionary, with distinct leading bytes.
#[divan::bench]
fn max_utf8_view_low_cardinality(bencher: Bencher) {
    bench_min_max(
        bencher,
        (0..STRING_N)
            .map(|i| {
                Some(
                    match i % 5 {
                        0 => "ERROR",
                        1 => "WARN",
                        2 => "INFO",
                        3 => "DEBUG",
                        _ => "TRACE",
                    }
                    .to_string(),
                )
            })
            .collect(),
    );
}

/// Thread names: short inlined values that tie on the inline prefix and are settled a few
/// bytes later.
#[divan::bench]
fn max_utf8_view_tied_prefix_inlined(bencher: Bencher) {
    bench_min_max(
        bencher,
        (0..STRING_N)
            .map(|i| Some(format!("thread-{}", i % 8)))
            .collect(),
    );
}

/// Request ids: 20-byte buffer-backed values that tie on the inline prefix `req-`.
#[divan::bench]
fn max_utf8_view_tied_prefix_outlined(bencher: Bencher) {
    bench_min_max(
        bencher,
        (0..STRING_N)
            .map(|i| Some(format!("req-{i:016x}")))
            .collect(),
    );
}

/// Log lines: long, buffer-backed values that agree for their first twenty-six bytes.
#[divan::bench]
fn max_utf8_view_shared_prefix(bencher: Bencher) {
    bench_min_max(
        bencher,
        (0..STRING_N)
            .map(|i| {
                Some(format!(
                    "This is log message number {i} with some additional content to simulate real logs"
                ))
            })
            .collect(),
    );
}

/// Distinct prefixes with ~50% random nulls, producing many short valid runs.
#[divan::bench]
fn max_utf8_view_nulls_scattered(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(7);
    bench_min_max(
        bencher,
        (0..STRING_N)
            .map(|_| {
                rng.random_bool(0.5)
                    .then(|| format!("{:016x}-{:016x}", rng.random::<u64>(), rng.random::<u64>()))
            })
            .collect(),
    );
}
