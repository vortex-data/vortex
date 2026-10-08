// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

//! Scalar construction and validation, stats reads and merges, and accumulator creation.
//!
//! Every benchmark does `N` operations per iteration, except the stats merge, which does `MERGES`.

use std::hint::black_box;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::array_session;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::stats::StatsSet;
use vortex_buffer::Buffer;
use vortex_buffer::BufferString;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const N: usize = 1024;
/// A merge costs about 300ns, so it runs fewer times to keep the CodSpeed simulation under 1ms.
const MERGES: usize = 64;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn struct8_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            FieldNames::from(["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]),
            vec![DType::Primitive(PType::I64, Nullability::NonNullable); 8],
        ),
        Nullability::NonNullable,
    )
}

fn struct8_value(i: i64) -> ScalarValue {
    ScalarValue::Tuple(
        (0..8)
            .map(|f| Some(ScalarValue::Primitive(PValue::I64(i + f))))
            .collect(),
    )
}

#[divan::bench]
fn try_new_primitive(bencher: Bencher) {
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for i in 0..N {
            black_box(
                Scalar::try_new(
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                    Some(ScalarValue::Primitive(PValue::I64(black_box(i as i64)))),
                )
                .unwrap(),
            );
        }
    });
}

#[divan::bench]
fn try_new_null(bencher: Bencher) {
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(
                Scalar::try_new(black_box(DType::Utf8(Nullability::Nullable)), None).unwrap(),
            );
        }
    });
}

#[divan::bench]
fn try_new_utf8(bencher: Bencher) {
    let value = BufferString::from("hello world, a string");
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(
                Scalar::try_new(
                    DType::Utf8(Nullability::NonNullable),
                    Some(ScalarValue::Utf8(black_box(&value).clone())),
                )
                .unwrap(),
            );
        }
    });
}

#[divan::bench]
fn try_new_struct8(bencher: Bencher) {
    let dtype = struct8_dtype();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for i in 0..N {
            black_box(Scalar::try_new(dtype.clone(), Some(struct8_value(i as i64))).unwrap());
        }
    });
}

#[divan::bench]
fn validate_struct8(bencher: Bencher) {
    let dtype = struct8_dtype();
    let value = struct8_value(0);
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(Scalar::validate(black_box(&dtype), Some(black_box(&value)))).unwrap();
        }
    });
}

fn i64_stats(offset: i64) -> StatsSet {
    StatsSet::from_iter([
        (Stat::Min, Precision::exact(ScalarValue::from(offset))),
        (Stat::Max, Precision::exact(ScalarValue::from(offset + 100))),
        (Stat::Sum, Precision::exact(ScalarValue::from(offset * 7))),
        (Stat::NullCount, Precision::exact(ScalarValue::from(3u64))),
        (Stat::IsConstant, Precision::exact(ScalarValue::from(false))),
        (Stat::IsSorted, Precision::exact(ScalarValue::from(true))),
        (
            Stat::IsStrictSorted,
            Precision::exact(ScalarValue::from(true)),
        ),
        (
            Stat::UncompressedSizeInBytes,
            Precision::exact(ScalarValue::from(800u64)),
        ),
    ])
}

#[divan::bench]
fn stats_get_min(bencher: Bencher) {
    let array = (0..1024i64).collect::<Buffer<i64>>().into_array();
    array.statistics().set_iter(i64_stats(0).into_iter());
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(black_box(&array).statistics().get(Stat::Min));
        }
    });
}

#[divan::bench]
fn stats_compute_is_constant_cached(bencher: Bencher) {
    let array = (0..1024i64).collect::<Buffer<i64>>().into_array();
    array.statistics().set_iter(i64_stats(0).into_iter());
    let mut ctx = SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(black_box(&array).statistics().compute_is_constant(&mut ctx));
        }
    });
}

#[divan::bench]
fn stats_merge_ordered(bencher: Bencher) {
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let left = i64_stats(0);
    let right = i64_stats(1000);
    bencher
        .counter(ItemsCount::new(MERGES))
        .with_inputs(|| vec![left.clone(); MERGES])
        .bench_local_values(|sets| {
            for set in sets {
                black_box(set.merge_ordered(&right, &dtype));
            }
        });
}

#[divan::bench]
fn accumulator_sum_v2(bencher: Bencher) {
    let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    let aggregate_fn = SumV2.bind(NumericalAggregateOpts::default());
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(aggregate_fn.accumulator(black_box(&dtype)).unwrap());
        }
    });
}

#[divan::bench]
fn accumulator_count_struct8(bencher: Bencher) {
    let dtype = struct8_dtype();
    let aggregate_fn = Count.bind(NumericalAggregateOpts::default());
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for _ in 0..N {
            black_box(aggregate_fn.accumulator(black_box(&dtype)).unwrap());
        }
    });
}
