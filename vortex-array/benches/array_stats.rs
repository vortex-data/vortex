// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the cost of array stats storage: creation, reads, writes and sharing.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::ScalarValue;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const N: usize = 1024;
const LEN: usize = 64;
const THREADS: &[usize] = &[1, 4, 16];

fn buffer() -> Buffer<u32> {
    Buffer::from_iter(0..LEN as u32)
}

fn primitive(buf: &Buffer<u32>) -> ArrayRef {
    PrimitiveArray::new(buf.clone(), Validity::NonNullable).into_array()
}

/// Stats known at construction, e.g. from a builder or a file.
fn seed(stats: usize) -> StatsSet {
    [
        (Stat::Min, Precision::exact(ScalarValue::from(0u32))),
        (Stat::Max, Precision::exact(ScalarValue::from(63u32))),
        (Stat::NullCount, Precision::exact(ScalarValue::from(0u64))),
        (Stat::IsSorted, Precision::exact(ScalarValue::from(true))),
    ]
    .into_iter()
    .take(stats)
    .collect()
}

fn with_min(buf: &Buffer<u32>) -> ArrayRef {
    primitive(buf).with_stats_set(seed(1))
}

/// Array creation and drop, no stats touched.
#[divan::bench]
fn create(bencher: Bencher) {
    let buf = buffer();
    bencher
        .counter(ItemsCount::new(N))
        .with_inputs(|| Vec::<ArrayRef>::with_capacity(N))
        .bench_refs(|out| {
            out.clear();
            out.extend((0..N).map(|_| primitive(&buf)));
        });
}

/// Slicing creates a new array per slice.
#[divan::bench]
fn slice(bencher: Bencher) {
    let array = PrimitiveArray::from_iter(0..(N * LEN) as u32).into_array();
    bencher
        .counter(ItemsCount::new(N))
        .with_inputs(|| Vec::<ArrayRef>::with_capacity(N))
        .bench_refs(|out| {
            out.clear();
            out.extend((0..N).map(|i| array.slice(i * LEN..(i + 1) * LEN).unwrap()));
        });
}

/// Read a stat that was never set.
#[divan::bench]
fn get_absent(bencher: Bencher) {
    let array = primitive(&buffer());
    bencher.counter(ItemsCount::new(N)).bench(|| {
        for _ in 0..N {
            divan::black_box(
                array
                    .statistics()
                    .get_cached(divan::black_box(Stat::Max).aggregate_fn()),
            );
        }
    });
}

/// Read a stat that was set, from many threads at once.
#[divan::bench(threads = THREADS)]
fn get_present(bencher: Bencher) {
    let array = with_min(&buffer());
    bencher.counter(ItemsCount::new(N)).bench(|| {
        for _ in 0..N {
            divan::black_box(
                array
                    .statistics()
                    .get_cached(divan::black_box(Stat::Min).aggregate_fn()),
            );
        }
    });
}

/// Create an array and seed `stats` stats on it.
#[divan::bench(args = [0, 1, 2, 4])]
fn create_seeded(bencher: Bencher, stats: usize) {
    let buf = buffer();
    bencher
        .counter(ItemsCount::new(N))
        .with_inputs(|| {
            let seeds = (0..N).map(|_| seed(stats)).collect::<Vec<_>>();
            (seeds, Vec::<ArrayRef>::with_capacity(N))
        })
        .bench_refs(|(seeds, out)| {
            out.extend(
                seeds
                    .drain(..)
                    .map(|stats| primitive(&buf).with_stats_set(stats)),
            );
        });
}

/// Create an array and compute min through the kernel, which caches the result.
#[divan::bench]
fn create_compute_min(bencher: Bencher) {
    let buf = buffer();
    let session = array_session();
    bencher
        .counter(ItemsCount::new(N))
        .with_inputs(|| session.create_execution_ctx())
        .bench_refs(|ctx| {
            for _ in 0..N {
                let array = primitive(&buf);
                divan::black_box(
                    array
                        .statistics()
                        .get_as::<u32>(Stat::Min.aggregate_fn(), ctx),
                );
            }
        });
}

/// Canonicalize a chunked array of many small chunks through the executor.
#[divan::bench]
fn chunked_canonical(bencher: Bencher) {
    let buf = buffer();
    let chunks: Vec<ArrayRef> = (0..N).map(|_| primitive(&buf)).collect();
    let dtype = chunks[0].dtype().clone();
    let session = array_session();
    bencher
        .with_inputs(|| {
            (
                ChunkedArray::try_new(chunks.clone(), dtype.clone())
                    .unwrap()
                    .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<RecursiveCanonical>(&mut ctx).unwrap());
}
