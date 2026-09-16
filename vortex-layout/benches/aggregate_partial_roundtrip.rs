// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for round-tripping aggregate partials, the way a zoned writer does.
//!
//! The writer accumulates one partial per zone and converts each to a scalar
//! (`partial_scalar`), and those scalars are later folded back into a single accumulator
//! (`combine_partials`). Both halves are timed separately, plus the whole round-trip, over
//! two partial shapes: a bloom filter, whose partial is a byte blob that grows with the
//! filter size, and `Sum`, whose partial is a single value so the cost is dispatch and
//! `Scalar` handling.

#![expect(clippy::expect_used)]

use std::num::NonZeroU32;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::BloomFilter;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::BloomOptions;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::HashFn;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

/// Zones merged per round-trip.
const ZONE_COUNT: usize = 16;
/// Rows per zone. Accumulation itself is setup, not part of what is timed here.
const ZONE_LEN: usize = 4096;
/// [Default, cache-unfriendly] block counts, i.e. 8KiB and 256KiB of bloom partial per zone.
const BLOCK_COUNTS: &[u32] = &[256, 8192];

fn input_dtype() -> DType {
    DType::Primitive(PType::I64, Nullability::NonNullable)
}

/// One zone of values, disjoint from every other zone so that merging keeps setting new bloom
/// bits rather than re-setting the same ones.
fn zone_array(zone: usize) -> ArrayRef {
    let start = (zone * ZONE_LEN) as i64;
    let values: Buffer<i64> = (start..start + ZONE_LEN as i64).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

fn bloom_accumulator(block_count: u32) -> AccumulatorRef {
    let options = BloomOptions::new(
        NonZeroU32::new(block_count).expect("benchmark block counts are non-zero"),
        HashFn::XxHash3_64,
    );
    Box::new(
        Accumulator::try_new(BloomFilter, options, input_dtype())
            .expect("bloom filter accepts i64 input"),
    )
}

fn sum_accumulator() -> AccumulatorRef {
    Box::new(
        Accumulator::try_new(Sum, NumericalAggregateOpts::default(), input_dtype())
            .expect("sum accepts i64 input"),
    )
}

/// One accumulator per zone, each holding that zone's partial, as the writer leaves them.
fn zone_partials(new_accumulator: impl Fn() -> AccumulatorRef) -> Vec<AccumulatorRef> {
    let mut ctx = SESSION.create_execution_ctx();
    (0..ZONE_COUNT)
        .map(|zone| {
            let mut accumulator = new_accumulator();
            accumulator
                .accumulate(&zone_array(zone), &mut ctx)
                .expect("accumulate zone");
            accumulator
        })
        .collect()
}

fn to_scalars(partials: &[AccumulatorRef]) -> Vec<Scalar> {
    partials
        .iter()
        .map(|partial| partial.partial_scalar().expect("partial scalar"))
        .collect()
}

fn merge_all(merged: &mut AccumulatorRef, scalars: &[Scalar]) {
    for scalar in scalars {
        merged
            .combine_partials(scalar.clone())
            .expect("combine partials");
    }
}

#[divan::bench(args = BLOCK_COUNTS)]
fn bloom_to_scalar(bencher: Bencher, block_count: u32) {
    let partials = zone_partials(|| bloom_accumulator(block_count));

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .bench_local(|| to_scalars(&partials));
}

#[divan::bench(args = BLOCK_COUNTS)]
fn bloom_merge_partials(bencher: Bencher, block_count: u32) {
    let scalars = to_scalars(&zone_partials(|| bloom_accumulator(block_count)));

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .with_inputs(|| bloom_accumulator(block_count))
        .bench_local_refs(|merged| merge_all(merged, &scalars));
}

#[divan::bench(args = BLOCK_COUNTS)]
fn bloom_roundtrip(bencher: Bencher, block_count: u32) {
    let partials = zone_partials(|| bloom_accumulator(block_count));

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .with_inputs(|| bloom_accumulator(block_count))
        .bench_local_refs(|merged| merge_all(merged, &to_scalars(&partials)));
}

#[divan::bench]
fn sum_to_scalar(bencher: Bencher) {
    let partials = zone_partials(sum_accumulator);

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .bench_local(|| to_scalars(&partials));
}

#[divan::bench]
fn sum_merge_partials(bencher: Bencher) {
    let scalars = to_scalars(&zone_partials(sum_accumulator));

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .with_inputs(sum_accumulator)
        .bench_local_refs(|merged| merge_all(merged, &scalars));
}

#[divan::bench]
fn sum_roundtrip(bencher: Bencher) {
    let partials = zone_partials(sum_accumulator);

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .with_inputs(sum_accumulator)
        .bench_local_refs(|merged| merge_all(merged, &to_scalars(&partials)));
}
