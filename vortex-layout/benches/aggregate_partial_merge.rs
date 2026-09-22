// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for reading stored aggregate partials back and merging them.
//!
//! A zoned layout stores one partial scalar per zone. Folding those back into a single
//! accumulator is `combine_partials`, which both reads the partial out of its scalar
//! representation and merges it into the accumulator's state. That is what is timed here,
//! over two partial shapes: a bloom filter, whose partial is a byte blob that grows with
//! the filter size, and `SumV2`, whose partial is a small
//! `{sum, is_overflow, is_empty}` struct.
//!
//! Building the partials and converting them to scalars is setup, not part of the
//! measurement.

#![expect(clippy::expect_used)]

use std::num::NonZeroU32;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
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

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

/// Zones merged per iteration.
const ZONE_COUNT: usize = 16;
/// Rows per zone. Only affects how densely the filters are populated: merging is a bitwise OR
/// over the whole partial, so its cost does not depend on the values, and accumulating them is
/// setup rather than part of the measurement.
const ZONE_LEN: usize = 4096;
/// [Default, larger] block counts, i.e. 8KiB and 32KiB of bloom partial per zone.
/// Merge cost scales with `ZONE_COUNT * blocks`, and the product is what has to stay under a
/// millisecond once CodSpeed's instrumentation overhead is applied.
const BLOCK_COUNTS: &[u32] = &[256, 1024];

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

fn sum_v2_accumulator() -> AccumulatorRef {
    Box::new(
        Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), input_dtype())
            .expect("sum_v2 accepts i64 input"),
    )
}

/// The stored partial scalar for each zone, as a zoned layout holds them.
fn zone_partial_scalars(new_accumulator: impl Fn() -> AccumulatorRef) -> Vec<Scalar> {
    let mut ctx = SESSION.create_execution_ctx();
    (0..ZONE_COUNT)
        .map(|zone| {
            let mut accumulator = new_accumulator();
            accumulator
                .accumulate(&zone_array(zone), &mut ctx)
                .expect("accumulate zone");
            accumulator.partial_scalar().expect("partial scalar")
        })
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
fn bloom(bencher: Bencher, block_count: u32) {
    let scalars = zone_partial_scalars(|| bloom_accumulator(block_count));

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .with_inputs(|| bloom_accumulator(block_count))
        .bench_local_refs(|merged| merge_all(merged, &scalars));
}

#[divan::bench]
fn sum_v2(bencher: Bencher) {
    let scalars = zone_partial_scalars(sum_v2_accumulator);

    bencher
        .counter(ItemsCount::new(ZONE_COUNT))
        .with_inputs(sum_v2_accumulator)
        .bench_local_refs(|merged| merge_all(merged, &scalars));
}
