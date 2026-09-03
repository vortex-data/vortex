// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for [`BloomPartial`] insertion and membership checks.
//!
//! The idea is to benchmark `insert` and `contain` across different filter sizes.
//! Larger filters are less cache-friendly.

#![expect(clippy::expect_used)]

use std::num::NonZeroU32;

use divan::Bencher;
use divan::counter::ItemsCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::BloomOptions;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::BloomPartial;
use vortex_layout::layouts::zoned::aggregates::bloom_filter::HashFn;

fn main() {
    divan::main();
}

/// [Tiny, Default, Cache-unfriendly].
const BLOCK_COUNTS: &[u32] = &[
    256,       // 8KiB
    32_768,    // 1MiB
    1_048_576, // 32 MiB
];
const VALUE_COUNT: usize = 8192;

fn options(block_count: u32) -> BloomOptions {
    BloomOptions::new(
        NonZeroU32::new(block_count).expect("benchmark block counts are non-zero"),
        HashFn::XxHash3_64,
    )
}

fn values(seed: u64) -> Vec<[u8; 16]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..VALUE_COUNT).map(|_| rng.random()).collect()
}

fn populated_filter(options: &BloomOptions, values: &[[u8; 16]]) -> BloomPartial {
    let mut filter = BloomPartial::from(options);

    for value in values {
        filter.insert(value);
    }

    filter
}

#[divan::bench(args = BLOCK_COUNTS)]
fn insert(bencher: Bencher, block_count: u32) {
    let options = options(block_count);
    let values = values(0xbeef);

    bencher
        .counter(ItemsCount::new(VALUE_COUNT))
        .with_inputs(|| (BloomPartial::from(&options), values.as_slice()))
        .bench_refs(|(filter, values)| {
            for value in *values {
                filter.insert(value);
            }
        });
}

#[divan::bench(args = BLOCK_COUNTS)]
fn contains_present(bencher: Bencher, block_count: u32) {
    let options = options(block_count);
    let values = values(0xbeef);
    let filter = populated_filter(&options, &values);

    bencher
        .counter(ItemsCount::new(VALUE_COUNT))
        .with_inputs(|| (&filter, values.as_slice()))
        .bench_refs(|(filter, values)| {
            let mut matches = 0;
            for value in *values {
                matches += usize::from(filter.contains(value));
            }
            divan::black_box(matches);
        });
}

#[divan::bench(args = BLOCK_COUNTS)]
fn contains_absent(bencher: Bencher, block_count: u32) {
    let options = options(block_count);
    let inserted_values = values(0xbeef);
    let queries = values(0xfeed);
    let filter = populated_filter(&options, &inserted_values);

    bencher
        .counter(ItemsCount::new(VALUE_COUNT))
        .with_inputs(|| (&filter, queries.as_slice()))
        .bench_refs(|(filter, queries)| {
            let mut matches = 0;
            for query in *queries {
                matches += usize::from(filter.contains(query));
            }
            divan::black_box(matches);
        });
}
