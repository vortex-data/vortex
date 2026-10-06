// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression statistics over arrays with and without nulls.

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_compressor::stats::GenerateStatsOptions;
use vortex_compressor::stats::IntegerStats;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const LEN: usize = 64 * 1024;

fn main() {
    divan::main();
}

/// A seeded xorshift generator, so that every build benchmarks the same data.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
}

/// Runs of up to 8 values, drawn from 1024 distinct values.
fn values() -> Buffer<u32> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut values = Vec::with_capacity(LEN);
    while values.len() < LEN {
        let value = rng.next() % 1024;
        let run = (rng.next() % 8 + 1) as usize;
        values.extend(std::iter::repeat_n(value, run.min(LEN - values.len())));
    }
    Buffer::from(values)
}

/// Computes the statistics of a fresh array each iteration, so that the bounds cached on the array
/// by a previous iteration are not reused.
fn bench_stats(bencher: Bencher, validity: Validity) {
    let values = values();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::new(values.clone(), validity.clone()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| {
            IntegerStats::generate_opts(
                array,
                GenerateStatsOptions {
                    count_distinct_values: false,
                },
                ctx,
            )
        });
}

#[divan::bench]
fn integer_stats_non_null(bencher: Bencher) {
    bench_stats(bencher, Validity::NonNullable);
}

/// One value in ten is null, at random, so almost every chunk of 64 values has a null.
#[divan::bench]
fn integer_stats_nullable(bencher: Bencher) {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let valid = BitBuffer::from_iter((0..LEN).map(|_| !rng.next().is_multiple_of(10)));
    bench_stats(bencher, Validity::from(valid));
}
