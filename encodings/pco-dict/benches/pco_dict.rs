// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `PcoDictArray` (P2d of the layered pcodec stack).
//!
//! The bench measures the cost of the PcoDict decomposition in isolation
//! against the full `vortex-pco` round trip on dict-favorable input. For
//! `N` elements we generate `DICT_SIZE` random `i64` values up front (with
//! seed `42`), then emit `N` cells by sampling that pool with a separately
//! seeded RNG. The result is a stream of exactly 256 unique values whose
//! occurrences are scattered uniformly across the buffer — the case where
//! pco's auto-mode would also pick Dict.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::BytesCount;
use divan::counter::ItemsCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_pco::Pco;
use vortex_pco_dict::PcoDict;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;
const SCALAR_AT_SAMPLES: usize = 1_000;
const SEED: u64 = 42;
const DICT_SIZE: usize = 256;
/// Tuned to match `vortex-pco` defaults used elsewhere in the repo.
const PCO_LEVEL: usize = 0;
const PCO_VALUES_PER_PAGE: usize = 0;

fn build_favorable() -> Buffer<i64> {
    let mut dict_rng = SmallRng::seed_from_u64(SEED);
    let dict: Vec<i64> = (0..DICT_SIZE).map(|_| dict_rng.random::<i64>()).collect();

    let mut idx_rng = SmallRng::seed_from_u64(SEED ^ 0xDEAD_BEEF_CAFE_F00D);
    (0..N)
        .map(|_| dict[idx_rng.random_range(0..DICT_SIZE)])
        .collect::<Buffer<i64>>()
}

fn build_primitive() -> PrimitiveArray {
    PrimitiveArray::new(build_favorable(), Validity::NonNullable)
}

fn sample_indices() -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    (0..SCALAR_AT_SAMPLES)
        .map(|_| rng.random_range(0..N))
        .collect()
}

fn throughput_counters() -> (BytesCount, ItemsCount) {
    (BytesCount::new(N * size_of::<i64>()), ItemsCount::new(N))
}

// ----- PcoDict benches -------------------------------------------------------

#[divan::bench]
fn encode_pco_dict(bencher: Bencher) {
    let parray = build_primitive();
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| PcoDict::encode(parray.as_view(), &mut ctx).unwrap());
}

#[divan::bench]
fn decode_pco_dict(bencher: Bencher) {
    let parray = build_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = PcoDict::encode(parray.as_view(), &mut ctx).unwrap();
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (encoded.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(encoded, mut ctx)| {
            encoded
                .into_array()
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn scalar_at_pco_dict(bencher: Bencher) {
    let parray = build_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = PcoDict::encode(parray.as_view(), &mut ctx)
        .unwrap()
        .into_array();
    let indices = sample_indices();
    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| (encoded.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            for &i in &indices {
                divan::black_box(array.execute_scalar(i, &mut ctx).unwrap());
            }
        });
}

// ----- Reference baselines: full Pco ----------------------------------------

#[divan::bench]
fn pco_full_encode(bencher: Bencher) {
    let parray = build_primitive();
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap()
        });
}

#[divan::bench]
fn pco_full_decode(bencher: Bencher) {
    let parray = build_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded =
        Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap();
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (encoded.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(encoded, mut ctx)| {
            encoded
                .into_array()
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn pco_scalar_at(bencher: Bencher) {
    let parray = build_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx)
        .unwrap()
        .into_array();
    let indices = sample_indices();
    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| (encoded.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            for &i in &indices {
                divan::black_box(array.execute_scalar(i, &mut ctx).unwrap());
            }
        });
}
