// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `FloatQuantArray` (P2c of the layered pcodec stack).
//!
//! The bench measures the cost of the FloatQuant decomposition in isolation
//! against the full `vortex-pco` round trip on FloatQuant-favorable input.
//! For `N` elements we generate a seeded random `Buffer<f64>` whose low `k`
//! mantissa bits are cleared (`f64::from_bits(rand_u64 & !((1<<k) - 1))`).
//! That makes the `secondary` child all zeros — the favorable case for
//! downstream entropy-coded layers. Only `f64` is exercised in this phase
//! and `k = 16` leaves 48 bits of effective f64 precision (a representative
//! lossy-sensor-style scenario).

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
use vortex_float_quant::FloatQuant;
use vortex_pco::Pco;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;
const SCALAR_AT_SAMPLES: usize = 1_000;
const SEED: u64 = 42;
const K: u32 = 16;
const LOW_MASK: u64 = (1u64 << K) - 1;
/// Tuned to match `vortex-pco` defaults used elsewhere in the repo.
const PCO_LEVEL: usize = 0;
const PCO_VALUES_PER_PAGE: usize = 0;

fn build_favorable() -> Buffer<f64> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    (0..N)
        .map(|_| f64::from_bits(rng.random::<u64>() & !LOW_MASK))
        .collect::<Buffer<f64>>()
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
    (BytesCount::new(N * size_of::<f64>()), ItemsCount::new(N))
}

// ----- FloatQuant benches ---------------------------------------------------

#[divan::bench]
fn encode_float_quant(bencher: Bencher) {
    let parray = build_primitive();
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            FloatQuant::encode(parray.as_view(), K, &mut ctx).unwrap()
        });
}

#[divan::bench]
fn decode_float_quant(bencher: Bencher) {
    let parray = build_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = FloatQuant::encode(parray.as_view(), K, &mut ctx).unwrap();
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
fn scalar_at_float_quant(bencher: Bencher) {
    let parray = build_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = FloatQuant::encode(parray.as_view(), K, &mut ctx)
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
