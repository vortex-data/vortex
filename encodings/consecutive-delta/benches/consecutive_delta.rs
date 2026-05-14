// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `ConsecutiveDeltaArray` (P3 of the layered pcodec
//! stack).
//!
//! The bench measures the cost of the first-order consecutive-delta layer
//! in isolation against a full `vortex-pco` round trip on the same input.
//! All cases run on `N = 1_000_000` `i64` values across two scenarios:
//!
//! - **A (favorable)**: monotone timestamps with small jitter. Delta values
//!   are ~1000 ± 50 and pco's auto mode would also pick a delta path.
//! - **B (control)**: uniform random `i64`. Deltas have full
//!   high-bit entropy.

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
use vortex_buffer::BufferMut;
use vortex_consecutive_delta::ConsecutiveDelta;
use vortex_pco::Pco;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;
const SCALAR_AT_SAMPLES: usize = 1_000;
const SEED: u64 = 42;
const PCO_LEVEL: usize = 0;
const PCO_VALUES_PER_PAGE: usize = 0;

/// Scenario A: monotone-with-jitter timestamps. Mean step ~1000, jitter
/// ±50, base epoch in ms. Delta-favorable.
fn build_scenario_a() -> Buffer<i64> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let base: i64 = 1_700_000_000_000;
    let mut out = BufferMut::<i64>::with_capacity(N);
    for i in 0..N {
        let noise: i64 = rng.random_range(-50i64..=50);
        out.push(base + (i as i64) * 1000 + noise);
    }
    out.freeze()
}

/// Scenario B: uniform random `i64`. Delta-unfavorable control.
fn build_scenario_b() -> Buffer<i64> {
    let mut rng = SmallRng::seed_from_u64(SEED ^ 0x5A5A_5A5A_5A5A_5A5A);
    let mut out = BufferMut::<i64>::with_capacity(N);
    for _ in 0..N {
        out.push(rng.random::<i64>());
    }
    out.freeze()
}

fn to_primitive(buf: Buffer<i64>) -> PrimitiveArray {
    PrimitiveArray::new(buf, Validity::NonNullable)
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

/// Scenario tag. Each bench dispatches on this to pick the input builder.
trait Scenario {
    fn build() -> PrimitiveArray;
}

/// Scenario A: monotone timestamps with small jitter (delta-favorable).
struct A;
/// Scenario B: uniform random `i64` (delta-unfavorable control).
struct B;

impl Scenario for A {
    fn build() -> PrimitiveArray {
        to_primitive(build_scenario_a())
    }
}

impl Scenario for B {
    fn build() -> PrimitiveArray {
        to_primitive(build_scenario_b())
    }
}

// ----- ConsecutiveDelta benches ----------------------------------------------

#[divan::bench(types = [A, B])]
fn encode_consec_delta<S: Scenario>(bencher: Bencher) {
    let parray = S::build();
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            ConsecutiveDelta::encode(parray.as_view(), &mut ctx).unwrap()
        });
}

#[divan::bench(types = [A, B])]
fn decode_consec_delta<S: Scenario>(bencher: Bencher) {
    let parray = S::build();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx).unwrap();
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

#[divan::bench(types = [A, B])]
fn scalar_at_consec_delta<S: Scenario>(bencher: Bencher) {
    let parray = S::build();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx)
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

// ----- Reference baselines: full Pco -----------------------------------------

#[divan::bench(types = [A, B])]
fn pco_full_encode<S: Scenario>(bencher: Bencher) {
    let parray = S::build();
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap()
        });
}

#[divan::bench(types = [A, B])]
fn pco_full_decode<S: Scenario>(bencher: Bencher) {
    let parray = S::build();
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

#[divan::bench(types = [A, B])]
fn pco_scalar_at<S: Scenario>(bencher: Bencher) {
    let parray = S::build();
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
