// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `IntMultArray` (P2a of the layered pcodec stack).
//!
//! The bench measures the cost of the IntMult decomposition in isolation
//! against the full `vortex-pco` round trip on IntMult-favorable input.
//! For every supported latent type `L ∈ {u32, u64}` we generate a seeded
//! random `Buffer<L>` of `N` elements with `latent[i] = base * k_i + r_i`,
//! `base = 1000`, `k_i ∈ [0, 1_000_000)`, `r_i ∈ [0, base)`. The narrow
//! widths `u8`/`u16` are skipped — the signal is identical and the bench
//! time per cell blows up.

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
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_int_mult::IntMult;
use vortex_pco::Pco;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;
const SCALAR_AT_SAMPLES: usize = 1_000;
const SEED: u64 = 42;
const BASE: u64 = 1000;
/// Tuned to match `vortex-pco` defaults used elsewhere in the repo.
const PCO_LEVEL: usize = 0;
const PCO_VALUES_PER_PAGE: usize = 0;

/// Marker trait that gates the latent widths covered by the bench.
trait BenchLatent: NativePType {
    fn favor(k: u64, r: u64) -> Self;
}

impl BenchLatent for u32 {
    #[inline]
    fn favor(k: u64, r: u64) -> Self {
        u32::try_from(BASE * k + r).unwrap()
    }
}

impl BenchLatent for u64 {
    #[inline]
    fn favor(k: u64, r: u64) -> Self {
        BASE * k + r
    }
}

fn build_favorable<L: BenchLatent>() -> Buffer<L> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    (0..N)
        .map(|_| {
            let k = rng.random_range(0u64..1_000_000);
            let r = rng.random_range(0u64..BASE);
            L::favor(k, r)
        })
        .collect::<Buffer<L>>()
}

fn build_primitive<L: BenchLatent>() -> PrimitiveArray {
    PrimitiveArray::new(build_favorable::<L>(), Validity::NonNullable)
}

fn sample_indices() -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    (0..SCALAR_AT_SAMPLES)
        .map(|_| rng.random_range(0..N))
        .collect()
}

fn throughput_counters<T: NativePType>() -> (BytesCount, ItemsCount) {
    (BytesCount::new(N * size_of::<T>()), ItemsCount::new(N))
}

// ----- IntMult benches --------------------------------------------------------

#[divan::bench(types = [u32, u64])]
fn encode_int_mult<L: BenchLatent>(bencher: Bencher) {
    let parray = build_primitive::<L>();
    let (bytes, items) = throughput_counters::<L>();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            IntMult::encode(parray.as_view(), BASE, &mut ctx).unwrap()
        });
}

#[divan::bench(types = [u32, u64])]
fn decode_int_mult<L: BenchLatent>(bencher: Bencher) {
    let parray = build_primitive::<L>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = IntMult::encode(parray.as_view(), BASE, &mut ctx).unwrap();
    let (bytes, items) = throughput_counters::<L>();
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

#[divan::bench(types = [u32, u64])]
fn scalar_at_int_mult<L: BenchLatent>(bencher: Bencher) {
    let parray = build_primitive::<L>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = IntMult::encode(parray.as_view(), BASE, &mut ctx)
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

#[divan::bench(types = [u32, u64])]
fn pco_full_encode<L: BenchLatent>(bencher: Bencher) {
    let parray = build_primitive::<L>();
    let (bytes, items) = throughput_counters::<L>();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap()
        });
}

#[divan::bench(types = [u32, u64])]
fn pco_full_decode<L: BenchLatent>(bencher: Bencher) {
    let parray = build_primitive::<L>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded =
        Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap();
    let (bytes, items) = throughput_counters::<L>();
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

#[divan::bench(types = [u32, u64])]
fn pco_scalar_at<L: BenchLatent>(bencher: Bencher) {
    let parray = build_primitive::<L>();
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
