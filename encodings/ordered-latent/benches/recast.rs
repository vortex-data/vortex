// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `OrderedLatentArray` (P1 of the layered pcodec stack).
//!
//! The bench measures the cost of the order-preserving recast in isolation
//! against memcpy of an identical buffer and against the much heavier full
//! `vortex-pco` round trip. For every primitive type `T` we generate a
//! seeded uniform-random `Vec<T>` of `N` elements and reuse it across all
//! related benches.

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
use vortex_ordered_latent::OrderedLatent;
use vortex_pco::Pco;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;
const SCALAR_AT_SAMPLES: usize = 1_000;
const SEED: u64 = 42;
/// Tuned to match `vortex-pco` defaults used elsewhere in the repo.
const PCO_LEVEL: usize = 0;
const PCO_VALUES_PER_PAGE: usize = 0;

/// Marker trait that gates the primitive types covered by the bench.
trait BenchType: NativePType {
    fn random(rng: &mut SmallRng) -> Self;
}

macro_rules! impl_bench_int {
    ($T:ty) => {
        impl BenchType for $T {
            #[inline]
            fn random(rng: &mut SmallRng) -> Self {
                rng.random::<Self>()
            }
        }
    };
}

impl_bench_int!(i8);
impl_bench_int!(i16);
impl_bench_int!(i32);
impl_bench_int!(i64);
impl_bench_int!(u32);
impl_bench_int!(u64);
impl_bench_int!(f32);
impl_bench_int!(f64);

fn build_input<T: BenchType>() -> Buffer<T> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    (0..N).map(|_| T::random(&mut rng)).collect::<Buffer<T>>()
}

fn build_primitive<T: BenchType>() -> PrimitiveArray {
    PrimitiveArray::new(build_input::<T>(), Validity::NonNullable)
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

// ----- OrderedLatent benches --------------------------------------------------

#[divan::bench(types = [i8, i16, i32, i64, u32, u64, f32, f64])]
fn encode_ordered_latent<T: BenchType>(bencher: Bencher) {
    let parray = build_primitive::<T>();
    let (bytes, items) = throughput_counters::<T>();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            OrderedLatent::encode(parray.as_view(), &mut ctx).unwrap()
        });
}

#[divan::bench(types = [i8, i16, i32, i64, u32, u64, f32, f64])]
fn decode_ordered_latent<T: BenchType>(bencher: Bencher) {
    let parray = build_primitive::<T>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = OrderedLatent::encode(parray.as_view(), &mut ctx).unwrap();
    let (bytes, items) = throughput_counters::<T>();
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

#[divan::bench(types = [i8, i16, i32, i64, u32, u64, f32, f64])]
fn scalar_at_ordered_latent<T: BenchType>(bencher: Bencher) {
    let parray = build_primitive::<T>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = OrderedLatent::encode(parray.as_view(), &mut ctx)
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

// ----- Reference baselines ----------------------------------------------------

#[divan::bench(types = [i8, i16, i32, i64, u32, u64, f32, f64])]
fn memcpy<T: BenchType>(bencher: Bencher) {
    let input: Buffer<T> = build_input::<T>();
    let (bytes, items) = throughput_counters::<T>();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| &input)
        .bench_refs(|input| divan::black_box((*input).clone()));
}

// Pco does not support 8-bit primitive widths; skip i8 for the reference rows.
#[divan::bench(types = [i16, i32, i64, u32, u64, f32, f64])]
fn pco_full_encode<T: BenchType>(bencher: Bencher) {
    let parray = build_primitive::<T>();
    let (bytes, items) = throughput_counters::<T>();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap()
        });
}

#[divan::bench(types = [i16, i32, i64, u32, u64, f32, f64])]
fn pco_full_decode<T: BenchType>(bencher: Bencher) {
    let parray = build_primitive::<T>();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded =
        Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap();
    let (bytes, items) = throughput_counters::<T>();
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

#[divan::bench(types = [i16, i32, i64, u32, u64, f32, f64])]
fn pco_scalar_at<T: BenchType>(bencher: Bencher) {
    let parray = build_primitive::<T>();
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
