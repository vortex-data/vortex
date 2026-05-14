// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `BinPartitionArray` + `VarWidthBitPackedArray` (P4 of
//! the layered pcodec stack).
//!
//! Three input scenarios over `N = 1_000_000` `i64` values are exercised:
//!
//! - **A — skewed-low** — `x[i] = (rng.f64().powi(3) * 1000.0) as i64`.
//!   Highly favorable for bin partition: most values are small with the
//!   occasional large outlier.
//! - **B — uniform random** — `i64` uniform over a wide range. Control:
//!   bin partition gets little to no compression win because every bin
//!   covers a full-width range.
//! - **C — quasi-monotone** — `x[i] = i + noise(±100)`. Mixes a wide range
//!   with low local entropy.
//!
//! Each scenario reports encode/decode/scalar_at throughput for both
//! `BinPartition` and the monolithic `PcoArray` baseline, plus a
//! compression ratio computed before `divan::main()` and printed to
//! stdout.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_precision_loss)]

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
use vortex_bin_partition::BinPartition;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_pco::Pco;

const N: usize = 1_000_000;
const SCALAR_AT_SAMPLES: usize = 1_000;
const SEED: u64 = 42;
const MAX_BINS: usize = 16;
const PCO_LEVEL: usize = 0;
const PCO_VALUES_PER_PAGE: usize = 0;

fn main() {
    // Compression-ratio comparison runs once, ahead of the divan benches,
    // so the numbers always reach stdout regardless of bench filtering.
    print_compression_ratio::<A>("A skewed");
    print_compression_ratio::<B>("B uniform");
    print_compression_ratio::<C>("C monotone");
    divan::main();
}

// ----- Scenarios -------------------------------------------------------------

/// Scenario tag. Each bench dispatches on this to pick the input builder.
trait Scenario {
    fn build() -> Buffer<i64>;
}

/// Scenario A: skewed-low. Most values are small; rare large values.
struct A;
/// Scenario B: uniform random `i64` in a wide range. Control.
struct B;
/// Scenario C: quasi-monotone. `i + noise(±100)`.
struct C;

impl Scenario for A {
    fn build() -> Buffer<i64> {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let mut out = BufferMut::<i64>::with_capacity(N);
        for _ in 0..N {
            let u: f64 = rng.random::<f64>();
            out.push((u.powi(3) * 1000.0) as i64);
        }
        out.freeze()
    }
}

impl Scenario for B {
    fn build() -> Buffer<i64> {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x5A5A_5A5A_5A5A_5A5A);
        let mut out = BufferMut::<i64>::with_capacity(N);
        for _ in 0..N {
            out.push(rng.random_range(-1_000_000_000i64..1_000_000_000));
        }
        out.freeze()
    }
}

impl Scenario for C {
    fn build() -> Buffer<i64> {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xCAFE_C0DE_F00D_FEED);
        let mut out = BufferMut::<i64>::with_capacity(N);
        for i in 0..N {
            let noise: i64 = rng.random_range(-100i64..=100);
            out.push(i as i64 + noise);
        }
        out.freeze()
    }
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

// ----- Compression-ratio headline -------------------------------------------

fn print_compression_ratio<S: Scenario>(label: &str) {
    let buf = S::build();
    let parray = to_primitive(buf);
    let raw_bytes = N as u64 * size_of::<i64>() as u64;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let bp = BinPartition::encode(parray.as_view(), MAX_BINS, &mut ctx).unwrap();
    let bp_bytes = bp.into_array().nbytes();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let pco_arr =
        Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap();
    let pco_bytes = pco_arr.into_array().nbytes();

    println!(
        "compression-ratio {label}: raw={:.2} MB | bin_partition={:.2} MB ({:.2} x) | \
         pco={:.2} MB ({:.2} x)",
        as_mb(raw_bytes),
        as_mb(bp_bytes),
        ratio(raw_bytes, bp_bytes),
        as_mb(pco_bytes),
        ratio(raw_bytes, pco_bytes),
    );
}

fn as_mb(n: u64) -> f64 {
    n as f64 / (1024.0 * 1024.0)
}

fn ratio(raw: u64, encoded: u64) -> f64 {
    if encoded == 0 {
        0.0
    } else {
        raw as f64 / encoded as f64
    }
}

// ----- BinPartition benches -------------------------------------------------

#[divan::bench(types = [A, B, C])]
fn encode_bin_partition<S: Scenario>(bencher: Bencher) {
    let parray = to_primitive(S::build());
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            BinPartition::encode(parray.as_view(), MAX_BINS, &mut ctx).unwrap()
        });
}

#[divan::bench(types = [A, B, C])]
fn decode_bin_partition<S: Scenario>(bencher: Bencher) {
    let parray = to_primitive(S::build());
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = BinPartition::encode(parray.as_view(), MAX_BINS, &mut ctx).unwrap();
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

#[divan::bench(types = [A, B, C])]
fn scalar_at_bin_partition<S: Scenario>(bencher: Bencher) {
    let parray = to_primitive(S::build());
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = BinPartition::encode(parray.as_view(), MAX_BINS, &mut ctx)
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

#[divan::bench(types = [A, B, C])]
fn pco_full_encode<S: Scenario>(bencher: Bencher) {
    let parray = to_primitive(S::build());
    let (bytes, items) = throughput_counters();
    bencher
        .counter(bytes)
        .counter(items)
        .with_inputs(|| (parray.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(parray, mut ctx)| {
            Pco::from_primitive(parray.as_view(), PCO_LEVEL, PCO_VALUES_PER_PAGE, &mut ctx).unwrap()
        });
}

#[divan::bench(types = [A, B, C])]
fn pco_full_decode<S: Scenario>(bencher: Bencher) {
    let parray = to_primitive(S::build());
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

#[divan::bench(types = [A, B, C])]
fn pco_scalar_at<S: Scenario>(bencher: Bencher) {
    let parray = to_primitive(S::build());
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
