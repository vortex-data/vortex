// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare-pushdown benchmark.
//!
//! Three pipelines for the same query "count values > threshold":
//!
//! 1. **raw_buffer**: a plain `&[f64]` scan — `iter().filter(|&&v| v > t).count()`. No Vortex,
//!    no compression. This is the floor any compressed-form pushdown has to beat.
//! 2. **btrblocks_decode_then_compare**: BtrBlocks-best compressed form (whatever scheme its
//!    sampling compressor picks) → decode to `PrimitiveArray` → count. The realistic baseline
//!    for "Vortex compressed file, ad-hoc compare".
//! 3. **neats_pushdown**: NeaTS-compressed form → `count_greater_than` using per-piece bounds.
//!    Pieces that can prove all-pass or all-fail from `[piece_min, piece_max]` are skipped
//!    without decoding.
//!
//! Encoding time is _not_ measured — only query latency on already-compressed data. The input
//! preparation happens inside `with_inputs`, which divan excludes from the timing.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt as _;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Buffer;
use vortex_neats::NeaTSArray;
use vortex_neats::NeaTSOptions;
use vortex_neats::compute::compare::count_greater_than;
use vortex_neats::neats_encode;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: std::sync::LazyLock<VortexSession> = std::sync::LazyLock::new(|| {
    let s = VortexSession::empty().with::<ArraySession>();
    vortex_neats::initialize(&s);
    s
});

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

fn linear_ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.001 * i as f64).collect()
}

fn sine_drift(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.01).sin() + 0.0005 * i as f64)
        .collect()
}

fn gps_trace(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut lat = 37.42_f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        lat += rng.random_range(-1e-5..1e-5);
        out.push(lat);
    }
    out
}

fn stock_walk(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut v = 100.0_f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        v *= 1.0 + rng.random_range(-0.005..0.005);
        out.push(v);
    }
    out
}

fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn primitive(values: &[f64]) -> PrimitiveArray {
    PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable)
}

fn make_btr(values: &[f64]) -> ArrayRef {
    let array = primitive(values).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    BtrBlocksCompressor::default()
        .compress(&array, &mut ctx)
        .unwrap()
}

fn make_neats(values: &[f64]) -> NeaTSArray {
    let array = primitive(values);
    neats_encode(array.as_view(), NeaTSOptions::default()).unwrap()
}

fn make_neats_lossy(values: &[f64], epsilon: f64) -> NeaTSArray {
    let array = primitive(values);
    neats_encode(
        array.as_view(),
        NeaTSOptions {
            epsilon: Some(epsilon),
            ..NeaTSOptions::default()
        },
    )
    .unwrap()
}

// ---- pipeline 1: raw &[f64] scan ----
fn bench_raw_buffer<F: Fn(usize) -> Vec<f64>>(bencher: Bencher, n: usize, generator: F) {
    let values = generator(n);
    let threshold = median(&values);
    bencher
        .with_inputs(|| values.clone())
        .bench_values(|buf| buf.iter().filter(|&&v| v > threshold).count());
}

// ---- pipeline 2: BtrBlocks-compressed -> decode -> count ----
fn bench_btrblocks_decode<F: Fn(usize) -> Vec<f64>>(bencher: Bencher, n: usize, generator: F) {
    let values = generator(n);
    let threshold = median(&values);
    let btr = make_btr(&values);
    bencher
        .with_inputs(|| (btr.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(compressed, mut ctx): (ArrayRef, ExecutionCtx)| {
            let decoded = compressed.execute::<PrimitiveArray>(&mut ctx).unwrap();
            decoded
                .as_slice::<f64>()
                .iter()
                .filter(|&&v| v > threshold)
                .count()
        });
}

// ---- pipeline 3: NeaTS-compressed -> piece-bound pushdown ----
fn bench_neats_pushdown<F: Fn(usize) -> Vec<f64>>(bencher: Bencher, n: usize, generator: F) {
    let values = generator(n);
    let threshold = median(&values);
    let neats = make_neats(&values);
    bencher
        .with_inputs(|| (neats.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(arr, mut ctx)| count_greater_than(&arr, threshold, &mut ctx).unwrap().0);
}

// ---- pipeline 3b: NeaTS lossy (eps=1e-3) -> pushdown ----
fn bench_neats_pushdown_lossy<F: Fn(usize) -> Vec<f64>>(bencher: Bencher, n: usize, generator: F) {
    let values = generator(n);
    let threshold = median(&values);
    let neats = make_neats_lossy(&values, 1e-3);
    bencher
        .with_inputs(|| (neats.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(arr, mut ctx)| count_greater_than(&arr, threshold, &mut ctx).unwrap().0);
}

macro_rules! compare_benches {
    ($mod_name:ident, $gen:ident) => {
        mod $mod_name {
            use super::*;

            #[divan::bench(args = SIZES)]
            fn raw_buffer(bencher: Bencher, n: usize) {
                bench_raw_buffer(bencher, n, $gen);
            }

            #[divan::bench(args = SIZES)]
            fn btrblocks_decode(bencher: Bencher, n: usize) {
                bench_btrblocks_decode(bencher, n, $gen);
            }

            #[divan::bench(args = SIZES)]
            fn neats_pushdown(bencher: Bencher, n: usize) {
                bench_neats_pushdown(bencher, n, $gen);
            }

            #[divan::bench(args = SIZES)]
            fn neats_pushdown_lossy_1e_minus_3(bencher: Bencher, n: usize) {
                bench_neats_pushdown_lossy(bencher, n, $gen);
            }
        }
    };
}

compare_benches!(linear_ramp_compare, linear_ramp);
compare_benches!(sine_drift_compare, sine_drift);
compare_benches!(gps_trace_compare, gps_trace);
compare_benches!(stock_walk_compare, stock_walk);
