// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

//! Per-encoding row-encode kernel throughput.
//!
//! Each encoding has two benchmarks, both fed the same encoded array:
//! - `*_kernel`: `RowEncoder::encode`, which routes the encoded array through the
//!   per-encoding kernel.
//! - `*_fallback`: canonicalize the encoded array, then encode the canonical form. This is
//!   exactly what the dispatcher does when no kernel claims the column, so it is the honest
//!   baseline the kernel replaces.
//!
//! The gap between the two is the kernel's end-to-end win (it avoids materializing the
//! canonical array and, for Dict/RunEnd/Constant, encodes each unique value once).

use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Alphanumeric;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_row::RowEncoder;
use vortex_runend::RunEnd;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const N: usize = 1_000_000;

fn main() {
    divan::main();
}

fn gen_words(n: usize, mean_len: usize, seed: u64) -> Vec<String> {
    let rng = &mut StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let len = rng.random_range(mean_len.saturating_sub(3)..=mean_len + 3);
            rng.sample_iter(&Alphanumeric)
                .take(len)
                .map(char::from)
                .collect::<String>()
        })
        .collect()
}

/// Encode the encoded array directly: the per-encoding kernel claims it.
fn run(bencher: divan::Bencher, col: ArrayRef) {
    let encoder = RowEncoder::default();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        encoder
            .encode(std::slice::from_ref(&col), &mut ctx)
            .unwrap()
    });
}

/// The dispatcher's fallback for an unclaimed column: canonicalize, then encode. Both steps
/// run inside the timed region so the decode cost the kernel avoids is included.
fn run_fallback(bencher: divan::Bencher, col: ArrayRef) {
    let encoder = RowEncoder::default();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let canon = col
            .clone()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();
        encoder.encode(&[canon], &mut ctx).unwrap()
    });
}

// ---------- Constant (Utf8) ----------

fn build_constant() -> ArrayRef {
    let scalar = Scalar::utf8(
        "a-moderately-sized-constant-string",
        Nullability::NonNullable,
    );
    ConstantArray::new(scalar, N).into_array()
}

#[divan::bench]
fn constant_utf8_kernel(bencher: divan::Bencher) {
    run(bencher, build_constant());
}

#[divan::bench]
fn constant_utf8_fallback(bencher: divan::Bencher) {
    run_fallback(bencher, build_constant());
}

// ---------- Dict (Utf8, 256 unique words) ----------

fn build_dict() -> ArrayRef {
    let words = gen_words(256, 8, 1);
    let values = VarBinViewArray::from_iter_str(words.iter().map(String::as_str)).into_array();
    let mut rng = StdRng::seed_from_u64(2);
    let codes: Vec<u32> = (0..N).map(|_| rng.random_range(0..256u32)).collect();
    let codes = PrimitiveArray::from_iter(codes).into_array();
    DictArray::new(codes, values).into_array()
}

#[divan::bench]
fn dict_utf8_kernel(bencher: divan::Bencher) {
    run(bencher, build_dict());
}

#[divan::bench]
fn dict_utf8_fallback(bencher: divan::Bencher) {
    run_fallback(bencher, build_dict());
}

// ---------- RunEnd (i64, mean run length ~16) ----------

fn build_runend() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(3);
    let mut vals: Vec<i64> = Vec::with_capacity(N);
    while vals.len() < N {
        let v: i64 = rng.random();
        let run = rng.random_range(8..=24);
        for _ in 0..run {
            if vals.len() < N {
                vals.push(v);
            }
        }
    }
    let raw = PrimitiveArray::from_iter(vals).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    RunEnd::encode(raw, &mut ctx).unwrap().into_array()
}

#[divan::bench]
fn runend_i64_kernel(bencher: divan::Bencher) {
    run(bencher, build_runend());
}

#[divan::bench]
fn runend_i64_fallback(bencher: divan::Bencher) {
    run_fallback(bencher, build_runend());
}

// ---------- BitPacked (u32, 10 bits) ----------

fn build_bitpacked() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(4);
    let vals: Vec<u32> = (0..N).map(|_| rng.random_range(0..1024u32)).collect();
    let arr = PrimitiveArray::from_iter(vals).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    BitPackedData::encode(&arr, 10, &mut ctx)
        .unwrap()
        .into_array()
}

#[divan::bench]
fn bitpacked_u32_kernel(bencher: divan::Bencher) {
    run(bencher, build_bitpacked());
}

#[divan::bench]
fn bitpacked_u32_fallback(bencher: divan::Bencher) {
    run_fallback(bencher, build_bitpacked());
}

// ---------- FoR (u32 over BitPacked, fused path) ----------

fn build_for() -> ArrayRef {
    let base = 1_000_000u32;
    let mut rng = StdRng::seed_from_u64(5);
    let enc_vals: Vec<u32> = (0..N).map(|_| rng.random_range(0..1024u32)).collect();
    let enc = PrimitiveArray::from_iter(enc_vals).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let bp = BitPackedData::encode(&enc, 10, &mut ctx)
        .unwrap()
        .into_array();
    FoR::try_new(bp, Scalar::from(base)).unwrap().into_array()
}

#[divan::bench]
fn for_u32_kernel(bencher: divan::Bencher) {
    run(bencher, build_for());
}

#[divan::bench]
fn for_u32_fallback(bencher: divan::Bencher) {
    run_fallback(bencher, build_for());
}

// ---------- Delta (i64) ----------

fn build_delta() -> ArrayRef {
    let vals: Vec<i64> = (0..N as i64).map(|i| 1000 + i * 3).collect();
    let p = PrimitiveArray::from_iter(vals);
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    Delta::try_from_primitive_array(&p, &mut ctx)
        .unwrap()
        .into_array()
}

#[divan::bench]
fn delta_i64_kernel(bencher: divan::Bencher) {
    run(bencher, build_delta());
}

#[divan::bench]
fn delta_i64_fallback(bencher: divan::Bencher) {
    run_fallback(bencher, build_delta());
}
