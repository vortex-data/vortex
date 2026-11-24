// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding Affine arrays over a BitPacked residual child, in each mode, against
//! per-chunk FoR on the same data.
//!
//! Every benchmark carries `#[cpu_features]`, so it is measured on each walltime CPU-feature leg
//! rather than in simulation: the loops under test are auto-vectorized, so the build decides
//! their speed.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench affine_decode`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::mark_optimized;
use vortex_error::vortex_panic;
use vortex_fastlanes::Affine;
use vortex_fastlanes::AffineArrayExt;
use vortex_fastlanes::AffineArraySlotsExt;
use vortex_fastlanes::AffineOptions;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: usize = 64 * 1024;

/// A deterministic xorshift stream of 15-bit noise (stays below `u16::MAX` so narrowed `u16`
/// residuals remain bit-packed rather than falling back to an unpacked primitive array).
fn noise() -> impl FnMut() -> i64 {
    let mut x = 0x1234_5678u64;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % 32_768) as i64
    }
}

/// Microsecond timestamps on a one-millisecond grid with 15 bits of spread per chunk, which a
/// common divisor narrows.
fn grid() -> PrimitiveArray {
    let mut noise = noise();
    PrimitiveArray::from_iter(
        (0..LEN as i64)
            .map(|i| 1_672_531_200_000_000 + (i / 1024) * 1_000_000_000 + noise() * 1_000),
    )
}

/// Nanosecond timestamps on a one-minute grid with 15 bits of jitter, which a slope fits.
fn trend() -> PrimitiveArray {
    let mut noise = noise();
    PrimitiveArray::from_iter(
        (0..LEN as i64).map(|i| 1_700_000_000_000_000_000 + i * 60_000_000_000 + noise()),
    )
}

fn bit_width(array: &ArrayRef) -> u8 {
    let mut ctx = SESSION.create_execution_ctx();
    let p = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
    let max = match_each_integer_ptype!(p.ptype(), |T| {
        p.as_slice::<T>()
            .iter()
            .map(|&v| AsPrimitive::<u64>::as_(v))
            .max()
            .unwrap_or(0)
    });
    u8::try_from(u64::BITS - max.leading_zeros()).unwrap()
}

/// Bit-pack `residuals`, leaving them unpacked when they fill their type, as the compressor does.
fn pack(residuals: &ArrayRef) -> ArrayRef {
    let width = bit_width(residuals);
    if usize::from(width) >= residuals.dtype().as_ptype().bit_width() {
        return residuals.clone();
    }
    let mut ctx = SESSION.create_execution_ctx();
    BitPacked::encode(residuals, width, &mut ctx)
        .unwrap()
        .into_array()
}

fn affine(values: &PrimitiveArray, options: AffineOptions) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let affine = Affine::encode(values, options, &mut ctx).unwrap();
    Affine::try_new(
        pack(affine.encoded()),
        affine.references().clone(),
        affine.scales().clone(),
        affine.slopes().clone(),
        affine.offset(),
        affine.slope_shift(),
    )
    .unwrap()
    .into_array()
}

fn per_chunk_for(values: &PrimitiveArray) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let for_array = FoR::encode_chunked(values.clone(), &mut ctx).unwrap();
    let FoRReferences::Blocked(references) = for_array.references() else {
        vortex_panic!("every chunk has its own reference");
    };
    FoR::try_new_chunked(pack(for_array.encoded()), references.clone(), 0)
        .unwrap()
        .into_array()
}

fn run(bencher: Bencher, array: ArrayRef) {
    mark_optimized(&array, &SESSION).unwrap();
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn for_chunked_grid(bencher: Bencher) {
    run(bencher, per_chunk_for(&grid()));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn affine_for_grid(bencher: Bencher) {
    run(bencher, affine(&grid(), AffineOptions::FOR));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn affine_scale_grid(bencher: Bencher) {
    run(bencher, affine(&grid(), AffineOptions::SCALE));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn for_chunked_trend(bencher: Bencher) {
    run(bencher, per_chunk_for(&trend()));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn affine_slope_trend(bencher: Bencher) {
    run(bencher, affine(&trend(), AffineOptions::SLOPE));
}

#[vortex_bench_support::cpu_features]
#[divan::bench]
fn affine_all_trend(bencher: Bencher) {
    run(bencher, affine(&trend(), AffineOptions::ALL));
}
