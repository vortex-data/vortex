// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks blocked FoR compute kernels (`compare`, `is_constant`, `is_sorted`) and narrowed
//! vs same-width blocked FoR `decode`.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench for_compute`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
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
const BIT_WIDTH: u8 = 7;

fn climbing_blocked_for(narrowed: bool) -> (ArrayRef, ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    // Each 1,024-row chunk climbs by 1,000 with an in-chunk spread < 128 so residuals pack to 7
    // bits (and narrow to `u8` when `narrowed` is true).
    let raw: Buffer<i64> = (0..LEN)
        .map(|i| 1_000 * (1 + (i / 1024) as i64) + ((i % 1024) as i64 * 100) / 1024)
        .collect();
    let prim = PrimitiveArray::new(raw, Validity::NonNullable);
    let for_arr = FoR::encode_chunked(prim, &mut ctx).unwrap();
    let FoRReferences::Blocked(refs) = for_arr.references() else {
        unreachable!("each chunk has a distinct reference");
    };
    let encoded = if narrowed {
        for_arr.encoded().clone()
    } else {
        // Re-widen residuals to `i64` to measure same-width vs narrowed blocked FoR decode.
        let wide: Buffer<i64> = for_arr
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
            .as_slice::<u8>()
            .iter()
            .map(|&v| i64::from(v))
            .collect();
        PrimitiveArray::new(wide, Validity::NonNullable).into_array()
    };
    let packed = BitPacked::encode(&encoded, BIT_WIDTH, &mut ctx)
        .unwrap()
        .into_array();
    (packed, refs.clone())
}

fn fresh_blocked_for(encoded: &ArrayRef, refs: &ArrayRef) -> ArrayRef {
    let array = FoR::try_new_chunked(encoded.clone(), refs.clone(), 0)
        .unwrap()
        .into_array();
    mark_optimized(&array, &SESSION).unwrap();
    array
}

#[divan::bench]
fn decode_blocked_narrowed_i64(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    let array = fresh_blocked_for(&encoded, &refs);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[divan::bench]
fn decode_blocked_same_width_i64(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(false);
    let array = fresh_blocked_for(&encoded, &refs);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[divan::bench]
fn compare_eq_blocked_pruned(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    let array = fresh_blocked_for(&encoded, &refs);
    // Target falls inside chunk 32 only; the other 63 chunks prune without unpacking.
    let rhs = ConstantArray::new(33_050i64, LEN).into_array();
    mark_optimized(&rhs, &SESSION).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        array
            .clone()
            .compare(rhs.clone(), CompareOperator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench]
fn compare_eq_decode_fallback(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    let array = fresh_blocked_for(&encoded, &refs);
    let rhs = ConstantArray::new(33_050i64, LEN).into_array();
    mark_optimized(&rhs, &SESSION).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        let prim = array
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
            .into_array();
        prim.compare(rhs.clone(), CompareOperator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench]
fn compare_lt_blocked_pruned(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    let array = fresh_blocked_for(&encoded, &refs);
    let rhs = ConstantArray::new(33_050i64, LEN).into_array();
    mark_optimized(&rhs, &SESSION).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        array
            .clone()
            .compare(rhs.clone(), CompareOperator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench]
fn compare_lt_decode_fallback(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    let array = fresh_blocked_for(&encoded, &refs);
    let rhs = ConstantArray::new(33_050i64, LEN).into_array();
    mark_optimized(&rhs, &SESSION).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
        let prim = array
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
            .into_array();
        prim.compare(rhs.clone(), CompareOperator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench]
fn is_constant_blocked_early_exit(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            (
                fresh_blocked_for(&encoded, &refs),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| is_constant(&array, &mut ctx).unwrap());
}

#[divan::bench]
fn is_constant_decode_fallback(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            (
                fresh_blocked_for(&encoded, &refs),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            let prim = array
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
                .into_array();
            is_constant(&prim, &mut ctx).unwrap()
        });
}

#[divan::bench]
fn is_sorted_blocked_true(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            (
                fresh_blocked_for(&encoded, &refs),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| is_sorted(&array, &mut ctx).unwrap());
}

#[divan::bench]
fn is_sorted_decode_fallback(bencher: Bencher) {
    let (encoded, refs) = climbing_blocked_for(true);
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            (
                fresh_blocked_for(&encoded, &refs),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            let prim = array
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
                .into_array();
            is_sorted(&prim, &mut ctx).unwrap()
        });
}
