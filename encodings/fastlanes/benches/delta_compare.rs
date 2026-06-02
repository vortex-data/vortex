// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare/between an already delta-encoded [`DeltaArray`] against a constant.
//!
//! For each operator (`lt`, `eq`, `between`) there are two benches over the same inputs:
//! * `fast_*` — the block-streaming kernel: decompress one 1024-element FastLanes chunk at a time
//!   into stack scratch and fold the predicate straight into a bit buffer. The full primitive is
//!   never materialised.
//! * `baseline_*` — what the generic fallback does: materialise the whole unpacked primitive on
//!   the heap, then run the Arrow compare/between over it (a second pass over `len * size_of::<T>()`
//!   bytes of freshly written memory).
//!
//! Both paths run the *identical* delta decompress (undelta + untranspose, a per-lane prefix sum
//! that is compute-bound), so the only thing the fast path saves is the materialise-and-reread of
//! the output primitive. The size sweep is chosen to cross every cache tier, because the size of
//! that avoided round-trip — and hence the speedup — is U-shaped in `len`:
//!
//! | len    | primitive | regime                  | typical speedup |
//! |--------|-----------|-------------------------|-----------------|
//! | 1 Ki   | 4 KiB     | per-call overhead        | high (~2.5x)   |
//! | 64 Ki  | 256 KiB   | L2-resident (round-trip cheap) | trough (~1.05x) |
//! | 1 Mi   | 4 MiB     | L2/L3 spill              | ~1.1x          |
//! | 16 Mi  | 64 MiB    | full DRAM round-trip     | high (~2.5x)   |
//!
//! i.e. the streaming kernel wins most exactly where it matters — arrays too big to cache — and
//! the tiny-array win is a fixed allocation/dispatch saving, not a cache effect.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench delta_compare`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::Delta;

fn main() {
    divan::main();
}

/// Spans per-call-overhead (1 Ki), L2-resident (64 Ki), L2/L3 spill (1 Mi), and full DRAM
/// round-trip (16 Mi) regimes so the U-shaped speedup curve is visible end to end.
const LENS: &[usize] = &[
    1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
];

/// Keep the largest (16 Mi → ~30 ms/iter) benches tractable while still low-variance.
const SAMPLES: u32 = 50;

const WINDOW: u32 = 4096;
const NON_STRICT: BetweenOptions = BetweenOptions {
    lower_strict: StrictComparison::NonStrict,
    upper_strict: StrictComparison::NonStrict,
};

/// A delta-encoded ramp that wraps modulo [`WINDOW`], so a midpoint constant splits the array
/// ~50/50 (compare/between are branch-free, so selectivity does not affect timing — this just
/// keeps the predicate honest).
fn build_array(len: usize) -> (ArrayRef, ExecutionCtx) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let buf: BufferMut<u32> = (0..len).map(|i| (i as u32) % WINDOW).collect();
    let array = Delta::try_from_primitive_array(
        &PrimitiveArray::new(buf.freeze(), Validity::NonNullable),
        &mut ctx,
    )
    .unwrap()
    .into_array();
    (array, ctx)
}

/// Streaming kernel path: dispatches through `binary`, which routes to the delta compare kernel.
fn compare_fast(bencher: Bencher, len: usize, op: Operator) {
    let (array, mut ctx) = build_array(len);
    let rhs = ConstantArray::new(WINDOW / 2, len).into_array();
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), op)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

/// Fallback path: materialise the full primitive, then Arrow-compare it.
fn compare_baseline(bencher: Bencher, len: usize, op: Operator) {
    let (array, mut ctx) = build_array(len);
    let rhs = ConstantArray::new(WINDOW / 2, len).into_array();
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), op)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

fn between_fast(bencher: Bencher, len: usize) {
    let (array, mut ctx) = build_array(len);
    let lower = ConstantArray::new(WINDOW / 4, len).into_array();
    let upper = ConstantArray::new(3 * WINDOW / 4, len).into_array();
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .between(lower.clone(), upper.clone(), NON_STRICT)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

fn between_baseline(bencher: Bencher, len: usize) {
    let (array, mut ctx) = build_array(len);
    let lower = ConstantArray::new(WINDOW / 4, len).into_array();
    let upper = ConstantArray::new(3 * WINDOW / 4, len).into_array();
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .between(lower.clone(), upper.clone(), NON_STRICT)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS, sample_count = SAMPLES)]
fn fast_lt(bencher: Bencher, len: usize) {
    compare_fast(bencher, len, Operator::Lt);
}

#[divan::bench(args = LENS, sample_count = SAMPLES)]
fn baseline_lt(bencher: Bencher, len: usize) {
    compare_baseline(bencher, len, Operator::Lt);
}

#[divan::bench(args = LENS, sample_count = SAMPLES)]
fn fast_eq(bencher: Bencher, len: usize) {
    compare_fast(bencher, len, Operator::Eq);
}

#[divan::bench(args = LENS, sample_count = SAMPLES)]
fn baseline_eq(bencher: Bencher, len: usize) {
    compare_baseline(bencher, len, Operator::Eq);
}

#[divan::bench(args = LENS, sample_count = SAMPLES)]
fn fast_between(bencher: Bencher, len: usize) {
    between_fast(bencher, len);
}

#[divan::bench(args = LENS, sample_count = SAMPLES)]
fn baseline_between(bencher: Bencher, len: usize) {
    between_baseline(bencher, len);
}
