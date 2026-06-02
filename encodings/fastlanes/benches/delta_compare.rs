// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare/between an already delta-encoded `DeltaArray` against a constant.
//!
//! Each operator has two benches:
//! * `fast_*` — the block-streaming kernel that decompresses one 1024-element FastLanes chunk at a
//!   time into stack scratch and folds the predicate straight into a bit buffer.
//! * `baseline_*` — what the fallback does: materialise the full unpacked primitive on the heap,
//!   then run the Arrow compare/between over it (two passes over memory).
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

const LENS: &[usize] = &[1024, 64 * 1024, 1024 * 1024];

/// Build a delta-encoded array of a slowly varying monotone-ish sequence, plus a midpoint
/// constant RHS so the predicate selectivity is ~50%.
fn build_inputs(len: usize) -> (ArrayRef, ArrayRef, ExecutionCtx) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // Small per-element deltas keep the encoding dense; values wrap modulo a window so the
    // midpoint constant splits the array roughly in half.
    let buf: BufferMut<u32> = (0..len).map(|i| (i as u32) % 4096).collect();
    let array = Delta::try_from_primitive_array(
        &PrimitiveArray::new(buf.freeze(), Validity::NonNullable),
        &mut ctx,
    )
    .unwrap()
    .into_array();
    let rhs = ConstantArray::new(2048u32, len).into_array();
    (array, rhs, ctx)
}

#[divan::bench(args = LENS)]
fn fast_lt(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS)]
fn baseline_lt(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS)]
fn fast_eq(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS)]
fn baseline_eq(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

const NON_STRICT: BetweenOptions = BetweenOptions {
    lower_strict: StrictComparison::NonStrict,
    upper_strict: StrictComparison::NonStrict,
};

#[divan::bench(args = LENS)]
fn fast_between(bencher: Bencher, len: usize) {
    let (array, _rhs, mut ctx) = build_inputs(len);
    let lower = ConstantArray::new(1024u32, len).into_array();
    let upper = ConstantArray::new(3072u32, len).into_array();
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .between(lower.clone(), upper.clone(), NON_STRICT)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS)]
fn baseline_between(bencher: Bencher, len: usize) {
    let (array, _rhs, mut ctx) = build_inputs(len);
    let lower = ConstantArray::new(1024u32, len).into_array();
    let upper = ConstantArray::new(3072u32, len).into_array();
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
