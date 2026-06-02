// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare decoding a `delta(for(bitpacking))` stack two ways:
//!   * `fused`   — the fused `Delta::unfor_undelta_pack` kernel (one pass over the packed buffer).
//!   * `unfused` — materialize the FoR(bitpacked) deltas child to a primitive buffer, then run the
//!     generic delta decode over it (two passes, two intermediate buffers).
//!
//! Both decode the same non-strictly-increasing (monotone non-decreasing) integer column.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench delta_for_bitpack`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::{Delta, FoR, FoRArrayExt, delta_compress};

fn main() {
    divan::main();
}

// Exact multiples of 1024 so the deltas bit-pack without a zero-padding wrap.
const LENS: &[usize] = &[64 * 1024, 1024 * 1024];

/// Build the `delta(for(bitpacking))` stack and return both the fused root array and the pieces
/// needed to reconstruct an unfused decode (the bases child and the FoR(bitpacked) deltas child).
fn build(values: PrimitiveArray) -> (ArrayRef, ArrayRef, ArrayRef, usize, ExecutionCtx) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let len = values.len();

    let (bases, deltas) = delta_compress(&values, &mut ctx).unwrap();
    let for_deltas = FoR::encode(deltas).unwrap();
    let reference = for_deltas.reference_scalar().clone();
    let for_encoded = for_deltas
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();

    // Smallest width that captures every value, so bit-packing introduces no patches.
    let unsigned = for_encoded.ptype().to_unsigned();
    let bit_width = match_each_unsigned_integer_ptype!(unsigned, |T| {
        let reinterpreted = for_encoded.reinterpret_cast(unsigned);
        let max = reinterpreted
            .as_slice::<T>()
            .iter()
            .copied()
            .max()
            .unwrap_or_default();
        (T::BITS - max.leading_zeros()) as u8
    });
    let bitpacked = bitpack_encode(&for_encoded, bit_width, None, &mut ctx).unwrap();

    let bases = bases.into_array();
    let for_child = FoR::try_new(bitpacked.into_array(), reference)
        .unwrap()
        .into_array();
    let fused = Delta::try_new(bases.clone(), for_child.clone(), 0, len)
        .unwrap()
        .into_array();
    (fused, bases, for_child, len, ctx)
}

fn u32_non_decreasing(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len as u32).map(|i| i / 4))
}

fn u64_non_decreasing(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len as u64).map(|i| (i / 6) * 3))
}

#[divan::bench(args = LENS)]
fn fused_u32(bencher: Bencher, len: usize) {
    let (fused, _, _, n, mut ctx) = build(u32_non_decreasing(len));
    bencher
        .counter(ItemsCount::new(n))
        .bench_local(|| fused.clone().execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = LENS)]
fn unfused_u32(bencher: Bencher, len: usize) {
    let (_, bases, for_child, n, mut ctx) = build(u32_non_decreasing(len));
    bencher.counter(ItemsCount::new(n)).bench_local(|| {
        // Pass 1: unpack + un-FoR the deltas into a materialized primitive buffer.
        let deltas = for_child
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        // Pass 2: generic delta decode (un-delta + untranspose) over the materialized deltas.
        Delta::try_new(bases.clone(), deltas.into_array(), 0, n)
            .unwrap()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS)]
fn fused_u64(bencher: Bencher, len: usize) {
    let (fused, _, _, n, mut ctx) = build(u64_non_decreasing(len));
    bencher
        .counter(ItemsCount::new(n))
        .bench_local(|| fused.clone().execute::<PrimitiveArray>(&mut ctx).unwrap());
}

#[divan::bench(args = LENS)]
fn unfused_u64(bencher: Bencher, len: usize) {
    let (_, bases, for_child, n, mut ctx) = build(u64_non_decreasing(len));
    bencher.counter(ItemsCount::new(n)).bench_local(|| {
        let deltas = for_child
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        Delta::try_new(bases.clone(), deltas.into_array(), 0, n)
            .unwrap()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
    });
}
