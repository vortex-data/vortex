// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A/B decode of a `delta(for(bitpacking))` column, both arms going through the real Vortex decode
//! entry points on the *same* array:
//!   * `fused`   — `delta_decompress` with the fused `unfor_undelta_pack` fast path.
//!   * `current` — `delta_decompress_generic`, the path Vortex took before the fused kernel:
//!     materialize the FoR(bitpacked) deltas child, then un-delta + untranspose.
//!
//! The column is non-strictly-increasing (monotone non-decreasing) so it compresses as
//! delta(for(bitpacking)).
//!
//! Run with `cargo bench -p vortex-fastlanes --bench delta_for_bitpack
//!   --features unstable_encodings,_test-harness`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_fastlanes::Delta;
use vortex_fastlanes::DeltaArray;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::delta_compress;
use vortex_fastlanes::delta_decompress;
use vortex_fastlanes::delta_decompress_generic;

fn main() {
    divan::main();
}

// Exact multiples of 1024 so the deltas bit-pack without a zero-padding wrap.
const LENS: &[usize] = &[64 * 1024, 1024 * 1024];

/// Build the `delta(for(bitpacking))` stack for `values`.
fn build(values: PrimitiveArray) -> (DeltaArray, usize, ExecutionCtx) {
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

    let for_child = FoR::try_new(bitpacked.into_array(), reference)
        .unwrap()
        .into_array();
    let array = Delta::try_new(bases.into_array(), for_child, 0, len).unwrap();
    (array, len, ctx)
}

fn u32_non_decreasing(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len as u32).map(|i| i / 4))
}

fn u64_non_decreasing(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len as u64).map(|i| (i / 6) * 3))
}

#[divan::bench(args = LENS)]
fn fused_u32(bencher: Bencher, len: usize) {
    let (array, n, mut ctx) = build(u32_non_decreasing(len));
    bencher
        .counter(ItemsCount::new(n))
        .bench_local(|| delta_decompress(&array, &mut ctx).unwrap());
}

#[divan::bench(args = LENS)]
fn current_u32(bencher: Bencher, len: usize) {
    let (array, n, mut ctx) = build(u32_non_decreasing(len));
    bencher
        .counter(ItemsCount::new(n))
        .bench_local(|| delta_decompress_generic(&array, &mut ctx).unwrap());
}

#[divan::bench(args = LENS)]
fn fused_u64(bencher: Bencher, len: usize) {
    let (array, n, mut ctx) = build(u64_non_decreasing(len));
    bencher
        .counter(ItemsCount::new(n))
        .bench_local(|| delta_decompress(&array, &mut ctx).unwrap());
}

#[divan::bench(args = LENS)]
fn current_u64(bencher: Bencher, len: usize) {
    let (array, n, mut ctx) = build(u64_non_decreasing(len));
    bencher
        .counter(ItemsCount::new(n))
        .bench_local(|| delta_decompress_generic(&array, &mut ctx).unwrap());
}
