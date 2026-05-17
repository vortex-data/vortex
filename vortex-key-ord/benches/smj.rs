// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort-merge join microbench. Two encoding pairings, each compared
//! against the row-by-row scalar_at baseline.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![allow(deprecated)]

use std::hint::black_box;

use divan::Bencher;
use divan::counter::ItemsCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_key_ord::smj::CONST_CONST_SMJ;
use vortex_key_ord::smj::PRIM_PRIM_SMJ;
use vortex_key_ord::smj::smj_naive;

fn main() {
    divan::main();
}

const N: usize = 50_000;
/// Sized so the 1000x1000 Cartesian (1M pairs) is large enough to expose
/// the encoding-aware-output speedup.
const CCN: usize = 1000;

fn make_sorted_with_overlap(seed: u64) -> (PrimitiveArray, ArrayRef) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<u64> = (0..N).map(|_| rng.random_range(0u64..(N as u64))).collect();
    data.sort_unstable();
    let typed = PrimitiveArray::new(Buffer::<u64>::copy_from(&data), Validity::NonNullable);
    let erased = typed.clone().into_array();
    (typed, erased)
}

// ---------- Primitive x Primitive ---------------------------------------

#[divan::bench(sample_count = 30)]
fn prim_prim_naive(bencher: Bencher) {
    let (_, l) = make_sorted_with_overlap(1);
    let (_, r) = make_sorted_with_overlap(2);
    bencher
        .counter(ItemsCount::new(2 * N))
        .bench_local(|| black_box(smj_naive(black_box(&l), black_box(&r))));
}

#[divan::bench(sample_count = 30)]
fn prim_prim_via_pks(bencher: Bencher) {
    let (l, _) = make_sorted_with_overlap(1);
    let (_, re) = make_sorted_with_overlap(2);
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(2 * N))
        .bench_local(|| {
            black_box(
                PRIM_PRIM_SMJ
                    .execute(l.as_view(), black_box(&re), &mut ctx)
                    .expect("execute"),
            )
        });
}

// ---------- Constant x Constant (encoding-aware Cartesian) --------------

#[divan::bench(sample_count = 30)]
fn const_const_naive(bencher: Bencher) {
    let l = ConstantArray::new(42u64, CCN).into_array();
    let r = ConstantArray::new(42u64, CCN).into_array();
    bencher
        .counter(ItemsCount::new(2 * CCN))
        .bench_local(|| black_box(smj_naive(black_box(&l), black_box(&r))));
}

#[divan::bench(sample_count = 30)]
fn const_const_via_pks(bencher: Bencher) {
    let l = ConstantArray::new(42u64, CCN);
    let r = ConstantArray::new(42u64, CCN).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(2 * CCN))
        .bench_local(|| {
            black_box(
                CONST_CONST_SMJ
                    .execute(l.as_view(), black_box(&r), &mut ctx)
                    .expect("execute"),
            )
        });
}
