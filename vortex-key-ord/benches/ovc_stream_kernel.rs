// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OVC microbench: row-by-row `scalar_at` baseline vs encoding-aware
//! kernel (direct call vs via [`ParentKernelSet`]).
//!
//! Each encoding has three rows:
//!   * `*_naive_scalar_at` -- polymorphic baseline.
//!   * `*_direct`          -- monomorphic call to the kernel body.
//!   * `*_via_pks`         -- through the registered `ParentKernelSet`.

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
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_key_ord::stream_kernel::CONSTANT_OVC_KERNELS;
use vortex_key_ord::stream_kernel::DICT_OVC_KERNELS;
use vortex_key_ord::stream_kernel::OvcKernel;
use vortex_key_ord::stream_kernel::PRIMITIVE_OVC_KERNELS;

fn main() {
    divan::main();
}

const N: usize = 100_000;

fn make_primitive() -> (PrimitiveArray, ArrayRef) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut data: Vec<u64> = (0..N).map(|_| rng.random()).collect();
    data.sort_unstable();
    let typed = PrimitiveArray::new(Buffer::<u64>::copy_from(&data), Validity::NonNullable);
    let erased = typed.clone().into_array();
    (typed, erased)
}

fn make_constant() -> (ConstantArray, ArrayRef) {
    let typed = ConstantArray::new(42u64, N);
    let erased = typed.clone().into_array();
    (typed, erased)
}

fn make_dict(dict_size: usize) -> (DictArray, ArrayRef) {
    let values_buf: Vec<u64> = (0..dict_size as u64).map(|v| v * 1000).collect();
    let codes_buf: Vec<u32> = (0..N).map(|i| (i % dict_size) as u32).collect();
    let values = PrimitiveArray::new(
        Buffer::<u64>::copy_from(&values_buf),
        Validity::NonNullable,
    )
    .into_array();
    let codes =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&codes_buf), Validity::NonNullable)
            .into_array();
    let typed = DictArray::new(codes, values);
    let erased = typed.clone().into_array();
    (typed, erased)
}

/// Naive walk: `scalar_at` per row + a fake OVC computation, to model
/// what a query engine without an encoding-aware kernel would do.
fn naive_scalar_at(arr: &ArrayRef) -> u64 {
    let mut prev = 0u64;
    let mut last = 0u64;
    for i in 0..arr.len() {
        let v = u64::try_from(&arr.scalar_at(i).expect("scalar_at")).expect("u64");
        let _diff = prev ^ v;
        prev = v;
        last = v;
    }
    last
}

// ---------- Primitive ----------------------------------------------------

#[divan::bench(sample_count = 30)]
fn primitive_naive_scalar_at(bencher: Bencher) {
    let (_, erased) = make_primitive();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| black_box(naive_scalar_at(black_box(&erased))));
}

#[divan::bench(sample_count = 30)]
fn primitive_direct(bencher: Bencher) {
    let (typed, _) = make_primitive();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        black_box(<Primitive as OvcKernel>::ovc_encode(typed.as_view(), 0))
    });
}

#[divan::bench(sample_count = 30)]
fn primitive_via_pks(bencher: Bencher) {
    let (typed, erased) = make_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        black_box(
            PRIMITIVE_OVC_KERNELS
                .execute(typed.as_view(), black_box(&erased), 0, &mut ctx)
                .expect("execute"),
        )
    });
}

// ---------- Constant ----------------------------------------------------

#[divan::bench(sample_count = 30)]
fn constant_naive_scalar_at(bencher: Bencher) {
    let (_, erased) = make_constant();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| black_box(naive_scalar_at(black_box(&erased))));
}

#[divan::bench(sample_count = 30)]
fn constant_direct(bencher: Bencher) {
    let (typed, _) = make_constant();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        black_box(<Constant as OvcKernel>::ovc_encode(typed.as_view(), 0))
    });
}

#[divan::bench(sample_count = 30)]
fn constant_via_pks(bencher: Bencher) {
    let (typed, erased) = make_constant();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        black_box(
            CONSTANT_OVC_KERNELS
                .execute(typed.as_view(), black_box(&erased), 0, &mut ctx)
                .expect("execute"),
        )
    });
}

// ---------- Dict (varies dict_size to surface the O(dict_size) curve) ----

#[divan::bench(args = [4usize, 64, 1024], sample_count = 30)]
fn dict_via_pks(bencher: Bencher, dict_size: usize) {
    let (typed, erased) = make_dict(dict_size);
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        black_box(
            DICT_OVC_KERNELS
                .execute(typed.as_view(), black_box(&erased), 0, &mut ctx)
                .expect("execute"),
        )
    });
}
