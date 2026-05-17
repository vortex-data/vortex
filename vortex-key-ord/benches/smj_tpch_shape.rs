// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-H-shaped SMJ benchmarks. Synthetic sorted u64 keys with
//! realistic cardinalities + duplicate-run distributions.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]
#![allow(deprecated)]

use std::hint::black_box;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_key_ord::smj::PRIM_PRIM_SMJ;
use vortex_key_ord::smj::smj_naive;

fn main() {
    divan::main();
}

/// Sorted u64 key column with `n_keys` distinct keys, each appearing
/// `dup_per_key` times in run.
fn keys_with_dups(n_keys: u64, dup_per_key: usize) -> (PrimitiveArray, ArrayRef) {
    let total = (n_keys as usize) * dup_per_key;
    let mut data = Vec::<u64>::with_capacity(total);
    for k in 0..n_keys {
        for _ in 0..dup_per_key {
            data.push(k);
        }
    }
    let typed = PrimitiveArray::new(Buffer::<u64>::copy_from(&data), Validity::NonNullable);
    let erased = typed.clone().into_array();
    (typed, erased)
}

fn unique_keys(total: usize, base: u64) -> (PrimitiveArray, ArrayRef) {
    let data: Vec<u64> = (0..total as u64).map(|i| i + base).collect();
    let typed = PrimitiveArray::new(Buffer::<u64>::copy_from(&data), Validity::NonNullable);
    let erased = typed.clone().into_array();
    (typed, erased)
}

fn bench_naive(bencher: Bencher, l: ArrayRef, r: ArrayRef) {
    let total = l.len() + r.len();
    bencher
        .counter(ItemsCount::new(total))
        .bench_local(|| black_box(smj_naive(black_box(&l), black_box(&r))));
}

fn bench_kernel(bencher: Bencher, l: PrimitiveArray, r: ArrayRef) {
    let total = l.len() + r.len();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(total))
        .bench_local(|| {
            black_box(
                PRIM_PRIM_SMJ
                    .execute(l.as_view(), black_box(&r), &mut ctx)
                    .expect("execute"),
            )
        });
}

/// lineitem ⋈ orders on orderkey: 600k ⋈ 150k, 4 lineitems per orderkey.
#[divan::bench(sample_count = 10)]
fn lineitem_orders_naive(bencher: Bencher) {
    let (_, l) = keys_with_dups(150_000, 4);
    let (_, o) = unique_keys(150_000, 0);
    bench_naive(bencher, l, o);
}

#[divan::bench(sample_count = 10)]
fn lineitem_orders_kernel(bencher: Bencher) {
    let (l, _) = keys_with_dups(150_000, 4);
    let (_, o) = unique_keys(150_000, 0);
    bench_kernel(bencher, l, o);
}

/// customer ⋈ orders on custkey: 15k ⋈ 150k, 10 orders per custkey.
#[divan::bench(sample_count = 10)]
fn customer_orders_naive(bencher: Bencher) {
    let (_, c) = unique_keys(15_000, 0);
    let (_, o) = keys_with_dups(15_000, 10);
    bench_naive(bencher, c, o);
}

#[divan::bench(sample_count = 10)]
fn customer_orders_kernel(bencher: Bencher) {
    let (c, _) = unique_keys(15_000, 0);
    let (_, o) = keys_with_dups(15_000, 10);
    bench_kernel(bencher, c, o);
}

/// 0 matches: fast empty-output path.
#[divan::bench(sample_count = 10)]
fn disjoint_kernel(bencher: Bencher) {
    let (l, _) = unique_keys(100_000, 0);
    let (_, r) = unique_keys(100_000, 500_000);
    bench_kernel(bencher, l, r);
}
