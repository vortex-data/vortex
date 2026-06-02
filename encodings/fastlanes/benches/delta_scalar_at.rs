// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the current per-lane `Delta::scalar_at` against the previous implementation, which
//! sliced to a single element and fully canonicalized the enclosing 1,024-element chunk.

use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_error::VortexExpect;
use vortex_fastlanes::Delta;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Total element count. All are multiples of 1,024 so the last index exercises a full chunk.
const LENS: &[usize] = &[1024, 8 * 1024, 64 * 1024];

/// The previous `scalar_at`: slice to one element, then decompress the whole chunk.
///
/// The real old code was reached through `ArrayRef::execute_scalar`, whose generic nullable guard
/// (`is_invalid` -> `Delta::validity`) ran before dispatch. We replicate that guard here so the
/// comparison against the new path — which still goes through `execute_scalar` — is apples to
/// apples; otherwise the old numbers would omit a cost the old path genuinely paid.
fn old_scalar_at(array: &ArrayRef, index: usize, ctx: &mut ExecutionCtx) -> Scalar {
    if array.dtype().is_nullable() && array.is_invalid(index, ctx).vortex_expect("is_invalid") {
        return Scalar::null(array.dtype().clone());
    }
    let decompressed = array
        .slice(index..index + 1)
        .vortex_expect("slice")
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("execute");
    decompressed
        .into_array()
        .execute_scalar(0, ctx)
        .vortex_expect("scalar")
}

/// Build a Delta array of `len` cumulative-sum `u32` values.
fn delta_u32(len: usize, nullable: bool) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let mut acc: u32 = 0;
    let prim = if nullable {
        PrimitiveArray::from_option_iter((0..len).map(|i| {
            acc = acc.wrapping_add(rng.random_range(0..16));
            (i % 7 != 0).then_some(acc)
        }))
    } else {
        PrimitiveArray::from_iter((0..len).map(|_| {
            acc = acc.wrapping_add(rng.random_range(0..16));
            acc
        }))
    };
    Delta::try_from_primitive_array(&prim, &mut SESSION.create_execution_ctx())
        .vortex_expect("compress")
        .into_array()
}

/// Build a Delta array of `len` cumulative-sum `i64` values (crosses zero).
fn delta_i64(len: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(1);
    let mut acc: i64 = 0;
    let prim = PrimitiveArray::from_iter((0..len).map(|_| {
        acc = acc.wrapping_add(rng.random_range(-8..8));
        acc
    }));
    Delta::try_from_primitive_array(&prim, &mut SESSION.create_execution_ctx())
        .vortex_expect("compress")
        .into_array()
}

// Index decoded by every bench: the final element of the array (a full, last chunk).
fn idx(len: usize) -> usize {
    len - 1
}

#[divan::bench(args = LENS)]
fn u32_nonnull_old(bencher: Bencher, len: usize) {
    let array = delta_u32(len, false);
    let index = idx(len);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| black_box(old_scalar_at(&array, black_box(index), &mut ctx)));
}

#[divan::bench(args = LENS)]
fn u32_nonnull_new(bencher: Bencher, len: usize) {
    let array = delta_u32(len, false);
    let index = idx(len);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| {
            black_box(
                array
                    .execute_scalar(black_box(index), &mut ctx)
                    .vortex_expect("scalar"),
            )
        });
}

#[divan::bench(args = LENS)]
fn u32_nullable_old(bencher: Bencher, len: usize) {
    let array = delta_u32(len, true);
    let index = idx(len);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| black_box(old_scalar_at(&array, black_box(index), &mut ctx)));
}

#[divan::bench(args = LENS)]
fn u32_nullable_new(bencher: Bencher, len: usize) {
    let array = delta_u32(len, true);
    let index = idx(len);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| {
            black_box(
                array
                    .execute_scalar(black_box(index), &mut ctx)
                    .vortex_expect("scalar"),
            )
        });
}

#[divan::bench(args = LENS)]
fn i64_nonnull_old(bencher: Bencher, len: usize) {
    let array = delta_i64(len);
    let index = idx(len);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| black_box(old_scalar_at(&array, black_box(index), &mut ctx)));
}

#[divan::bench(args = LENS)]
fn i64_nonnull_new(bencher: Bencher, len: usize) {
    let array = delta_i64(len);
    let index = idx(len);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| {
            black_box(
                array
                    .execute_scalar(black_box(index), &mut ctx)
                    .vortex_expect("scalar"),
            )
        });
}
