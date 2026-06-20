// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the Sparse pushdown kernels (`is_constant`, `sum`, `min_max`,
//! `null_count`, compare).
//!
//! Each benchmark exercises the registered kernel path on a single representative
//! sparse `i32` array. All are `O(num_patches)`; the patch counts below are sized so
//! each lands in the ~10-100µs range for a stable CodSpeed signal. `between`/`fill_null`/
//! `nan_count` are omitted since they mirror the compare/null_count cost profiles.

#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::null_count::null_count;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

fn main() {
    divan::main();
}

const LEN: usize = 1_000_000;

/// Session with Sparse and its pushdown kernels registered.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_sparse::initialize(&session);
    session
});

/// Build a sparse `i32` array of `LEN` with `num_patches` uniformly-spaced patches and
/// fill value 1. When `constant` is true every patch also equals 1, so the whole array
/// is constant (the worst case for `is_constant`: it must scan all patches to confirm).
fn make_sparse(num_patches: usize, constant: bool) -> ArrayRef {
    let stride = LEN / num_patches;
    let indices: Buffer<u32> = (0..num_patches).map(|i| (i * stride) as u32).collect();
    let values: Buffer<i32> = (0..num_patches)
        .map(|i| if constant { 1 } else { 2 + i as i32 })
        .collect();
    Sparse::try_new(
        indices.into_array(),
        values.into_array(),
        LEN,
        Scalar::from(1i32),
    )
    .vortex_expect("valid sparse")
    .into_array()
}

/// Build a sparse `i32` array of `LEN` with a null fill and `num_patches` nullable patches
/// (every third patch null), so `null_count` does real `O(P)` work over the patch validity.
fn make_sparse_nullable(num_patches: usize) -> ArrayRef {
    let stride = LEN / num_patches;
    let indices: Buffer<u32> = (0..num_patches).map(|i| (i * stride) as u32).collect();
    let values = PrimitiveArray::from_option_iter(
        (0..num_patches).map(|i| if i % 3 == 0 { None } else { Some(i as i32) }),
    )
    .into_array();
    let nullable = DType::Primitive(PType::I32, Nullability::Nullable);
    Sparse::try_new(indices.into_array(), values, LEN, Scalar::null(nullable))
        .vortex_expect("valid sparse")
        .into_array()
}

#[divan::bench]
fn sparse_is_constant(bencher: Bencher) {
    bencher
        .with_inputs(|| (make_sparse(100_000, true), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(is_constant(&array, &mut ctx).vortex_expect("is_constant"))
        });
}

#[divan::bench]
fn sparse_sum(bencher: Bencher) {
    bencher
        .with_inputs(|| (make_sparse(100_000, false), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(sum(&array, &mut ctx).vortex_expect("sum"))
        });
}

#[divan::bench]
fn sparse_min_max(bencher: Bencher) {
    bencher
        .with_inputs(|| (make_sparse(40_000, false), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(
                min_max(&array, &mut ctx, NumericalAggregateOpts::default())
                    .vortex_expect("min_max"),
            )
        });
}

#[divan::bench]
fn sparse_null_count(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            (
                make_sparse_nullable(130_000),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            divan::black_box(null_count(&array, &mut ctx).vortex_expect("null_count"))
        });
}

#[divan::bench]
fn sparse_compare(bencher: Bencher) {
    bencher
        .with_inputs(|| (make_sparse(10_000, false), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let rhs = ConstantArray::new(Scalar::from(1i32), array.len()).into_array();
            let result = array.binary(rhs, Operator::Eq).vortex_expect("binary");
            divan::black_box(materialize(result, &mut ctx))
        });
}

fn materialize(array: ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    array
        .execute::<Canonical>(ctx)
        .vortex_expect("execute")
        .into_array()
}
