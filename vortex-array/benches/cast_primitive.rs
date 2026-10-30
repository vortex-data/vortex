// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive casts over dense and nullable inputs.
//!
//! Integer widening cases run on the walltime CPU-feature legs to track their generated loops.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::prelude::*;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Stat;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

// Sizes used for the fallible-path benches below. Kept small enough to fit in L2 so
// the kernel cost shows up clearly rather than being hidden by DRAM bandwidth, and
// to keep the CodSpeed simulation under 1ms per benchmark.
const SIZES: &[usize] = &[8_192];

const WIDEN_SIZES: &[usize] = &[65, 128, 129, 8_192, 16_384];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
});

#[vortex_bench_support::cpu_features]
#[divan::bench(args = WIDEN_SIZES)]
fn cast_u16_to_u32(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter((0..n).map(|i| {
        #[expect(clippy::cast_possible_truncation)]
        rng.random_bool(0.5).then_some(i as u16)
    }))
    .into_array();
    // Pre-compute min/max so values_fit_in is a cache hit during the benchmark.
    arr.statistics()
        .compute_all(&[Stat::Min, Stat::Max], &mut SESSION.create_execution_ctx())
        .ok();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::U32, Nullability::Nullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = WIDEN_SIZES)]
fn cast_u16_to_u32_dense(bencher: Bencher, n: usize) {
    let arr = PrimitiveArray::from_iter((0..n).map(|i| u16::try_from(i).unwrap())).into_array();

    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::U32, Nullability::NonNullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = WIDEN_SIZES)]
fn cast_i32_to_i64_dense(bencher: Bencher, n: usize) {
    let arr =
        PrimitiveArray::from_iter((0..n).map(|i| i32::try_from(i).unwrap() - 8_192)).into_array();

    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::I64, Nullability::NonNullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = WIDEN_SIZES)]
fn cast_i32_to_i64_nullable(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter((0..n).map(|i| {
        rng.random_bool(0.5)
            .then_some(i32::try_from(i).unwrap() - 8_192)
    }))
    .into_array();

    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::I64, Nullability::Nullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

/// Narrowing fallible cast that goes through `try_map_with_mask`. Inputs are bounded
/// so every value fits, isolating the kernel's per-lane checked-cast overhead.
#[divan::bench(args = SIZES)]
fn cast_u32_to_u8(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let arr = PrimitiveArray::from_option_iter((0..n).map(|_| {
        rng.random_bool(0.7)
            .then(|| rng.random_range(0..u8::MAX) as u32)
    }))
    .into_array();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::U8, Nullability::Nullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

/// Sign-change cast i32 -> u32. Values are non-negative so min/max range scan proves
/// they fit and cast reinterprets the buffer in O(1)
#[divan::bench(args = SIZES)]
fn cast_i32_to_u32(bencher: Bencher, n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let parts = PrimitiveArray::from_option_iter(
        (0..n).map(|_| rng.random_bool(0.7).then(|| rng.random_range(0..i32::MAX))),
    )
    .into_data_parts();
    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_buffer_handle(
                    parts.buffer.clone(),
                    parts.ptype,
                    parts.validity.clone(),
                )
                .into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Primitive(PType::U32, Nullability::Nullable))
                .unwrap()
                .execute::<Canonical>(ctx)
        });
}

/// Integer-to-decimal cast at the largest precision that uses an i128 backing buffer. This
/// exercises the common rescaling path without paying for i256 arithmetic per input value.
#[divan::bench(args = SIZES)]
fn cast_i64_to_decimal38_scale2(bencher: Bencher, n: usize) {
    let arr =
        PrimitiveArray::from_iter((0..n).map(|value| i64::try_from(value).unwrap())).into_array();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Decimal(
                DecimalDType::new(38, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .execute::<Canonical>(ctx)
        });
}

/// Common integer-to-decimal cast that rescales directly in its i32 output buffer.
#[divan::bench(args = SIZES)]
fn cast_i32_to_decimal9_scale2(bencher: Bencher, n: usize) {
    let arr =
        PrimitiveArray::from_iter((0..n).map(|value| i32::try_from(value).unwrap())).into_array();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Decimal(
                DecimalDType::new(9, 2),
                Nullability::NonNullable,
            ))
            .unwrap()
            .execute::<Canonical>(ctx)
        });
}

/// Same-width scale-zero cast that validates then reuses the source values buffer.
#[divan::bench(args = SIZES)]
fn cast_i32_to_decimal9_scale0(bencher: Bencher, n: usize) {
    let arr =
        PrimitiveArray::from_iter((0..n).map(|value| i32::try_from(value).unwrap())).into_array();
    bencher
        .with_inputs(|| (arr.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.cast(DType::Decimal(
                DecimalDType::new(9, 0),
                Nullability::NonNullable,
            ))
            .unwrap()
            .execute::<Canonical>(ctx)
        });
}
