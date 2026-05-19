// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks measuring the wins from pushdown kernels on Sparse arrays.
//!
//! For each kernel we compare the registered Sparse-aware path (`with_kernel`) against
//! the baseline canonical-fallback path (`canonical`) using the same input. The session
//! difference is the only knob: the canonical baseline runs against a session in which
//! Sparse has no aggregate/compare kernel registered, forcing the accumulator to
//! materialize the full array.

#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

fn main() {
    divan::main();
}

/// Session with Sparse encoding registered but no Sparse-specific kernels.
/// This is the "before" path: dispatch falls through to canonical materialization.
static CANONICAL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    session.arrays().register(Sparse);
    session
});

/// Session with Sparse encoding *and* its pushdown kernels registered.
/// This is the "after" path.
static KERNEL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_sparse::initialize(&session);
    session
});

/// (array_len, num_patches)
const ARGS: &[(usize, usize)] = &[
    (1_000_000, 10),     // 0.001% patches → 10⁵× upside ceiling
    (1_000_000, 1_000),  // 0.1% patches
    (1_000_000, 10_000), // 1% patches
    (100_000, 10),       // 0.01% patches
];

/// Build a sparse i32 array of `len` with `num_patches` uniformly-spaced patches.
/// Fill is a non-null constant (1), patches are increasing values (2, 3, …) so the
/// array is NOT constant — exercises the full-comparison path of the kernel.
fn make_sparse_i32(len: usize, num_patches: usize) -> ArrayRef {
    assert!(num_patches > 0 && num_patches <= len);
    let stride = len / num_patches;
    let indices: Buffer<u32> = (0..num_patches).map(|i| (i * stride) as u32).collect();
    let values: Buffer<i32> = (0..num_patches as i32).map(|i| 2 + i).collect();
    Sparse::try_new(
        indices.into_array(),
        values.into_array(),
        len,
        Scalar::from(1i32),
    )
    .vortex_expect("valid sparse")
    .into_array()
}

// ---------- is_constant ----------

#[divan::bench(args = ARGS)]
fn is_constant_canonical(bencher: Bencher, (len, np): (usize, usize)) {
    bencher
        .with_inputs(|| {
            (
                make_sparse_i32(len, np),
                CANONICAL_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            divan::black_box(is_constant(&array, &mut ctx).vortex_expect("is_constant"))
        });
}

#[divan::bench(args = ARGS)]
fn is_constant_with_kernel(bencher: Bencher, (len, np): (usize, usize)) {
    bencher
        .with_inputs(|| {
            (
                make_sparse_i32(len, np),
                KERNEL_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            divan::black_box(is_constant(&array, &mut ctx).vortex_expect("is_constant"))
        });
}

// ---------- sum ----------

#[divan::bench(args = ARGS)]
fn sum_canonical(bencher: Bencher, (len, np): (usize, usize)) {
    bencher
        .with_inputs(|| {
            (
                make_sparse_i32(len, np),
                CANONICAL_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            divan::black_box(sum(&array, &mut ctx).vortex_expect("sum"))
        });
}

#[divan::bench(args = ARGS)]
fn sum_with_kernel(bencher: Bencher, (len, np): (usize, usize)) {
    bencher
        .with_inputs(|| {
            (
                make_sparse_i32(len, np),
                KERNEL_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| {
            divan::black_box(sum(&array, &mut ctx).vortex_expect("sum"))
        });
}

// ---------- compare (Sparse == constant) ----------
//
// NOTE: `CompareExecuteAdaptor(Sparse)` is registered in `PARENT_KERNELS`, which is
// statically attached to the Sparse encoding vtable (not session-scoped). To benchmark
// the "no-kernel" baseline we explicitly canonicalize the input first so the comparison
// runs against a `PrimitiveArray`. The kernel path lets the comparison push through.

fn compare_with_pushdown(array: ArrayRef, mut ctx: vortex_array::ExecutionCtx) {
    let rhs = ConstantArray::new(Scalar::from(1i32), array.len()).into_array();
    let result = array
        .binary(rhs, Operator::Eq)
        .vortex_expect("binary build");
    divan::black_box(
        result
            .execute::<vortex_array::Canonical>(&mut ctx)
            .vortex_expect("execute"),
    );
}

fn compare_after_canonicalize(array: ArrayRef, mut ctx: vortex_array::ExecutionCtx) {
    let canonical = array
        .execute::<vortex_array::Canonical>(&mut ctx)
        .vortex_expect("canonicalize")
        .into_array();
    let rhs = ConstantArray::new(Scalar::from(1i32), canonical.len()).into_array();
    let result = canonical
        .binary(rhs, Operator::Eq)
        .vortex_expect("binary build");
    divan::black_box(
        result
            .execute::<vortex_array::Canonical>(&mut ctx)
            .vortex_expect("execute"),
    );
}

#[divan::bench(args = ARGS)]
fn compare_canonical(bencher: Bencher, (len, np): (usize, usize)) {
    bencher
        .with_inputs(|| {
            (
                make_sparse_i32(len, np),
                KERNEL_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, ctx)| compare_after_canonicalize(array, ctx));
}

#[divan::bench(args = ARGS)]
fn compare_with_kernel(bencher: Bencher, (len, np): (usize, usize)) {
    bencher
        .with_inputs(|| {
            (
                make_sparse_i32(len, np),
                KERNEL_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, ctx)| compare_with_pushdown(array, ctx));
}
