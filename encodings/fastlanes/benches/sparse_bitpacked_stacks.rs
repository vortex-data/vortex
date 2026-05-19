// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tree-vs-canonical benches for `BitPacked<u32>` and `Sparse<i64>`.
//!
//! Operations:
//! - **predicate**: `col > X` -> canonical Bool[N]
//! - **is_constant**: aggregate
//! - **sum**: aggregate
//!
//! Built with `target-cpu=native` for AVX-512 in the hot paths.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPackedData;
use vortex_sparse::Sparse;

fn main() {
    divan::main();
}

const N: usize = 1_000_000;

// =====================================================================================
// BitPacked<u32>: dense values squeezed into BIT_WIDTH bits per element
// =====================================================================================

/// Build a BitPacked<u32>[N] where values cycle through `0..(1<<bit_width)`.
fn build_bitpacked(n: usize, bit_width: u8) -> vortex_array::ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let range = 1u32 << bit_width;
    let buf: Buffer<u32> = (0..n).map(|i| (i as u32) % range).collect();
    let prim = PrimitiveArray::new(buf, Validity::NonNullable).into_array();
    BitPackedData::encode(&prim, bit_width, &mut ctx)
        .unwrap()
        .into_array()
}

const BIT_WIDTHS: &[u8] = &[4, 12];

#[divan::bench(args = BIT_WIDTHS, sample_count = 30)]
fn predicate_bitpacked_tree(bencher: Bencher, bw: u8) {
    let col = build_bitpacked(N, bw);
    // RHS is in-range so the kernel actually runs (no out-of-range fast path).
    let rhs_value = (1u32 << bw) / 2;
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let rhs = ConstantArray::new(rhs_value, N).into_array();
            let cmp = col.binary(rhs, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = BIT_WIDTHS, sample_count = 30)]
fn predicate_bitpacked_canonical(bencher: Bencher, bw: u8) {
    let col = build_bitpacked(N, bw);
    let rhs_value = (1u32 << bw) / 2;
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            let rhs = ConstantArray::new(rhs_value, N).into_array();
            let cmp = canon.binary(rhs, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = BIT_WIDTHS, sample_count = 30)]
fn is_constant_bitpacked_tree(bencher: Bencher, bw: u8) {
    bencher
        .with_inputs(|| {
            (
                build_bitpacked(N, bw),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| is_constant(&col, &mut ctx).unwrap());
}

#[divan::bench(args = BIT_WIDTHS, sample_count = 30)]
fn is_constant_bitpacked_canonical(bencher: Bencher, bw: u8) {
    bencher
        .with_inputs(|| {
            (
                build_bitpacked(N, bw),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            is_constant(&canon, &mut ctx).unwrap()
        });
}

#[divan::bench(args = BIT_WIDTHS, sample_count = 30)]
fn sum_bitpacked_tree(bencher: Bencher, bw: u8) {
    bencher
        .with_inputs(|| {
            (
                build_bitpacked(N, bw),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| sum(&col, &mut ctx).unwrap());
}

#[divan::bench(args = BIT_WIDTHS, sample_count = 30)]
fn sum_bitpacked_canonical(bencher: Bencher, bw: u8) {
    bencher
        .with_inputs(|| {
            (
                build_bitpacked(N, bw),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            sum(&canon, &mut ctx).unwrap()
        });
}

// =====================================================================================
// Sparse<i64>: fill_value + patches at given indices
// =====================================================================================

/// Build a SparseArray<i64> of length `n`, fill_value=0, with `num_patches` patches at
/// evenly-spaced indices.
fn build_sparse(n: usize, num_patches: usize) -> vortex_array::ArrayRef {
    let step = n / num_patches.max(1);
    let indices_buf: Buffer<u32> = (0..num_patches).map(|i| (i * step) as u32).collect();
    let values_buf: Buffer<i64> = (0..num_patches).map(|i| (i as i64) + 1).collect();
    let indices = PrimitiveArray::new(indices_buf, Validity::NonNullable).into_array();
    let values = PrimitiveArray::new(values_buf, Validity::NonNullable).into_array();
    Sparse::try_new(indices, values, n, Scalar::from(0i64))
        .unwrap()
        .into_array()
}

// 1% patch density = 10k patches in 1M, 10% = 100k, 50% = 500k
const PATCH_COUNTS: &[usize] = &[10_000, 100_000];

#[divan::bench(args = PATCH_COUNTS, sample_count = 30)]
fn predicate_sparse_tree(bencher: Bencher, num_patches: usize) {
    let col = build_sparse(N, num_patches);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let rhs = ConstantArray::new(0i64, N).into_array();
            let cmp = col.binary(rhs, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = PATCH_COUNTS, sample_count = 30)]
fn predicate_sparse_canonical(bencher: Bencher, num_patches: usize) {
    let col = build_sparse(N, num_patches);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            let rhs = ConstantArray::new(0i64, N).into_array();
            let cmp = canon.binary(rhs, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = PATCH_COUNTS, sample_count = 30)]
fn is_constant_sparse_tree(bencher: Bencher, num_patches: usize) {
    bencher
        .with_inputs(|| {
            (
                build_sparse(N, num_patches),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| is_constant(&col, &mut ctx).unwrap());
}

#[divan::bench(args = PATCH_COUNTS, sample_count = 30)]
fn is_constant_sparse_canonical(bencher: Bencher, num_patches: usize) {
    bencher
        .with_inputs(|| {
            (
                build_sparse(N, num_patches),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            is_constant(&canon, &mut ctx).unwrap()
        });
}

#[divan::bench(args = PATCH_COUNTS, sample_count = 30)]
fn sum_sparse_tree(bencher: Bencher, num_patches: usize) {
    bencher
        .with_inputs(|| {
            (
                build_sparse(N, num_patches),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| sum(&col, &mut ctx).unwrap());
}

#[divan::bench(args = PATCH_COUNTS, sample_count = 30)]
fn sum_sparse_canonical(bencher: Bencher, num_patches: usize) {
    bencher
        .with_inputs(|| {
            (
                build_sparse(N, num_patches),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            sum(&canon, &mut ctx).unwrap()
        });
}
