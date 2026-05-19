// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-iterative execution benches: predicate, `is_constant`, filter on RunEnd-encoded data,
//! across several encoding stacks.
//!
//! Stacks tested:
//! - `Dict(Primitive<i32>[64], RunEnd<u8>)` -- low-cardinality dictionary with run-encoded codes
//! - `RunEnd<i32>` -- pure run-encoding
//!
//! Operations:
//! - **predicate**: `col > 5` -> Bool. Chunked vs flat input.
//! - **is_constant**: aggregate. Bails on first non-constant value through encoding pushdown.
//! - **filter**: apply a Mask to a value column; the RunEnd filter kernel walks runs.
//!
//! Built with `target-cpu=native` to get AVX-512 in the hot paths.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_mask::Mask;
use vortex_runend::RunEnd;

fn main() {
    divan::main();
}

const DICT_SIZE: usize = 64;
const RUN_LENGTH: usize = 10;

// =====================================================================================
// Builders
// =====================================================================================

/// `Dict(values=Primitive<i32>[64], codes=RunEnd(ends<u32>, values<u8>))` for `n` logical rows.
fn build_dict_runend(n: usize) -> vortex_array::ArrayRef {
    build_dict_runend_with_dict_values(n, &(0..DICT_SIZE).map(|i| (i as i32) * 3).collect::<Vec<_>>())
}

/// Same shape, but caller picks the dictionary values - useful for `is_constant` cases.
fn build_dict_runend_with_dict_values(n: usize, dict_values: &[i32]) -> vortex_array::ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    let num_runs = n.div_ceil(RUN_LENGTH);
    let ends: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * RUN_LENGTH, n) as u32)
        .collect();
    let runend_values: Buffer<u8> = (0..num_runs).map(|i| (i % DICT_SIZE) as u8).collect();

    let ends_arr = PrimitiveArray::new(ends, Validity::NonNullable).into_array();
    let runend_values_arr =
        PrimitiveArray::new(runend_values, Validity::NonNullable).into_array();

    let codes = RunEnd::new(ends_arr, runend_values_arr, &mut ctx).into_array();

    let dict_values_buf: Buffer<i32> = dict_values.iter().copied().collect();
    let dict_values_arr = PrimitiveArray::new(dict_values_buf, Validity::NonNullable).into_array();

    DictArray::new(codes, dict_values_arr).into_array()
}

/// Pure `RunEnd<i32>` of length `n` with `n / RUN_LENGTH` runs.
fn build_runend_primitive(n: usize) -> vortex_array::ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let num_runs = n.div_ceil(RUN_LENGTH);
    let ends: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * RUN_LENGTH, n) as u32)
        .collect();
    let values: Buffer<i32> = (0..num_runs).map(|i| (i % DICT_SIZE) as i32 * 3).collect();
    let ends_arr = PrimitiveArray::new(ends, Validity::NonNullable).into_array();
    let values_arr = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    RunEnd::new(ends_arr, values_arr, &mut ctx).into_array()
}

/// `ChunkedArray` of `k` chunks, each `Dict(Primitive<i32>[64], RunEnd<u8>)` covering `n/k` rows.
fn build_chunked_dict_runend(n: usize, k: usize) -> vortex_array::ArrayRef {
    let chunk_size = n / k;
    let chunks: Vec<_> = (0..k).map(|_| build_dict_runend(chunk_size)).collect();
    ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

/// `ChunkedArray` of `k` chunks, each `RunEnd<i32>` covering `n/k` rows.
fn build_chunked_runend_primitive(n: usize, k: usize) -> vortex_array::ArrayRef {
    let chunk_size = n / k;
    let chunks: Vec<_> = (0..k).map(|_| build_runend_primitive(chunk_size)).collect();
    ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap()
    .into_array()
}

// =====================================================================================
// Bench 1 - Predicate `col > 5` across chunkings
// =====================================================================================

const N_TOTAL: usize = 1_000_000;
const CHUNK_COUNTS: &[usize] = &[1, 4, 16, 64];

#[divan::bench(args = CHUNK_COUNTS, sample_count = 30)]
fn predicate_dict_runend_chunked(bencher: Bencher, k: usize) {
    let col = build_chunked_dict_runend(N_TOTAL, k);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let five = ConstantArray::new(5i32, N_TOTAL).into_array();
            let cmp = col.binary(five, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

/// All-in-one: canonicalise (decode every chunk) then compare.
#[divan::bench(args = CHUNK_COUNTS, sample_count = 30)]
fn predicate_dict_runend_chunked_canonical(bencher: Bencher, k: usize) {
    let col = build_chunked_dict_runend(N_TOTAL, k);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            let five = ConstantArray::new(5i32, N_TOTAL).into_array();
            let cmp = canon.binary(five, Operator::Gt).unwrap();
            cmp.execute::<Canonical>(&mut ctx).unwrap()
        });
}

// =====================================================================================
// Bench 2 - is_constant aggregate (bail-early)
// =====================================================================================

/// Truly-constant column (every logical value == 7): is_constant should be near-free via pushdown.
fn build_dict_runend_all_constant(n: usize) -> vortex_array::ArrayRef {
    // Every dict.values entry is 7 -> regardless of code lookup, value is 7.
    build_dict_runend_with_dict_values(n, &vec![7i32; DICT_SIZE])
}

/// Mixed column (values vary): is_constant must traverse - no early bail at top, but pushdown
/// still avoids decoding N rows.
fn build_dict_runend_mixed(n: usize) -> vortex_array::ArrayRef {
    build_dict_runend(n) // values are 3*i, varied
}

#[divan::bench(args = CHUNK_COUNTS, sample_count = 30)]
fn is_constant_all_constant_tree(bencher: Bencher, k: usize) {
    let chunk_size = N_TOTAL / k;
    let chunks: Vec<_> = (0..k)
        .map(|_| build_dict_runend_all_constant(chunk_size))
        .collect();
    let col = ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap()
    .into_array();
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| is_constant(&col, &mut ctx).unwrap());
}

#[divan::bench(args = CHUNK_COUNTS, sample_count = 30)]
fn is_constant_all_constant_canonical(bencher: Bencher, k: usize) {
    let chunk_size = N_TOTAL / k;
    let chunks: Vec<_> = (0..k)
        .map(|_| build_dict_runend_all_constant(chunk_size))
        .collect();
    let col = ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap()
    .into_array();
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            is_constant(&canon, &mut ctx).unwrap()
        });
}

#[divan::bench(sample_count = 30)]
fn is_constant_mixed_tree(bencher: Bencher) {
    let col = build_dict_runend_mixed(N_TOTAL);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| is_constant(&col, &mut ctx).unwrap());
}

#[divan::bench(sample_count = 30)]
fn is_constant_mixed_canonical(bencher: Bencher) {
    let col = build_dict_runend_mixed(N_TOTAL);
    bencher
        .with_inputs(|| (col.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            is_constant(&canon, &mut ctx).unwrap()
        });
}

// =====================================================================================
// Bench 3 - Filter on RunEnd value column with a Mask
// =====================================================================================

/// Build a 50%-selective Mask of length `n`.
fn build_mask_half(n: usize) -> Mask {
    Mask::from_buffer(BitBuffer::from_indices(n, (0..n).step_by(2)))
}

/// Build a 1%-selective Mask of length `n`.
fn build_mask_sparse(n: usize) -> Mask {
    Mask::from_buffer(BitBuffer::from_indices(n, (0..n).step_by(100)))
}

/// `RunEnd<i32>` with explicit run length `r` rather than the constant `RUN_LENGTH`.
fn build_runend_primitive_with_r(n: usize, r: usize) -> vortex_array::ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let num_runs = n.div_ceil(r);
    let ends: Buffer<u32> = (1..=num_runs)
        .map(|i| std::cmp::min(i * r, n) as u32)
        .collect();
    let values: Buffer<i32> = (0..num_runs).map(|i| (i % DICT_SIZE) as i32 * 3).collect();
    let ends_arr = PrimitiveArray::new(ends, Validity::NonNullable).into_array();
    let values_arr = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    RunEnd::new(ends_arr, values_arr, &mut ctx).into_array()
}

const RUN_LENGTHS: &[usize] = &[10, 100, 1000];

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn filter_runend_tree_half(bencher: Bencher, r: usize) {
    let col = build_runend_primitive_with_r(N_TOTAL, r);
    let mask = build_mask_half(N_TOTAL);
    bencher
        .with_inputs(|| (col.clone(), mask.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mask, mut ctx)| {
            let filtered = col.filter(mask).unwrap();
            filtered.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn filter_runend_canonical_half(bencher: Bencher, r: usize) {
    let col = build_runend_primitive_with_r(N_TOTAL, r);
    let mask = build_mask_half(N_TOTAL);
    bencher
        .with_inputs(|| (col.clone(), mask.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mask, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            let filtered = canon.filter(mask).unwrap();
            filtered.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn filter_runend_tree_sparse(bencher: Bencher, r: usize) {
    let col = build_runend_primitive_with_r(N_TOTAL, r);
    let mask = build_mask_sparse(N_TOTAL);
    bencher
        .with_inputs(|| (col.clone(), mask.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mask, mut ctx)| {
            let filtered = col.filter(mask).unwrap();
            filtered.execute::<Canonical>(&mut ctx).unwrap()
        });
}

#[divan::bench(args = RUN_LENGTHS, sample_count = 30)]
fn filter_runend_canonical_sparse(bencher: Bencher, r: usize) {
    let col = build_runend_primitive_with_r(N_TOTAL, r);
    let mask = build_mask_sparse(N_TOTAL);
    bencher
        .with_inputs(|| (col.clone(), mask.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mask, mut ctx)| {
            let canon = col.execute::<Canonical>(&mut ctx).unwrap().into_array();
            let filtered = canon.filter(mask).unwrap();
            filtered.execute::<Canonical>(&mut ctx).unwrap()
        });
}

// =====================================================================================
// Bench 4 - Filter on chunked RunEnd<i32>
// =====================================================================================

#[divan::bench(args = CHUNK_COUNTS, sample_count = 30)]
fn filter_chunked_runend(bencher: Bencher, k: usize) {
    let col = build_chunked_runend_primitive(N_TOTAL, k);
    let mask = build_mask_half(N_TOTAL);
    bencher
        .with_inputs(|| (col.clone(), mask.clone(), LEGACY_SESSION.create_execution_ctx()))
        .bench_values(|(col, mask, mut ctx)| {
            let filtered = col.filter(mask).unwrap();
            filtered.execute::<Canonical>(&mut ctx).unwrap()
        });
}
